"""Model allocation, forward, backward, optimizer, and throughput benchmark utilities."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import time
from typing import Any, Dict

import psutil
import os
import torch
import torch.nn as nn

from ankalm.core.config import ModelProfileConfig, TrainingProfileConfig


@dataclass
class ModelBenchmarkResult:
    profile: str
    model_name: str
    parameter_count: int
    device: str
    precision: str
    batch_size: int
    seq_len: int
    allocation_ms: float
    forward_ms: float
    backward_ms: float
    optimizer_ms: float
    step_total_ms: float
    tokens_per_second: float
    peak_ram_mb: float


def benchmark_model_profile(
    model_module: nn.Module,
    model_cfg: ModelProfileConfig,
    train_cfg: TrainingProfileConfig,
    device_str: str = "cpu",
    warmup_iters: int = 2,
    bench_iters: int = 5,
) -> ModelBenchmarkResult:
    """Measure allocation, forward, backward, optimizer update, and tokens/sec for a given model."""
    proc = psutil.Process(os.getpid())
    device = torch.device(device_str)
    t_alloc0 = time.perf_counter()
    model = model_module.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)
    alloc_ms = (time.perf_counter() - t_alloc0) * 1000.0

    param_count = sum(p.numel() for p in model.parameters())
    b = min(train_cfg.batch_size, 8)
    s = model_cfg.max_seq_len
    x = torch.randint(0, model_cfg.vocab_size, (b, s), device=device)
    y = torch.randint(0, model_cfg.vocab_size, (b, s), device=device)

    use_amp = train_cfg.precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if train_cfg.precision == "bf16" else torch.float16

    for _ in range(warmup_iters):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits, loss, _ = model(x, targets=y)
        loss.backward()
        optimizer.step()

    fwd_times = []
    bwd_times = []
    opt_times = []
    for _ in range(bench_iters):
        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits, loss, _ = model(x, targets=y)
        t1 = time.perf_counter()
        loss.backward()
        t2 = time.perf_counter()
        optimizer.step()
        t3 = time.perf_counter()

        fwd_times.append((t1 - t0) * 1000.0)
        bwd_times.append((t2 - t1) * 1000.0)
        opt_times.append((t3 - t2) * 1000.0)

    avg_fwd = sum(fwd_times) / len(fwd_times)
    avg_bwd = sum(bwd_times) / len(bwd_times)
    avg_opt = sum(opt_times) / len(opt_times)
    total_ms = avg_fwd + avg_bwd + avg_opt
    tokens_per_step = b * s
    tps = (tokens_per_step / (total_ms / 1000.0)) if total_ms > 0 else 0.0
    peak_ram_mb = proc.memory_info().rss / (1024 * 1024)

    return ModelBenchmarkResult(
        profile=model_cfg.profile,
        model_name=model_cfg.name,
        parameter_count=param_count,
        device=device_str,
        precision=train_cfg.precision,
        batch_size=b,
        seq_len=s,
        allocation_ms=round(alloc_ms, 2),
        forward_ms=round(avg_fwd, 2),
        backward_ms=round(avg_bwd, 2),
        optimizer_ms=round(avg_opt, 2),
        step_total_ms=round(total_ms, 2),
        tokens_per_second=round(tps, 1),
        peak_ram_mb=round(peak_ram_mb, 2),
    )
