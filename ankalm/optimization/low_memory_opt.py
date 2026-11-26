"""Low-memory optimizers for AnkaLM: Block-Quantized 8-bit AdamW and Factored Adafactor (Section 3.1)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from ankalm.core.config import ModelProfileConfig
from ankalm.core.constants import PROJECT_ROOT
from ankalm.optimization.scheduler import build_adamw_optimizer


class AdamW8bit(torch.optim.Optimizer):
    """
    Block-Quantized 8-bit State AdamW Optimizer.
    Stores first moment `m` (int8) and second moment `v` (uint8) with per-tensor FP32 scales,
    reducing optimizer state memory from 8 bytes/param to ~2 bytes/param (75% memory reduction).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | List[Dict[str, Any]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.08,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.float()
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["q_exp_avg"] = torch.zeros_like(p.data, dtype=torch.int8)
                    state["scale_m"] = torch.tensor(1e-7, device=p.device, dtype=torch.float32)
                    state["q_exp_avg_sq"] = torch.zeros_like(p.data, dtype=torch.uint8)
                    state["scale_v"] = torch.tensor(1e-7, device=p.device, dtype=torch.float32)

                state["step"] += 1
                step_t = state["step"]

                if wd != 0.0 and p.dim() >= 2:
                    p.data.mul_(1.0 - lr * wd)

                # Dequantize 8-bit states (q_exp_avg stores int8 m; q_exp_avg_sq stores uint8 sqrt(v))
                m = state["q_exp_avg"].float() * state["scale_m"]
                sqrt_v = state["q_exp_avg_sq"].float() * state["scale_v"]
                v = sqrt_v.pow(2)

                # Update moments in FP32
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - (beta1 ** step_t)
                bias_correction2 = 1.0 - (beta2 ** step_t)
                step_size = lr / bias_correction1
                denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                p.data.add_((m / denom).to(p.dtype), alpha=-step_size)

                # Re-quantize m (int8) and sqrt(v) (uint8) for high dynamic range accuracy
                max_m = m.abs().max().clamp(min=1e-12)
                scale_m = max_m / 127.0
                state["q_exp_avg"].copy_(torch.clamp(torch.round(m / scale_m), -127, 127).to(torch.int8))
                state["scale_m"].copy_(scale_m)

                new_sqrt_v = v.sqrt()
                max_sv = new_sqrt_v.max().clamp(min=1e-12)
                scale_v = max_sv / 255.0
                state["q_exp_avg_sq"].copy_(torch.clamp(torch.round(new_sqrt_v / scale_v), 0, 255).to(torch.uint8))
                state["scale_v"].copy_(scale_v)

        return loss


class Adafactor(torch.optim.Optimizer):
    """
    Factored Second-Moment Low-Memory Optimizer (Shazeer & Stern 2018).
    For 2D weight tensors of shape (R, C), stores only row factors (R,) and col factors (C,)
    instead of full (R, C) second-moment matrices, achieving >95% second-moment memory reduction.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | List[Dict[str, Any]],
        lr: float = 1e-3,
        beta2_decay: float = -0.8,
        eps1: float = 1e-30,
        clip_threshold: float = 1.0,
        weight_decay: float = 0.05,
    ) -> None:
        defaults = dict(
            lr=lr,
            beta2_decay=beta2_decay,
            eps1=eps1,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta2_decay = group["beta2_decay"]
            eps1 = group["eps1"]
            d_clip = group["clip_threshold"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.float()
                grad_sq = grad.pow(2).add_(eps1)
                state = self.state[p]
                factored = p.dim() == 2

                if len(state) == 0:
                    state["step"] = 0
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(p.size(0), device=p.device, dtype=torch.float32)
                        state["exp_avg_sq_col"] = torch.zeros(p.size(1), device=p.device, dtype=torch.float32)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(p.data, dtype=torch.float32)

                state["step"] += 1
                step_t = state["step"]
                beta2_t = 1.0 - (step_t ** beta2_decay)

                if wd != 0.0 and p.dim() >= 2:
                    p.data.mul_(1.0 - lr * wd)

                if factored:
                    r_factor = state["exp_avg_sq_row"]
                    c_factor = state["exp_avg_sq_col"]
                    r_factor.mul_(beta2_t).add_(grad_sq.mean(dim=1), alpha=1.0 - beta2_t)
                    c_factor.mul_(beta2_t).add_(grad_sq.mean(dim=0), alpha=1.0 - beta2_t)
                    r_Mean = r_factor.mean().clamp(min=1e-12)
                    approx_v = torch.outer(r_factor / r_Mean, c_factor)
                    update = grad * torch.rsqrt(approx_v.clamp(min=1e-12))
                else:
                    v = state["exp_avg_sq"]
                    v.mul_(beta2_t).add_(grad_sq, alpha=1.0 - beta2_t)
                    update = grad * torch.rsqrt(v.clamp(min=1e-12))

                # Update RMS clipping
                rms_u = update.pow(2).mean().sqrt().clamp(min=1e-12)
                update.div_(max(1.0, float(rms_u.item()) / d_clip))
                p.data.add_(update.to(p.dtype), alpha=-lr)

        return loss


def measure_optimizer_state_mb(optimizer: torch.optim.Optimizer) -> float:
    total_bytes = 0
    for state in optimizer.state.values():
        for val in state.values():
            if isinstance(val, torch.Tensor):
                total_bytes += val.numel() * val.element_size()
    return round(total_bytes / (1024 * 1024), 4)


def build_configured_optimizer(
    model: nn.Module,
    optimizer_type: str,
    lr: float,
    weight_decay: float,
    beta1: float = 0.9,
    beta2: float = 0.95,
) -> torch.optim.Optimizer:
    opt_key = optimizer_type.lower()
    if opt_key == "adamw_8bit":
        params = [p for p in model.parameters() if p.requires_grad]
        return AdamW8bit(params, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    if opt_key == "adafactor":
        params = [p for p in model.parameters() if p.requires_grad]
        return Adafactor(params, lr=lr, weight_decay=weight_decay)
    return build_adamw_optimizer(model, lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2)


@dataclass
class OptimizerBenchmarkEntry:
    optimizer_type: str
    model_name: str
    parameter_count: int
    optimizer_state_mb: float
    memory_reduction_vs_adamw_pct: float
    initial_loss: float
    final_loss: float
    loss_reduced: bool
    tokens_per_second: float


def benchmark_low_memory_optimizers(model_cfg: ModelProfileConfig, steps: int = 8) -> List[OptimizerBenchmarkEntry]:
    """Measure actual optimizer state memory, throughput, and loss convergence across AdamW, AdamW8bit, and Adafactor."""
    from ankalm.model import build_ankalm_model

    torch.manual_seed(42)
    x = torch.randint(0, model_cfg.vocab_size, (8, 64))
    y = torch.randint(0, model_cfg.vocab_size, (8, 64))

    results: List[OptimizerBenchmarkEntry] = []
    adamw_state_mb = 0.0

    for opt_type in ("adamw", "adamw_8bit", "adafactor"):
        torch.manual_seed(42)
        model = build_ankalm_model(model_cfg)
        lr = 2e-3 if opt_type == "adafactor" else 1e-3
        opt = build_configured_optimizer(model, opt_type, lr=lr, weight_decay=0.05)

        init_loss = 0.0
        final_loss = 0.0
        t0 = time.perf_counter()
        for s in range(steps):
            opt.zero_grad(set_to_none=True)
            _, loss, _ = model(x, targets=y)
            if s == 0:
                init_loss = float(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            final_loss = float(loss.item())
        dt = max(time.perf_counter() - t0, 1e-6)
        tps = (steps * x.numel()) / dt

        state_mb = measure_optimizer_state_mb(opt)
        if opt_type == "adamw":
            adamw_state_mb = max(state_mb, 1e-6)
        red_pct = round((1.0 - (state_mb / max(adamw_state_mb, 1e-6))) * 100.0, 2)

        results.append(
            OptimizerBenchmarkEntry(
                optimizer_type=opt_type,
                model_name=model_cfg.name,
                parameter_count=model.count_parameters(),
                optimizer_state_mb=state_mb,
                memory_reduction_vs_adamw_pct=max(0.0, red_pct),
                initial_loss=round(init_loss, 4),
                final_loss=round(final_loss, 4),
                loss_reduced=final_loss < init_loss,
                tokens_per_second=round(tps, 1),
            )
        )

    out_file = PROJECT_ROOT / "logs" / "optimizer_benchmarks.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    return results
