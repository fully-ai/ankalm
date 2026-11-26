"""Reproducible INT8, INT4, BF16, and FP16 quantization with memory, speed, and perplexity benchmarks."""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

from ankalm.core.config import ModelProfileConfig
from ankalm.core.constants import PROJECT_ROOT
from ankalm.data.dataset import PackedTokenDataset
from ankalm.model.transformer import AnkaLMForCausalLM
from ankalm.storage.checkpoint import CheckpointManager
from ankalm.tokenizer.tokenizer import AnkaLMTokenizer


class Int8WeightOnlyLinear(nn.Module):
    """Per-channel symmetric INT8 weight-only quantized Linear layer."""

    def __init__(self, in_features: int, out_features: int, weight_fp: torch.Tensor) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        with torch.no_grad():
            w = weight_fp.detach().float()
            max_abs = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            scale = max_abs / 127.0
            q_weight = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
        self.register_buffer("q_weight", q_weight)
        self.register_buffer("scale", scale.to(torch.float32))

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return (self.q_weight.float() * self.scale).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize(x.dtype)
        return F.linear(x, w, None)


class Int4WeightOnlyLinear(nn.Module):
    """Grouped symmetric 4-bit (INT4 packed into uint8 pairs) weight-only quantized Linear layer."""

    def __init__(self, in_features: int, out_features: int, weight_fp: torch.Tensor, group_size: int = 32) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        with torch.no_grad():
            w = weight_fp.detach().float()
            flat = w.reshape(-1)
            pad_len = (group_size - (flat.numel() % group_size)) % group_size
            if pad_len > 0:
                flat = F.pad(flat, (0, pad_len))
            grouped = flat.view(-1, group_size)
            max_abs = grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            scale = max_abs / 7.0
            # Map [-7, 7] -> [1, 15] unsigned 4-bit nibbles (offset +8)
            q_nibbles = (torch.clamp(torch.round(grouped / scale), -7, 7).to(torch.int16) + 8).to(torch.uint8)
            flat_nibbles = q_nibbles.view(-1)
            if flat_nibbles.numel() % 2 != 0:
                flat_nibbles = F.pad(flat_nibbles, (0, 1), value=8)
            high = flat_nibbles[0::2] << 4
            low = flat_nibbles[1::2] & 0x0F
            packed = (high | low).contiguous()

        self.total_padded = grouped.numel()
        self.orig_shape = w.shape
        self.register_buffer("packed_weight", packed)
        self.register_buffer("scale", scale.to(torch.float16))

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        high = ((self.packed_weight >> 4) & 0x0F).to(torch.int16) - 8
        low = (self.packed_weight & 0x0F).to(torch.int16) - 8
        unpacked = torch.stack([high, low], dim=-1).view(-1)[: self.total_padded]
        grouped = unpacked.view(-1, self.group_size).float() * self.scale.float()
        w = grouped.view(-1)[: self.out_features * self.in_features].view(self.orig_shape)
        return w.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize(x.dtype)
        return F.linear(x, w, None)


def _quantize_ffn_module_int8(ffn: nn.Module) -> None:
    for attr in ("gate_proj", "up_proj", "down_proj"):
        if hasattr(ffn, attr):
            lin = getattr(ffn, attr)
            setattr(ffn, attr, Int8WeightOnlyLinear(lin.in_features, lin.out_features, lin.weight))


def _quantize_ffn_module_int4(ffn: nn.Module, group_size: int = 32) -> None:
    for attr in ("gate_proj", "up_proj", "down_proj"):
        if hasattr(ffn, attr):
            lin = getattr(ffn, attr)
            setattr(ffn, attr, Int4WeightOnlyLinear(lin.in_features, lin.out_features, lin.weight, group_size=group_size))


def apply_int8_dynamic_quantization(model: nn.Module) -> nn.Module:
    """Replace internal Transformer Block Linear layers with INT8 weight-only quantized layers (Dense & MoE)."""
    q_model = copy.deepcopy(model)
    for layer in q_model.layers:
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = getattr(layer.attn, attr)
            setattr(layer.attn, attr, Int8WeightOnlyLinear(lin.in_features, lin.out_features, lin.weight))
        if hasattr(layer, "ffn"):
            _quantize_ffn_module_int8(layer.ffn)
        elif hasattr(layer, "moe"):
            if layer.moe.shared_expert is not None:
                _quantize_ffn_module_int8(layer.moe.shared_expert)
            for exp in layer.moe.routed_experts:
                _quantize_ffn_module_int8(exp)
    return q_model


def apply_int4_weight_quantization(model: nn.Module, group_size: int = 32) -> nn.Module:
    """Replace internal Transformer Block Linear layers with packed INT4 weight-only quantized layers (Dense & MoE)."""
    q_model = copy.deepcopy(model)
    for layer in q_model.layers:
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = getattr(layer.attn, attr)
            setattr(
                layer.attn,
                attr,
                Int4WeightOnlyLinear(lin.in_features, lin.out_features, lin.weight, group_size=group_size),
            )
        if hasattr(layer, "ffn"):
            _quantize_ffn_module_int4(layer.ffn, group_size=group_size)
        elif hasattr(layer, "moe"):
            if layer.moe.shared_expert is not None:
                _quantize_ffn_module_int4(layer.moe.shared_expert, group_size=group_size)
            for exp in layer.moe.routed_experts:
                _quantize_ffn_module_int4(exp, group_size=group_size)
    return q_model


def compute_model_storage_mb(model: nn.Module) -> float:
    seen_ptrs = set()
    total_bytes = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.data_ptr() not in seen_ptrs:
            seen_ptrs.add(tensor.data_ptr())
            total_bytes += tensor.numel() * tensor.element_size()
    return round(total_bytes / (1024 * 1024), 3)


@dataclass
class QuantizationBenchmarkEntry:
    precision: str
    model_storage_mb: float
    compression_ratio: float
    ram_usage_mb: float
    vram_usage_mb: float
    load_time_ms: float
    inference_tokens_per_sec: float
    validation_loss: float
    validation_perplexity: float
    sample_output: str
    export_path: str


def run_quantization_suite(
    checkpoint_name: str = "latest",
    checkpoints_dir: Optional[Path] = None,
    exports_dir: Optional[Path] = None,
    tokenizer_dir: Optional[Path] = None,
) -> List[QuantizationBenchmarkEntry]:
    """
    Quantize a trained checkpoint into FP32, BF16, FP16, INT8, and INT4 without modifying
    the original checkpoint, measure memory/speed/perplexity, and save quantized exports.
    """
    ckpt_mgr = CheckpointManager(checkpoints_dir)
    exp_dir = exports_dir or (PROJECT_ROOT / "exports")
    exp_dir.mkdir(parents=True, exist_ok=True)
    tok = AnkaLMTokenizer.load(tokenizer_dir)

    meta, payload = ckpt_mgr.load_checkpoint(checkpoint_name, restore_rng=False, device="cpu")
    m_cfg = ModelProfileConfig(**meta.model_config)
    base_model = AnkaLMForCausalLM(m_cfg)
    base_model.load_state_dict(payload["model_state_dict"])
    base_model.eval()
    fp32_mb = compute_model_storage_mb(base_model)

    val_jsonl = PROJECT_ROOT / "datasets" / "validation" / "val.jsonl"
    val_ds = PackedTokenDataset.from_jsonl(val_jsonl, encode_fn=lambda s: tok.encode(s), seq_len=min(128, m_cfg.max_seq_len))
    xb, yb = val_ds[0]
    xb = xb.unsqueeze(0)
    yb = yb.unsqueeze(0)

    prompt = "Türkiye ve Türkçe doğal dil işleme"
    prompt_ids = torch.tensor([tok.encode(prompt, add_bos=True)], dtype=torch.long)

    proc = psutil.Process(os.getpid())
    results: List[QuantizationBenchmarkEntry] = []

    for mode in ("fp32", "bf16", "fp16", "int8", "int4"):
        t_load0 = time.perf_counter()
        if mode == "fp32":
            q_model = copy.deepcopy(base_model).float()
        elif mode == "bf16":
            q_model = copy.deepcopy(base_model).to(torch.bfloat16)
        elif mode == "fp16":
            q_model = copy.deepcopy(base_model).to(torch.float16)
        elif mode == "int8":
            q_model = apply_int8_dynamic_quantization(base_model)
        elif mode == "int4":
            q_model = apply_int4_weight_quantization(base_model, group_size=32)
        else:
            continue
        q_model.eval()
        load_ms = (time.perf_counter() - t_load0) * 1000.0

        storage_mb = compute_model_storage_mb(q_model)
        comp_ratio = round(fp32_mb / max(storage_mb, 1e-4), 2)

        with torch.no_grad():
            _, loss, _ = q_model(xb, targets=yb)
            v_loss = float(loss.item()) if loss is not None else 10.0
            v_ppl = math.exp(min(v_loss, 20.0))

            # Measure autoregressive generation speed (24 tokens with KV cache)
            t_gen0 = time.perf_counter()
            curr = prompt_ids.clone()
            past_kv = None
            gen_ids: List[int] = []
            for _ in range(24):
                logits, _, past_kv = q_model(curr, past_key_values=past_kv, use_cache=True)
                next_tok = int(torch.argmax(logits[0, -1, :].float()).item())
                gen_ids.append(next_tok)
                curr = torch.tensor([[next_tok]], dtype=torch.long)
            dt_gen = max(time.perf_counter() - t_gen0, 1e-6)
            tps = len(gen_ids) / dt_gen
            sample_str = tok.decode(gen_ids, skip_special_tokens=True)

        mode_dir = exp_dir / f"quantized_{mode}"
        mode_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"mode": mode, "model_config": asdict(m_cfg), "state_dict": q_model.state_dict(), "source_checkpoint": meta.checkpoint_name},
            mode_dir / "quantized_model.pt",
        )
        ram_mb = proc.memory_info().rss / (1024 * 1024)

        results.append(
            QuantizationBenchmarkEntry(
                precision=mode.upper(),
                model_storage_mb=storage_mb,
                compression_ratio=comp_ratio,
                ram_usage_mb=round(ram_mb, 2),
                vram_usage_mb=0.0,
                load_time_ms=round(load_ms, 2),
                inference_tokens_per_sec=round(tps, 1),
                validation_loss=round(v_loss, 5),
                validation_perplexity=round(v_ppl, 3),
                sample_output=sample_str[:160],
                export_path=str(mode_dir / "quantized_model.pt"),
            )
        )

    report_path = exp_dir / "quantization_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    return results
