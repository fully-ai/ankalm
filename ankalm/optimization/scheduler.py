"""Cosine learning rate scheduler with linear warmup and AdamW parameter group builder."""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class CosineWarmupScheduler:
    """Stateful learning rate scheduler with linear warmup and cosine decay to min_lr."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        max_steps: int,
    ) -> None:
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = max(0, warmup_steps)
        self.max_steps = max(1, max_steps)
        self.current_step = 0
        self.current_lr = self.get_lr(0)

    def get_lr(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.max_lr * ((step + 1) / self.warmup_steps)
        if step >= self.max_steps:
            return self.min_lr
        decay_Span = max(1, self.max_steps - self.warmup_steps)
        progress = (step - self.warmup_steps) / decay_Span
        coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def step(self, step_num: int | None = None) -> float:
        if step_num is not None:
            self.current_step = step_num
        else:
            self.current_step += 1
        lr = self.get_lr(self.current_step)
        self.current_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def state_dict(self) -> Dict[str, Any]:
        return {
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "current_step": self.current_step,
            "current_lr": self.current_lr,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.max_lr = float(state.get("max_lr", self.max_lr))
        self.min_lr = float(state.get("min_lr", self.min_lr))
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        self.max_steps = int(state.get("max_steps", self.max_steps))
        self.current_step = int(state.get("current_step", 0))
        self.current_lr = float(state.get("current_lr", self.get_lr(self.current_step)))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.current_lr


def build_adamw_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    beta1: float = 0.9,
    beta2: float = 0.95,
) -> torch.optim.AdamW:
    """Separate 2D weight tensors (decayed) from 1D RMSNorm/bias tensors (non-decayed)."""
    decay_params = []
    no_decay_params = []
    seen_ptrs = set()
    for _, param in model.named_parameters():
        if not param.requires_grad or param.data_ptr() in seen_ptrs:
            continue
        seen_ptrs.add(param.data_ptr())
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(beta1, beta2), eps=1e-8)


def check_gradient_health(model: nn.Module, max_norm: float = 1.0) -> Tuple[float, bool]:
    """Clip gradients and check for NaN/Inf gradients. Returns (grad_norm, is_healthy)."""
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    norm_val = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
    is_healthy = math.isfinite(norm_val)
    return (round(norm_val, 4), is_healthy)
