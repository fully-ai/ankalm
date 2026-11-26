"""Knowledge Distillation Engine: Distill AnkaLM's own trained Teacher into AnkaLM-Edge Student (Section 14)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, Optional, Tuple

import psutil
import torch
import torch.nn.functional as F

from ankalm.core.config import AnkaLMConfig, ModelProfileConfig, PREDEFINED_MODEL_PROFILES
from ankalm.core.constants import PROJECT_ROOT
from ankalm.data.dataset import PackedTokenDataset
from ankalm.model import build_ankalm_model
from ankalm.optimization.scheduler import CosineWarmupScheduler, build_adamw_optimizer, check_gradient_health
from ankalm.storage.checkpoint import CheckpointManager
from ankalm.tokenizer.tokenizer import AnkaLMTokenizer
from ankalm.utils.logging import get_logger

logger = get_logger("ankalm.distillation")


@dataclass
class DistillationRunResult:
    teacher_checkpoint: str
    teacher_model_name: str
    teacher_parameters: int
    teacher_val_loss: float
    teacher_val_perplexity: float
    student_checkpoint: str
    student_model_name: str
    student_parameters: int
    compression_ratio: float
    steps_executed: int
    processed_tokens: int
    temperature: float
    alpha_kd: float
    initial_student_loss: float
    final_student_loss: float
    student_val_loss: float
    student_val_perplexity: float
    capability_retention_ratio: float
    avg_tokens_per_second: float
    param_l2_delta_from_init: float
    weights_updated_verified: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnkaLMDistillationTrainer:
    """
    Controlled Knowledge Distillation pipeline from AnkaLM's own trained Teacher
    (e.g. AnkaLM-Small) into a compact AnkaLM-Edge Student for <2GB VRAM & fast draft decoding.
    Never uses external third-party teacher models.
    """

    def __init__(
        self,
        student_config: Optional[AnkaLMConfig] = None,
        tokenizer: Optional[AnkaLMTokenizer] = None,
        checkpoints_dir: Optional[Path] = None,
    ) -> None:
        if student_config is None:
            edge_yaml = PROJECT_ROOT / "configs" / "edge_2gb.yaml"
            if edge_yaml.exists():
                student_config = AnkaLMConfig.from_yaml(edge_yaml)
            else:
                student_config = AnkaLMConfig(model=PREDEFINED_MODEL_PROFILES["edge"])
        self.config = student_config
        self.paths = self.config.storage.resolve(PROJECT_ROOT)
        self.checkpoints_dir = checkpoints_dir or self.paths["checkpoints_dir"]
        self.ckpt_manager = CheckpointManager(self.checkpoints_dir)
        self.tokenizer = tokenizer or AnkaLMTokenizer.load(self.paths["tokenizer_dir"])
        self.config.model.vocab_size = self.tokenizer.vocab_size
        self.device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)
        if self.device.type == "cpu":
            phys = psutil.cpu_count(logical=False) or 4
            torch.set_num_threads(min(phys, 16))

    @torch.no_grad()
    def _eval_loss(self, model: torch.nn.Module, val_ds: PackedTokenDataset, max_batches: int = 8) -> Tuple[float, float]:
        model.eval()
        total = 0.0
        cnt = 0
        for i in range(min(len(val_ds), max_batches)):
            x, y = val_ds[i]
            _, loss, _ = model(x.unsqueeze(0).to(self.device), targets=y.unsqueeze(0).to(self.device))
            if loss is not None and torch.isfinite(loss):
                total += float(loss.item())
                cnt += 1
        model.train()
        avg = total / max(cnt, 1)
        return (round(avg, 5), round(math.exp(min(avg, 20.0)), 3))

    def distill(
        self,
        teacher_checkpoint_name: str = "small_final",
        student_slot_name: str = "edge_final",
        steps: int = 100,
        temperature: float = 2.0,
        alpha_kd: float = 0.5,
        learning_rate: float = 1.2e-3,
    ) -> DistillationRunResult:
        # 1. Load AnkaLM's own genuinely trained teacher checkpoint
        t_meta, t_payload = self.ckpt_manager.load_checkpoint(teacher_checkpoint_name, restore_rng=False, device=self.device_str)
        if t_meta.maturity_level == "random_init" or t_meta.global_step == 0:
            raise RuntimeError(f"Teacher checkpoint '{teacher_checkpoint_name}' is not trained!")

        t_cfg = ModelProfileConfig(**t_meta.model_config)
        teacher = build_ankalm_model(t_cfg).to(self.device)
        teacher.load_state_dict(t_payload["model_state_dict"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # 2. Initialize AnkaLM-Edge student from random initialization
        torch.manual_seed(self.config.training.seed)
        student = build_ankalm_model(self.config.model).to(self.device)
        init_snapshot = student.snapshot_weights()
        init_fp = student.compute_parameter_fingerprint()

        seq_len = min(self.config.model.max_seq_len, t_cfg.max_seq_len)
        train_jsonl = self.paths["processed_data_dir"] / "train.jsonl"
        val_jsonl = self.paths["validation_data_dir"] / "val.jsonl"
        train_ds = PackedTokenDataset.from_jsonl(train_jsonl, encode_fn=lambda s: self.tokenizer.encode(s), seq_len=seq_len, stride=seq_len // 2)
        val_ds = PackedTokenDataset.from_jsonl(val_jsonl, encode_fn=lambda s: self.tokenizer.encode(s), seq_len=seq_len, stride=seq_len)

        t_val_loss, t_val_ppl = self._eval_loss(teacher, val_ds)

        optimizer = build_adamw_optimizer(student, lr=learning_rate, weight_decay=0.05)
        scheduler = CosineWarmupScheduler(
            optimizer,
            max_lr=learning_rate,
            min_lr=learning_rate * 0.1,
            warmup_steps=max(5, steps // 8),
            max_steps=steps,
        )

        use_amp = self.config.training.precision in ("bf16", "fp16")
        amp_dtype = torch.bfloat16 if self.config.training.precision == "bf16" else torch.float16
        bsz = min(self.config.training.batch_size, 12)
        cursor = 0
        order = list(range(len(train_ds)))
        random.shuffle(order)

        processed_tokens = 0
        init_loss: Optional[float] = None
        last_loss = 0.0
        t0_all = time.perf_counter()

        student.train()
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            xs, ys = [], []
            for _ in range(bsz):
                if cursor >= len(order):
                    cursor = 0
                    random.shuffle(order)
                x, y = train_ds[order[cursor]]
                cursor += 1
                xs.append(x)
                ys.append(y)
            xb = torch.stack(xs, dim=0).to(self.device)
            yb = torch.stack(ys, dim=0).to(self.device)

            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=use_amp):
                    t_logits, _, _ = teacher(xb)

            with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=use_amp):
                s_logits, ce_loss, _ = student(xb, targets=yb)
                # Soft-target KL divergence loss at temperature T
                s_log_probs = F.log_softmax(s_logits.float() / temperature, dim=-1)
                t_probs = F.softmax(t_logits.float() / temperature, dim=-1)
                kd_loss = F.kl_div(
                    s_log_probs.view(-1, s_logits.size(-1)),
                    t_probs.view(-1, t_logits.size(-1)),
                    reduction="batchmean",
                ) * (temperature ** 2)
                loss = ((1.0 - alpha_kd) * ce_loss) + (alpha_kd * kd_loss)

            loss.backward()
            check_gradient_health(student, max_norm=1.0)
            scheduler.step(step - 1)
            optimizer.step()

            last_loss = float(loss.item())
            if init_loss is None:
                init_loss = last_loss
            processed_tokens += int(xb.numel())

            if step % 25 == 0 or step == steps:
                sv_loss, sv_ppl = self._eval_loss(student, val_ds)
                logger.info(
                    "[Distill %s -> %s] step %d/%d | kd_total_loss=%.4f | student_val_loss=%.4f | student_ppl=%.2f (teacher_ppl=%.2f)",
                    t_cfg.name,
                    self.config.model.name,
                    step,
                    steps,
                    last_loss,
                    sv_loss,
                    sv_ppl,
                    t_val_ppl,
                )

        total_dt = max(time.perf_counter() - t0_all, 1e-6)
        avg_tps = processed_tokens / total_dt
        s_val_loss, s_val_ppl = self._eval_loss(student, val_ds)
        l2_delta = student.compute_l2_delta(init_snapshot)
        final_fp = student.compute_parameter_fingerprint()

        # Capability retention ratio measured via inverse validation NLL ratio (teacher_val_loss / student_val_loss)
        retention = round(min(1.0, t_val_loss / max(s_val_loss, 1e-5)), 4)

        tok_meta = {
            "version": self.tokenizer.VERSION,
            "vocab_size": self.tokenizer.vocab_size,
            "vocab_fingerprint": self.tokenizer.vocab_fingerprint(),
        }
        saved_dir = self.ckpt_manager.save_checkpoint(
            slot_name=student_slot_name,
            model=student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            config=self.config,
            global_step=steps,
            epoch=steps * bsz // max(len(train_ds), 1),
            processed_tokens=processed_tokens,
            train_loss=last_loss,
            val_loss=s_val_loss,
            val_perplexity=s_val_ppl,
            initial_fingerprint=init_fp,
            param_l2_delta=l2_delta,
            tokenizer_meta=tok_meta,
            dataset_manifest_summary={
                "distilled_from_teacher": t_meta.checkpoint_name,
                "teacher_model": t_meta.model_name,
                "temperature": temperature,
                "alpha_kd": alpha_kd,
            },
            dataset_sampling_state={"cursor": cursor},
            maturity_level="distilled_edge",
            update_latest_and_best=False,
        )

        t_params = teacher.count_parameters(trainable_only=False)
        s_params = student.count_parameters(trainable_only=False)
        result = DistillationRunResult(
            teacher_checkpoint=t_meta.checkpoint_name,
            teacher_model_name=t_meta.model_name,
            teacher_parameters=t_params,
            teacher_val_loss=t_val_loss,
            teacher_val_perplexity=t_val_ppl,
            student_checkpoint=str(saved_dir),
            student_model_name=self.config.model.name,
            student_parameters=s_params,
            compression_ratio=round(t_params / max(s_params, 1), 2),
            steps_executed=steps,
            processed_tokens=processed_tokens,
            temperature=temperature,
            alpha_kd=alpha_kd,
            initial_student_loss=round(init_loss or last_loss, 5),
            final_student_loss=round(last_loss, 5),
            student_val_loss=s_val_loss,
            student_val_perplexity=s_val_ppl,
            capability_retention_ratio=retention,
            avg_tokens_per_second=round(avg_tps, 1),
            param_l2_delta_from_init=round(l2_delta, 6),
            weights_updated_verified=(final_fp != init_fp) and (l2_delta > 0.0),
        )
        with open(PROJECT_ROOT / "logs" / "distillation_result.json", "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        return result
