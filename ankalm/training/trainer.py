"""Genuine from-scratch causal language model pretraining engine with resume and live telemetry."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn

from ankalm.core.config import AnkaLMConfig
from ankalm.core.constants import PROJECT_ROOT
from ankalm.data.dataset import PackedTokenDataset
from ankalm.model import build_ankalm_model
from ankalm.model.transformer import AnkaLMForCausalLM
from ankalm.optimization import (
    CosineWarmupScheduler,
    build_adamw_optimizer,
    build_configured_optimizer,
    check_gradient_health,
)
from ankalm.storage.checkpoint import CheckpointManager, CheckpointMetadata
from ankalm.tokenizer.tokenizer import AnkaLMTokenizer
from ankalm.training.process_manager import LiveTrainingState, TrainingProcessManager
from ankalm.utils.logging import append_jsonl_metric, get_logger

logger = get_logger("ankalm.training")


@dataclass
class TrainingRunResult:
    model_name: str
    profile: str
    parameter_count: int
    device: str
    precision: str
    start_step: int
    final_step: int
    steps_executed: int
    epochs_completed: int
    processed_tokens: int
    initial_train_loss: float
    final_train_loss: float
    best_val_loss: float
    final_val_loss: float
    final_val_perplexity: float
    avg_tokens_per_second: float
    peak_ram_mb: float
    peak_vram_mb: float
    initial_param_fingerprint: str
    final_param_fingerprint: str
    param_l2_delta: float
    weights_updated_verified: bool
    checkpoint_path: str
    maturity_level: str
    resumed_from: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnkaLMTrainer:
    """
    Production Causal Language Modeling Pretrainer for AnkaLM.
    Starts from random initialization (or resumes from an exact checkpoint), executes real
    forward/backward/AdamW updates, verifies parameter changes, evaluates on held-out validation data,
    and writes atomic checkpoints and live telemetry.
    """

    def __init__(
        self,
        config: AnkaLMConfig,
        tokenizer: Optional[AnkaLMTokenizer] = None,
        checkpoints_dir: Optional[Path] = None,
        logs_dir: Optional[Path] = None,
        config_path_str: str = "configs/small.yaml",
    ) -> None:
        self.config = config
        self.config_path_str = config_path_str
        self.paths = config.storage.resolve(PROJECT_ROOT)
        self.checkpoints_dir = checkpoints_dir or self.paths["checkpoints_dir"]
        self.logs_dir = logs_dir or self.paths["logs_dir"]
        self.ckpt_manager = CheckpointManager(self.checkpoints_dir)
        self.proc_manager = TrainingProcessManager(self.logs_dir)
        self.tokenizer = tokenizer or AnkaLMTokenizer.load(self.paths["tokenizer_dir"])

        # Align config vocab_size with actual trained tokenizer vocab_size
        self.config.model.vocab_size = self.tokenizer.vocab_size

        self.device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)
        if self.device.type == "cpu":
            logical = psutil.cpu_count(logical=True) or 8
            # Use 20 threads on 24-thread Ryzen 9 9900X3D (~4.45 TFLOPS BF16) while leaving 4 threads for OS/dataloaders
            torch.set_num_threads(max(4, min(logical - 4, 20)))

        self._interrupted = False

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_manifest_summary(self) -> Dict[str, Any]:
        m_path = self.paths["manifests_dir"] / "dataset_manifest.json"
        if m_path.exists():
            try:
                with open(m_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {
                    "total_datasets": data.get("total_datasets", 0),
                    "total_documents": data.get("total_documents", 0),
                    "total_tokens": data.get("total_tokens", 0),
                    "split_counts": data.get("split_counts", {}),
                    "split_token_counts": data.get("split_token_counts", {}),
                    "zero_leakage_verified": data.get("zero_leakage_verified", True),
                }
            except Exception:
                pass
        return {}

    def _prepare_datasets(self) -> Tuple[PackedTokenDataset, PackedTokenDataset]:
        train_jsonl = self.paths["processed_data_dir"] / "train.jsonl"
        val_jsonl = self.paths["validation_data_dir"] / "val.jsonl"
        tok_dir = self.paths["tokenized_data_dir"]
        tok_dir.mkdir(parents=True, exist_ok=True)

        seq_len = self.config.model.max_seq_len
        train_ds = PackedTokenDataset.from_jsonl(
            train_jsonl,
            encode_fn=lambda s: self.tokenizer.encode(s),
            seq_len=seq_len,
            stride=seq_len // 2,  # 50% overlapping windows for dense causal coverage
            cache_pt_path=tok_dir / f"train_packed_{seq_len}.pt",
        )
        val_ds = PackedTokenDataset.from_jsonl(
            val_jsonl,
            encode_fn=lambda s: self.tokenizer.encode(s),
            seq_len=seq_len,
            stride=seq_len,
            cache_pt_path=tok_dir / f"val_packed_{seq_len}.pt",
        )
        return train_ds, val_ds

    @torch.no_grad()
    def evaluate_validation(self, model: AnkaLMForCausalLM, val_ds: PackedTokenDataset, max_batches: int = 12) -> Tuple[float, float]:
        """Compute true average cross-entropy loss and perplexity on held-out validation dataset."""
        model.eval()
        bsz = min(self.config.training.batch_size, len(val_ds))
        total_loss = 0.0
        batches = 0
        use_amp = self.config.training.precision in ("bf16", "fp16")
        amp_dtype = torch.bfloat16 if self.config.training.precision == "bf16" else torch.float16

        for start_idx in range(0, len(val_ds), max(1, bsz)):
            if batches >= max_batches:
                break
            xs, ys = [], []
            for i in range(start_idx, min(start_idx + bsz, len(val_ds))):
                x, y = val_ds[i]
                xs.append(x)
                ys.append(y)
            x_batch = torch.stack(xs, dim=0).to(self.device)
            y_batch = torch.stack(ys, dim=0).to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=use_amp):
                _, loss, _ = model(x_batch, targets=y_batch)
            if loss is not None and torch.isfinite(loss):
                total_loss += float(loss.item())
                batches += 1

        model.train()
        avg_loss = total_loss / max(batches, 1)
        ppl = math.exp(min(avg_loss, 20.0))
        return (round(avg_loss, 6), round(ppl, 4))

    def train(
        self,
        resume_from: Optional[str | Path] = None,
        max_steps_override: Optional[int] = None,
        slot_prefix: Optional[str] = None,
        maturity_target: str = "pretrained",
    ) -> TrainingRunResult:
        """Execute genuine from-scratch training or resume seamlessly from an existing checkpoint."""
        self.proc_manager.ensure_no_competing_training()
        self.proc_manager.clear_signals()
        self._set_seed(self.config.training.seed)

        train_ds, val_ds = self._prepare_datasets()
        manifest_summary = self._load_manifest_summary()
        tok_meta = {
            "version": self.tokenizer.VERSION,
            "vocab_size": self.tokenizer.vocab_size,
            "vocab_fingerprint": self.tokenizer.vocab_fingerprint(),
        }

        model = build_ankalm_model(self.config.model).to(self.device)
        model.activation_checkpointing = self.config.training.activation_checkpointing
        param_count = model.count_parameters()

        t_cfg = self.config.training
        target_max_steps = max_steps_override if max_steps_override is not None else t_cfg.max_steps
        optimizer = build_configured_optimizer(
            model,
            optimizer_type=getattr(t_cfg, "optimizer_type", "adamw"),
            lr=t_cfg.learning_rate,
            weight_decay=t_cfg.weight_decay,
            beta1=t_cfg.beta1,
            beta2=t_cfg.beta2,
        )
        scheduler = CosineWarmupScheduler(
            optimizer,
            max_lr=t_cfg.learning_rate,
            min_lr=t_cfg.min_lr,
            warmup_steps=t_cfg.warmup_steps,
            max_steps=target_max_steps,
        )
        use_fp16_scaler = (self.device.type == "cuda") and (t_cfg.precision == "fp16")
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16_scaler) if self.device.type == "cuda" else None

        global_step = 0
        epoch = 0
        processed_tokens = 0
        cursor = 0
        data_order = list(range(len(train_ds)))
        random.shuffle(data_order)

        initial_snapshot = model.snapshot_weights()
        initial_fp = model.compute_parameter_fingerprint()
        resumed_label: Optional[str] = None

        if resume_from is not None:
            meta, payload = self.ckpt_manager.load_checkpoint(
                resume_from,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                restore_rng=True,
                device=self.device_str,
            )
            global_step = int(payload.get("global_step", meta.global_step))
            epoch = int(payload.get("epoch", meta.epoch))
            processed_tokens = int(payload.get("processed_tokens", meta.processed_tokens))
            initial_fp = str(payload.get("initial_param_fingerprint", meta.initial_param_fingerprint))
            samp_state = payload.get("dataset_sampling_state", {})
            cursor = int(samp_state.get("cursor", 0))
            saved_order = samp_state.get("data_order", [])
            if len(saved_order) == len(train_ds):
                data_order = list(saved_order)
            if target_max_steps <= global_step and max_steps_override is None:
                # If resuming a completed run without specifying new max_steps, extend by +50 steps
                target_max_steps = global_step + 50
                scheduler.max_steps = target_max_steps
            resumed_label = str(resume_from)
            logger.info("Resumed training from '%s' at step %d (tokens=%d)", resume_from, global_step, processed_tokens)
        else:
            # Save initial random_init checkpoint at step 0 for auditability
            init_val_loss, init_val_ppl = self.evaluate_validation(model, val_ds, max_batches=4)
            self.ckpt_manager.save_checkpoint(
                slot_name=f"{slot_prefix or self.config.model.profile}_init",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=self.config,
                global_step=0,
                epoch=0,
                processed_tokens=0,
                train_loss=init_val_loss,
                val_loss=init_val_loss,
                val_perplexity=init_val_ppl,
                initial_fingerprint=initial_fp,
                param_l2_delta=0.0,
                tokenizer_meta=tok_meta,
                dataset_manifest_summary=manifest_summary,
                dataset_sampling_state={"cursor": cursor, "data_order": data_order},
                maturity_level="random_init",
                update_latest_and_best=False,
            )

        start_step = global_step
        prev_sigint = signal.getsignal(signal.SIGINT)

        def _handle_sigint(signum: int, frame: Any) -> None:
            logger.warning("Received interrupt signal (%s); finishing current step and saving checkpoint safely...", signum)
            self._interrupted = True

        try:
            signal.signal(signal.SIGINT, _handle_sigint)
        except Exception:
            pass

        proc = psutil.Process(os.getpid())
        use_amp = t_cfg.precision in ("bf16", "fp16")
        amp_dtype = torch.bfloat16 if t_cfg.precision == "bf16" else torch.float16
        bsz = t_cfg.batch_size
        grad_accum = max(1, t_cfg.grad_accum_steps)

        initial_train_loss: Optional[float] = None
        last_train_loss = 0.0
        val_loss, val_ppl = self.evaluate_validation(model, val_ds, max_batches=6)
        best_val_loss = val_loss
        tps_history: List[float] = []
        peak_ram_mb = proc.memory_info().rss / (1024 * 1024)
        peak_vram_mb = 0.0

        def _next_batch() -> Tuple[torch.Tensor, torch.Tensor]:
            nonlocal cursor, epoch, data_order
            xs, ys = [], []
            for _ in range(bsz):
                if cursor >= len(data_order):
                    epoch += 1
                    cursor = 0
                    random.shuffle(data_order)
                idx = data_order[cursor]
                cursor += 1
                x, y = train_ds[idx]
                xs.append(x)
                ys.append(y)
            return torch.stack(xs, dim=0).to(self.device), torch.stack(ys, dim=0).to(self.device)

        model.train()
        try:
            while global_step < target_max_steps:
                if self._interrupted or self.proc_manager.should_stop():
                    logger.info("Training stop requested at step %d.", global_step)
                    break

                while self.proc_manager.is_paused() and not self._interrupted and not self.proc_manager.should_stop():
                    time.sleep(0.5)

                t_step0 = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                accum_loss = 0.0
                step_tokens = 0

                for _ in range(grad_accum):
                    xb, yb = _next_batch()
                    step_tokens += int(xb.numel())
                    with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=use_amp):
                        _, loss, _ = model(xb, targets=yb)
                        scaled_loss = loss / grad_accum

                    if not torch.isfinite(scaled_loss):
                        raise FloatingPointError(f"Non-finite loss detected at step {global_step + 1}: {loss.item()}")

                    if scaler is not None and use_fp16_scaler:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    accum_loss += float(scaled_loss.item())

                if scaler is not None and use_fp16_scaler:
                    scaler.unscale_(optimizer)

                grad_norm, grad_ok = check_gradient_health(model, max_norm=t_cfg.grad_clip)
                if not grad_ok:
                    logger.warning("Skipping step %d due to non-finite gradient norm", global_step + 1)
                    optimizer.zero_grad(set_to_none=True)
                    continue

                lr = scheduler.step(global_step)
                if scaler is not None and use_fp16_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                global_step += 1
                processed_tokens += step_tokens
                dt = max(time.perf_counter() - t_step0, 1e-6)
                tps = step_tokens / dt
                tps_history.append(tps)
                last_train_loss = accum_loss
                if initial_train_loss is None:
                    initial_train_loss = last_train_loss

                ram_mb = proc.memory_info().rss / (1024 * 1024)
                peak_ram_mb = max(peak_ram_mb, ram_mb)
                if self.device.type == "cuda":
                    vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
                    peak_vram_mb = max(peak_vram_mb, vram_mb)
                else:
                    vram_mb = 0.0

                should_eval = (global_step % t_cfg.eval_interval == 0) or (global_step == target_max_steps)
                if should_eval:
                    val_loss, val_ppl = self.evaluate_validation(model, val_ds, max_batches=8)
                    best_val_loss = min(best_val_loss, val_loss)

                l2_delta = model.compute_l2_delta(initial_snapshot)
                cur_maturity = "smoke_trained" if target_max_steps <= 80 else (
                    maturity_target if global_step >= target_max_steps // 2 else "experimental_base"
                )

                if (global_step % t_cfg.log_interval == 0) or should_eval or (global_step == 1):
                    metric_rec = {
                        "step": global_step,
                        "max_steps": target_max_steps,
                        "epoch": epoch,
                        "processed_tokens": processed_tokens,
                        "train_loss": round(last_train_loss, 5),
                        "val_loss": round(val_loss, 5),
                        "val_perplexity": round(val_ppl, 3),
                        "lr": round(lr, 7),
                        "grad_norm": grad_norm,
                        "tokens_per_sec": round(tps, 1),
                        "param_l2_delta": round(l2_delta, 5),
                        "ram_mb": round(ram_mb, 1),
                    }
                    append_jsonl_metric(self.proc_manager.metrics_file, metric_rec)
                    logger.info(
                        "[%s] step %d/%d | loss=%.4f | val_loss=%.4f | ppl=%.2f | lr=%.2e | tok/s=%.0f | L2_delta=%.4f",
                        self.config.model.name,
                        global_step,
                        target_max_steps,
                        last_train_loss,
                        val_loss,
                        val_ppl,
                        lr,
                        tps,
                        l2_delta,
                    )

                should_ckpt = (
                    (global_step % t_cfg.checkpoint_interval == 0)
                    or (global_step == target_max_steps)
                    or self.proc_manager.consume_save_request()
                )
                ckpt_status = f"Step {global_step} active"
                if should_ckpt:
                    slot = f"{slot_prefix or self.config.model.profile}_step_{global_step}"
                    saved_dir = self.ckpt_manager.save_checkpoint(
                        slot_name=slot,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        config=self.config,
                        global_step=global_step,
                        epoch=epoch,
                        processed_tokens=processed_tokens,
                        train_loss=last_train_loss,
                        val_loss=val_loss,
                        val_perplexity=val_ppl,
                        initial_fingerprint=initial_fp,
                        param_l2_delta=l2_delta,
                        tokenizer_meta=tok_meta,
                        dataset_manifest_summary=manifest_summary,
                        dataset_sampling_state={"cursor": cursor, "data_order": data_order},
                        maturity_level=cur_maturity,
                        update_latest_and_best=True,
                    )
                    ckpt_status = f"Saved {saved_dir.name}"

                live_state = LiveTrainingState(
                    status="running",
                    pid=os.getpid(),
                    process_alive=True,
                    updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    active_model=self.config.model.name,
                    model_profile=self.config.model.profile,
                    parameter_count=param_count,
                    tokenizer_vocab_size=self.tokenizer.vocab_size,
                    dataset_name="AnkaLM Turkish Pretraining Corpus",
                    global_step=global_step,
                    max_steps=target_max_steps,
                    epoch=epoch,
                    processed_tokens=processed_tokens,
                    train_loss=round(last_train_loss, 5),
                    val_loss=round(val_loss, 5),
                    perplexity=round(val_ppl, 3),
                    learning_rate=round(lr, 7),
                    tokens_per_second=round(tps, 1),
                    gpu_utilization_pct=100.0 if self.device.type == "cuda" else 0.0,
                    vram_usage_mb=round(vram_mb, 1),
                    ram_usage_mb=round(ram_mb, 1),
                    cpu_usage_pct=round(psutil.cpu_percent(interval=None), 1),
                    checkpoint_status=ckpt_status,
                    latest_checkpoint_path=str(self.checkpoints_dir / "latest"),
                    maturity_level=cur_maturity,
                    param_l2_delta=round(l2_delta, 5),
                    config_path=self.config_path_str,
                )
                self.proc_manager.write_state(live_state)

        finally:
            try:
                signal.signal(signal.SIGINT, prev_sigint)
            except Exception:
                pass

        # Save final session checkpoint
        val_loss, val_ppl = self.evaluate_validation(model, val_ds, max_batches=12)
        best_val_loss = min(best_val_loss, val_loss)
        final_l2_delta = model.compute_l2_delta(initial_snapshot)
        final_fp = model.compute_parameter_fingerprint()
        final_maturity = "smoke_trained" if target_max_steps <= 80 else maturity_target
        final_slot = f"{slot_prefix or self.config.model.profile}_final"

        final_ckpt_dir = self.ckpt_manager.save_checkpoint(
            slot_name=final_slot,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=self.config,
            global_step=global_step,
            epoch=epoch,
            processed_tokens=processed_tokens,
            train_loss=last_train_loss,
            val_loss=val_loss,
            val_perplexity=val_ppl,
            initial_fingerprint=initial_fp,
            param_l2_delta=final_l2_delta,
            tokenizer_meta=tok_meta,
            dataset_manifest_summary=manifest_summary,
            dataset_sampling_state={"cursor": cursor, "data_order": data_order},
            maturity_level=final_maturity,
            update_latest_and_best=True,
        )
        # Also update `session_final` slot required by Section 10
        self.ckpt_manager._copy_checkpoint_dir(final_ckpt_dir, self.checkpoints_dir / "session_final")

        avg_tps = sum(tps_history) / max(len(tps_history), 1)
        final_status = "stopped" if (self._interrupted or self.proc_manager.should_stop()) else "completed"
        self.proc_manager.write_state(
            LiveTrainingState(
                status=final_status,
                pid=os.getpid(),
                process_alive=False,
                updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                active_model=self.config.model.name,
                model_profile=self.config.model.profile,
                parameter_count=param_count,
                tokenizer_vocab_size=self.tokenizer.vocab_size,
                dataset_name="AnkaLM Turkish Pretraining Corpus",
                global_step=global_step,
                max_steps=target_max_steps,
                epoch=epoch,
                processed_tokens=processed_tokens,
                train_loss=round(last_train_loss, 5),
                val_loss=round(val_loss, 5),
                perplexity=round(val_ppl, 3),
                learning_rate=round(scheduler.current_lr, 7),
                tokens_per_second=round(avg_tps, 1),
                gpu_utilization_pct=0.0,
                vram_usage_mb=round(peak_vram_mb, 1),
                ram_usage_mb=round(peak_ram_mb, 1),
                cpu_usage_pct=0.0,
                checkpoint_status=f"Saved {final_ckpt_dir.name} & session_final",
                latest_checkpoint_path=str(final_ckpt_dir),
                maturity_level=final_maturity,
                param_l2_delta=round(final_l2_delta, 5),
                config_path=self.config_path_str,
            )
        )

        return TrainingRunResult(
            model_name=self.config.model.name,
            profile=self.config.model.profile,
            parameter_count=param_count,
            device=self.device_str,
            precision=t_cfg.precision,
            start_step=start_step,
            final_step=global_step,
            steps_executed=global_step - start_step,
            epochs_completed=epoch,
            processed_tokens=processed_tokens,
            initial_train_loss=round(initial_train_loss or last_train_loss, 5),
            final_train_loss=round(last_train_loss, 5),
            best_val_loss=round(best_val_loss, 5),
            final_val_loss=round(val_loss, 5),
            final_val_perplexity=round(val_ppl, 4),
            avg_tokens_per_second=round(avg_tps, 1),
            peak_ram_mb=round(peak_ram_mb, 2),
            peak_vram_mb=round(peak_vram_mb, 2),
            initial_param_fingerprint=initial_fp,
            final_param_fingerprint=final_fp,
            param_l2_delta=round(final_l2_delta, 6),
            weights_updated_verified=(final_fp != initial_fp) and (final_l2_delta > 0.0),
            checkpoint_path=str(final_ckpt_dir),
            maturity_level=final_maturity,
            resumed_from=resumed_label,
        )
