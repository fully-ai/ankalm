"""Atomic checkpoint creation, verification, rotation, and full-state resume support."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from safetensors.torch import save_file as save_safetensors
import torch
import torch.nn as nn

from ankalm.core.config import AnkaLMConfig, ModelProfileConfig
from ankalm.core.constants import CHECKPOINT_MATURITY_LEVELS, PROJECT_ROOT


@dataclass
class CheckpointMetadata:
    checkpoint_name: str
    created_at: str
    maturity_level: str
    maturity_description: str
    model_name: str
    model_profile: str
    parameter_count: int
    global_step: int
    epoch: int
    processed_tokens: int
    train_loss: float
    val_loss: float
    val_perplexity: float
    learning_rate: float
    initial_param_fingerprint: str
    current_param_fingerprint: str
    param_l2_delta_from_init: float
    weights_updated_verified: bool
    tokenizer_version: str
    tokenizer_vocab_size: int
    tokenizer_fingerprint: str
    dataset_manifest_summary: Dict[str, Any]
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CheckpointManager:
    """
    Atomic checkpoint manager maintaining `latest`, `previous`, `best`, `session_final`,
    and named slots while never deleting the only valid checkpoint.
    """

    def __init__(self, checkpoints_dir: Optional[Path] = None) -> None:
        self.checkpoints_dir = checkpoints_dir or (PROJECT_ROOT / "checkpoints")
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def _capture_rng_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "python_random": random.getstate(),
            "numpy_random": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, rng_state: Dict[str, Any]) -> None:
        try:
            if "python_random" in rng_state:
                random.setstate(rng_state["python_random"])
            if "numpy_random" in rng_state:
                np.random.set_state(rng_state["numpy_random"])
            if "torch_cpu" in rng_state:
                torch.set_rng_state(rng_state["torch_cpu"])
            if "torch_cuda" in rng_state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        except Exception:
            pass

    def save_checkpoint(
        self,
        slot_name: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        scaler: Optional[Any],
        config: AnkaLMConfig,
        global_step: int,
        epoch: int,
        processed_tokens: int,
        train_loss: float,
        val_loss: float,
        val_perplexity: float,
        initial_fingerprint: str,
        param_l2_delta: float,
        tokenizer_meta: Dict[str, Any],
        dataset_manifest_summary: Dict[str, Any],
        dataset_sampling_state: Dict[str, Any],
        maturity_level: str = "pretrained",
        update_latest_and_best: bool = True,
    ) -> Path:
        """Atomically save a full training checkpoint and rotate `latest` -> `previous` safely."""
        if maturity_level not in CHECKPOINT_MATURITY_LEVELS:
            raise ValueError(f"Invalid maturity_level: {maturity_level}")

        current_fp = model.compute_parameter_fingerprint() if hasattr(model, "compute_parameter_fingerprint") else ""
        weights_changed = (global_step > 0) and (current_fp != initial_fingerprint) and (param_l2_delta > 0.0)
        if global_step == 0:
            maturity_level = "random_init"

        meta = CheckpointMetadata(
            checkpoint_name=slot_name,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            maturity_level=maturity_level,
            maturity_description=CHECKPOINT_MATURITY_LEVELS[maturity_level],
            model_name=config.model.name,
            model_profile=config.model.profile,
            parameter_count=sum(p.numel() for p in model.parameters()),
            global_step=global_step,
            epoch=epoch,
            processed_tokens=processed_tokens,
            train_loss=round(float(train_loss), 6),
            val_loss=round(float(val_loss), 6),
            val_perplexity=round(float(val_perplexity), 4),
            learning_rate=round(float(scheduler.current_lr if scheduler else config.training.learning_rate), 8),
            initial_param_fingerprint=initial_fingerprint,
            current_param_fingerprint=current_fp,
            param_l2_delta_from_init=round(float(param_l2_delta), 6),
            weights_updated_verified=weights_changed,
            tokenizer_version=str(tokenizer_meta.get("version", "1.0.0")),
            tokenizer_vocab_size=int(tokenizer_meta.get("vocab_size", config.model.vocab_size)),
            tokenizer_fingerprint=str(tokenizer_meta.get("vocab_fingerprint", "")),
            dataset_manifest_summary=dataset_manifest_summary,
            model_config=asdict(config.model),
            training_config=asdict(config.training),
        )

        # Write atomically to a temporary directory first
        tmp_dir = self.checkpoints_dir / f".tmp_{slot_name}_{os.getpid()}_{int(time.time()*1000)}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Metadata JSON
            with open(tmp_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta.to_dict(), f, indent=2, ensure_ascii=False)

            # 2. Config YAML
            config.to_yaml(tmp_dir / "config.yaml")

            # 3. Safetensors weights (handle tied weights cleanly by cloning detached tensors)
            st_dict = {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}
            save_safetensors(st_dict, str(tmp_dir / "model.safetensors"))

            # 4. Full PyTorch training state (model, optimizer, scheduler, scaler, RNG, sampler)
            training_payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "global_step": global_step,
                "epoch": epoch,
                "processed_tokens": processed_tokens,
                "initial_param_fingerprint": initial_fingerprint,
                "param_l2_delta": param_l2_delta,
                "dataset_sampling_state": dataset_sampling_state,
                "rng_state": self._capture_rng_state(),
                "metadata": meta.to_dict(),
            }
            torch.save(training_payload, tmp_dir / "training_state.pt")

            # Verify files exist and are non-empty before atomic move
            for req in ("metadata.json", "config.yaml", "model.safetensors", "training_state.pt"):
                p = tmp_dir / req
                if not p.exists() or p.stat().st_size == 0:
                    raise RuntimeError(f"Checkpoint verification failed: missing {req}")

            target_dir = self.checkpoints_dir / slot_name
            self._atomic_replace_dir(tmp_dir, target_dir)

            if update_latest_and_best:
                latest_dir = self.checkpoints_dir / "latest"
                prev_dir = self.checkpoints_dir / "previous"
                if latest_dir.exists() and (latest_dir / "metadata.json").exists():
                    self._copy_checkpoint_dir(latest_dir, prev_dir)
                self._copy_checkpoint_dir(target_dir, latest_dir)

                # Update `best` if val_loss is lower than existing best
                best_dir = self.checkpoints_dir / "best"
                should_update_best = True
                if best_dir.exists() and (best_dir / "metadata.json").exists():
                    try:
                        with open(best_dir / "metadata.json", "r", encoding="utf-8") as bf:
                            best_meta = json.load(bf)
                        if float(best_meta.get("val_loss", 9999.0)) <= val_loss:
                            should_update_best = False
                    except Exception:
                        should_update_best = True
                if should_update_best and global_step > 0:
                    self._copy_checkpoint_dir(target_dir, best_dir)

            return target_dir
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _atomic_replace_dir(self, src_tmp: Path, target: Path) -> None:
        backup = self.checkpoints_dir / f".bak_{target.name}_{os.getpid()}"
        if target.exists():
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
            target.rename(backup)
        try:
            src_tmp.rename(target)
        except Exception:
            if backup.exists() and not target.exists():
                backup.rename(target)
            raise
        finally:
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)

    def _copy_checkpoint_dir(self, src: Path, dst: Path) -> None:
        if src.resolve() == dst.resolve():
            return
        tmp_dst = self.checkpoints_dir / f".tmp_copy_{dst.name}_{os.getpid()}"
        if tmp_dst.exists():
            shutil.rmtree(tmp_dst, ignore_errors=True)
        shutil.copytree(src, tmp_dst)
        self._atomic_replace_dir(tmp_dst, dst)

    def resolve_checkpoint_dir(self, name_or_path: str | Path = "latest") -> Path:
        p = Path(name_or_path)
        if p.exists() and (p / "metadata.json").exists():
            return p.resolve()
        candidate = self.checkpoints_dir / str(name_or_path)
        if candidate.exists() and (candidate / "metadata.json").exists():
            return candidate.resolve()
        raise FileNotFoundError(f"Valid AnkaLM checkpoint not found for '{name_or_path}' in {self.checkpoints_dir}")

    def load_checkpoint(
        self,
        name_or_path: str | Path = "latest",
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        restore_rng: bool = True,
        device: str = "cpu",
    ) -> Tuple[CheckpointMetadata, Dict[str, Any]]:
        ckpt_dir = self.resolve_checkpoint_dir(name_or_path)
        with open(ckpt_dir / "metadata.json", "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
        meta = CheckpointMetadata(**meta_dict)

        payload = torch.load(ckpt_dir / "training_state.pt", map_location=device, weights_only=False)
        if model is not None:
            model.load_state_dict(payload["model_state_dict"])
        if optimizer is not None and payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if scaler is not None and payload.get("scaler_state_dict") is not None:
            scaler.load_state_dict(payload["scaler_state_dict"])
        if restore_rng and payload.get("rng_state") is not None:
            self._restore_rng_state(payload["rng_state"])

        return meta, payload

    def list_checkpoints(self) -> List[CheckpointMetadata]:
        results: List[CheckpointMetadata] = []
        if not self.checkpoints_dir.exists():
            return results
        for child in sorted(self.checkpoints_dir.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                meta_file = child / "metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        results.append(CheckpointMetadata(**data))
                    except Exception:
                        pass
        return results
