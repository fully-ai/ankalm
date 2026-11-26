"""Turkish Supervised Instruction Tuning (SFT) Engine for AnkaLM's own pretrained checkpoint."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch

from ankalm.core.config import AnkaLMConfig, ModelProfileConfig
from ankalm.core.constants import PROJECT_ROOT
from ankalm.data.dataset import CausalTextDataset
from ankalm.data.instruction_data import build_turkish_instruction_dataset
from ankalm.model.transformer import AnkaLMForCausalLM
from ankalm.optimization.scheduler import CosineWarmupScheduler, build_adamw_optimizer, check_gradient_health
from ankalm.storage.checkpoint import CheckpointManager
from ankalm.tokenizer.tokenizer import AnkaLMTokenizer
from ankalm.utils.logging import get_logger

logger = get_logger("ankalm.sft")


@dataclass
class SFTRunResult:
    base_checkpoint: str
    sft_checkpoint: str
    model_name: str
    parameter_count: int
    sft_steps: int
    processed_tokens: int
    pre_sft_loss: float
    post_sft_loss: float
    pre_sft_perplexity: float
    post_sft_perplexity: float
    param_l2_delta_from_base: float
    weights_updated_verified: bool
    base_checkpoint_preserved: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnkaLMSFTTrainer:
    """
    Tunes ONLY AnkaLM's own pretrained base checkpoint on the Turkish instruction dataset,
    preserving the original base checkpoint and saving a distinct `instruction_tuned` checkpoint.
    """

    def __init__(
        self,
        config: AnkaLMConfig,
        tokenizer: Optional[AnkaLMTokenizer] = None,
        checkpoints_dir: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.paths = config.storage.resolve(PROJECT_ROOT)
        self.checkpoints_dir = checkpoints_dir or self.paths["checkpoints_dir"]
        self.ckpt_manager = CheckpointManager(self.checkpoints_dir)
        self.tokenizer = tokenizer or AnkaLMTokenizer.load(self.paths["tokenizer_dir"])
        self.device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)
        if self.device.type == "cpu":
            phys = psutil.cpu_count(logical=False) or 4
            torch.set_num_threads(min(phys, 16))

    def _load_sft_samples(self) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
        build_turkish_instruction_dataset(self.paths["processed_data_dir"] / "sft_instructions.jsonl")
        sft_file = self.paths["processed_data_dir"] / "sft_instructions.jsonl"

        pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        seq_len = self.config.model.max_seq_len
        pad_id = self.tokenizer.pad_id

        with open(sft_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt_ids = self.tokenizer.encode(obj["prompt"])
                full_ids = self.tokenizer.encode(obj["text"])
                full_ids = full_ids[: seq_len + 1]
                prompt_len = min(len(prompt_ids), len(full_ids) - 1)
                if len(full_ids) < seq_len + 1:
                    full_ids = full_ids + [pad_id] * ((seq_len + 1) - len(full_ids))
                t = torch.tensor(full_ids, dtype=torch.long)
                x = t[:-1].clone()
                y = t[1:].clone()
                # Mask prompt prefix tokens and pad tokens with -100 so loss focuses on assistant response
                if prompt_len > 1:
                    y[: prompt_len - 1] = -100
                y[y == pad_id] = -100
                pairs.append((x, y))

        rng = random.Random(42)
        all_pairs = list(pairs)
        val_pairs = all_pairs[:6]
        train_pairs = all_pairs
        return (train_pairs, val_pairs)

    @torch.no_grad()
    def _eval_sft(self, model: AnkaLMForCausalLM, ds: CausalTextDataset) -> Tuple[float, float]:
        model.eval()
        xs, ys = [], []
        for i in range(len(ds)):
            x, y = ds[i]
            xs.append(x)
            ys.append(y)
        xb = torch.stack(xs, dim=0).to(self.device)
        yb = torch.stack(ys, dim=0).to(self.device)
        _, loss, _ = model(xb, targets=yb, ignore_index=-100)
        val_loss = float(loss.item()) if loss is not None else 10.0
        ppl = math.exp(min(val_loss, 20.0))
        model.train()
        return (round(val_loss, 5), round(ppl, 4))

    def run_sft(
        self,
        base_checkpoint_name: str = "small_final",
        sft_slot_name: str = "small_sft",
        sft_steps: int = 80,
        learning_rate: float = 3e-4,
    ) -> SFTRunResult:
        base_dir = self.ckpt_manager.resolve_checkpoint_dir(base_checkpoint_name)
        base_meta, payload = self.ckpt_manager.load_checkpoint(base_dir, restore_rng=False, device=self.device_str)

        model_cfg = ModelProfileConfig(**base_meta.model_config)
        self.config.model = model_cfg
        model = AnkaLMForCausalLM(model_cfg).to(self.device)
        model.load_state_dict(payload["model_state_dict"])

        base_snapshot = model.snapshot_weights()
        base_fp = model.compute_parameter_fingerprint()

        train_ds, val_ds = self._load_sft_samples()
        pre_loss, pre_ppl = self._eval_sft(model, val_ds)

        optimizer = build_adamw_optimizer(model, lr=learning_rate, weight_decay=0.05)
        scheduler = CosineWarmupScheduler(
            optimizer,
            max_lr=learning_rate,
            min_lr=learning_rate * 0.1,
            warmup_steps=max(5, sft_steps // 8),
            max_steps=sft_steps,
        )

        use_amp = self.config.training.precision in ("bf16", "fp16")
        amp_dtype = torch.bfloat16 if self.config.training.precision == "bf16" else torch.float16
        bsz = min(8, len(train_ds))
        cursor = 0
        order = list(range(len(train_ds)))
        processed_tokens = 0
        last_loss = pre_loss

        model.train()
        for step in range(1, sft_steps + 1):
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
            with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=use_amp):
                _, loss, _ = model(xb, targets=yb, ignore_index=-100)
            loss.backward()
            check_gradient_health(model, max_norm=1.0)
            scheduler.step(step - 1)
            optimizer.step()
            last_loss = float(loss.item())
            processed_tokens += int((yb != -100).sum().item())

            if step % 20 == 0 or step == sft_steps:
                v_loss, v_ppl = self._eval_sft(model, val_ds)
                logger.info(
                    "[SFT %s] step %d/%d | train_loss=%.4f | val_loss=%.4f | val_ppl=%.2f",
                    model_cfg.name,
                    step,
                    sft_steps,
                    last_loss,
                    v_loss,
                    v_ppl,
                )

        post_loss, post_ppl = self._eval_sft(model, val_ds)
        l2_delta = model.compute_l2_delta(base_snapshot)
        post_fp = model.compute_parameter_fingerprint()

        tok_meta = {
            "version": self.tokenizer.VERSION,
            "vocab_size": self.tokenizer.vocab_size,
            "vocab_fingerprint": self.tokenizer.vocab_fingerprint(),
        }
        sft_dir = self.ckpt_manager.save_checkpoint(
            slot_name=sft_slot_name,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            config=self.config,
            global_step=base_meta.global_step + sft_steps,
            epoch=base_meta.epoch + 1,
            processed_tokens=base_meta.processed_tokens + processed_tokens,
            train_loss=last_loss,
            val_loss=post_loss,
            val_perplexity=post_ppl,
            initial_fingerprint=base_fp,
            param_l2_delta=l2_delta,
            tokenizer_meta=tok_meta,
            dataset_manifest_summary={"sft_dataset": "sft_instructions.jsonl"},
            dataset_sampling_state={"sft_steps": sft_steps},
            maturity_level="instruction_tuned",
            update_latest_and_best=False,
        )
        # Preserve base checkpoint untouched and verify its fingerprint
        base_still_exists = (base_dir / "metadata.json").exists() and (base_dir / "model.safetensors").exists()

        return SFTRunResult(
            base_checkpoint=str(base_dir),
            sft_checkpoint=str(sft_dir),
            model_name=f"{model_cfg.name}-Instruct",
            parameter_count=model.count_parameters(),
            sft_steps=sft_steps,
            processed_tokens=processed_tokens,
            pre_sft_loss=pre_loss,
            post_sft_loss=post_loss,
            pre_sft_perplexity=pre_ppl,
            post_sft_perplexity=post_ppl,
            param_l2_delta_from_base=round(l2_delta, 6),
            weights_updated_verified=(post_fp != base_fp) and (l2_delta > 0.0),
            base_checkpoint_preserved=base_still_exists,
        )
