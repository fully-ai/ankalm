"""Local inference engine supporting KV cache, streaming, temperature, top-k, top-p, and repetition penalty."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import time
from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn.functional as F

from ankalm.core.config import ModelProfileConfig
from ankalm.core.constants import CHECKPOINT_MATURITY_LEVELS, PROJECT_ROOT
from ankalm.model import build_ankalm_model
from ankalm.model.transformer import AnkaLMForCausalLM
from ankalm.storage.checkpoint import CheckpointManager, CheckpointMetadata
from ankalm.tokenizer.tokenizer import AnkaLMTokenizer


@dataclass
class GenerationConfig:
    max_new_tokens: int = 96
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    greedy: bool = False
    repetition_penalty: float = 1.15
    seed: Optional[int] = 42
    stop_sequences: List[str] = field(default_factory=lambda: ["<|eos|>", "<|eot|>", "<|user|>"])
    use_kv_cache: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    prompt: str
    generated_text: str
    full_text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    first_token_latency_ms: float
    total_time_ms: float
    tokens_per_second: float
    finish_reason: str
    model_name: str
    checkpoint_name: str
    maturity_level: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnkaLMInferenceEngine:
    """
    Production Local Inference Runtime for AnkaLM.
    Refuses to present random_init weights as an inference-ready trained model (Section 1.3 & 13).
    """

    def __init__(
        self,
        model: AnkaLMForCausalLM,
        tokenizer: AnkaLMTokenizer,
        metadata: CheckpointMetadata,
        device_str: str = "cpu",
        precision: str = "fp32",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.device_str = device_str
        self.device = torch.device(device_str)
        self.precision = precision.lower()
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_name_or_path: str | Path = "latest",
        checkpoints_dir: Optional[Path] = None,
        tokenizer_dir: Optional[Path] = None,
        device_str: Optional[str] = None,
        precision: str = "fp32",
        allow_random_init: bool = False,
    ) -> "AnkaLMInferenceEngine":
        ckpt_mgr = CheckpointManager(checkpoints_dir)
        meta, payload = ckpt_mgr.load_checkpoint(checkpoint_name_or_path, restore_rng=False, device="cpu")

        if not allow_random_init and (meta.maturity_level == "random_init" or meta.global_step == 0 or not meta.weights_updated_verified):
            raise RuntimeError(
                f"Checkpoint '{meta.checkpoint_name}' has maturity='{meta.maturity_level}' (step={meta.global_step}). "
                "AnkaLM refuses to present randomly initialized weights as a trained model."
            )

        tok = AnkaLMTokenizer.load(tokenizer_dir)
        m_cfg = ModelProfileConfig(**meta.model_config)
        dev = device_str or ("cuda:0" if torch.cuda.is_available() else "cpu")

        model = build_ankalm_model(m_cfg)
        model.load_state_dict(payload["model_state_dict"])

        prec = precision.lower()
        if prec == "bf16":
            model = model.to(dtype=torch.bfloat16)
        elif prec == "fp16":
            model = model.to(dtype=torch.float16)
        elif prec == "int8":
            from ankalm.quantization.quantizer import apply_int8_dynamic_quantization
            model = apply_int8_dynamic_quantization(model)
        elif prec == "int4":
            from ankalm.quantization.quantizer import apply_int4_weight_quantization
            model = apply_int4_weight_quantization(model)

        model = model.to(dev)
        model.eval()
        return cls(model=model, tokenizer=tok, metadata=meta, device_str=dev, precision=prec)

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        generated_ids: List[int],
        gen_cfg: GenerationConfig,
        generator: Optional[torch.Generator],
    ) -> int:
        next_logits = logits[0, -1, :].float().clone()

        # Prevent sampling PAD or UNK, and prevent immediate empty EOS/EOT on first 2 tokens
        next_logits[self.tokenizer.pad_id] = -1e9
        next_logits[self.tokenizer.unk_id] = -1e9
        if switch_guard := (len(generated_ids) < 2):
            next_logits[self.tokenizer.eos_id] = -1e9
            next_logits[self.tokenizer.eot_id] = -1e9

        # Apply repetition penalty
        if gen_cfg.repetition_penalty != 1.0 and generated_ids:
            for prev_id in set(generated_ids[-64:]):
                if next_logits[prev_id] > 0:
                    next_logits[prev_id] /= gen_cfg.repetition_penalty
                else:
                    next_logits[prev_id] *= gen_cfg.repetition_penalty

        if gen_cfg.greedy or gen_cfg.temperature <= 1e-5:
            return int(torch.argmax(next_logits).item())

        scaled = next_logits / max(gen_cfg.temperature, 1e-5)

        # Top-K filtering
        if gen_cfg.top_k > 0 and gen_cfg.top_k < scaled.size(-1):
            topk_vals, _ = torch.topk(scaled, gen_cfg.top_k)
            min_topk = topk_vals[-1]
            scaled[scaled < min_topk] = -float("inf")

        # Top-P (nucleus) filtering
        if 0.0 < gen_cfg.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > gen_cfg.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            scaled[indices_to_remove] = -float("inf")

        probs = F.softmax(scaled, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1, generator=generator)
        return int(sampled.item())

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        gen_cfg: Optional[GenerationConfig] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Stream generated text chunks with token timing and KV cache management."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        cfg = gen_cfg or GenerationConfig()
        generator: Optional[torch.Generator] = None
        if cfg.seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(cfg.seed))

        input_ids = self.tokenizer.encode(prompt, add_bos= not prompt.startswith("<|bos|>"))
        max_ctx = self.model.config.max_seq_len
        if len(input_ids) >= max_ctx:
            input_ids = input_ids[-(max_ctx - 1) :]

        prompt_len = len(input_ids)
        generated_ids: List[int] = []
        past_kv = None
        t_start = time.perf_counter()
        first_token_ms = 0.0
        prev_decoded = ""
        finish_reason = "length"

        eos_ids = {self.tokenizer.eos_id, self.tokenizer.eot_id}
        max_steps = min(cfg.max_new_tokens, max(1, max_ctx * 2 - prompt_len))

        curr_input = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        for step_idx in range(max_steps):
            if not cfg.use_kv_cache:
                full_seq = (input_ids + generated_ids)[-max_ctx:]
                curr_input = torch.tensor([full_seq], dtype=torch.long, device=self.device)
                past_kv = None

            logits, _, past_kv = self.model(curr_input, past_key_values=past_kv, use_cache=cfg.use_kv_cache)
            next_id = self._sample_next_token(logits, input_ids + generated_ids, cfg, generator)

            if step_idx == 0:
                first_token_ms = (time.perf_counter() - t_start) * 1000.0

            if next_id in eos_ids:
                finish_reason = "eos"
                break

            generated_ids.append(next_id)
            curr_decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Check stop sequences
            raw_decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            hit_stop = False
            for stop_seq in cfg.stop_sequences:
                if stop_seq and stop_seq in raw_decoded:
                    finish_reason = "stop_sequence"
                    hit_stop = True
                    break

            delta = curr_decoded[len(prev_decoded) :]
            prev_decoded = curr_decoded

            if delta:
                yield {
                    "token_id": next_id,
                    "delta": delta,
                    "text": curr_decoded,
                    "step": step_idx + 1,
                    "first_token_latency_ms": round(first_token_ms, 2),
                    "finished": hit_stop,
                    "finish_reason": finish_reason if hit_stop else None,
                }

            if hit_stop:
                break

            curr_input = torch.tensor([[next_id]], dtype=torch.long, device=self.device)

        total_ms = (time.perf_counter() - t_start) * 1000.0
        comp_tokens = len(generated_ids)
        tps = (comp_tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0
        yield {
            "token_id": None,
            "delta": "",
            "text": prev_decoded,
            "step": comp_tokens,
            "prompt_tokens": prompt_len,
            "completion_tokens": comp_tokens,
            "total_tokens": prompt_len + comp_tokens,
            "first_token_latency_ms": round(first_token_ms, 2),
            "total_time_ms": round(total_ms, 2),
            "tokens_per_second": round(tps, 2),
            "finished": True,
            "finish_reason": finish_reason,
        }

    def generate(self, prompt: str, gen_cfg: Optional[GenerationConfig] = None) -> GenerationResult:
        last_chunk: Dict[str, Any] = {}
        for chunk in self.generate_stream(prompt, gen_cfg=gen_cfg):
            last_chunk = chunk
        gen_text = last_chunk.get("text", "")
        return GenerationResult(
            prompt=prompt,
            generated_text=gen_text,
            full_text=prompt + gen_text,
            prompt_tokens=int(last_chunk.get("prompt_tokens", 0)),
            completion_tokens=int(last_chunk.get("completion_tokens", 0)),
            total_tokens=int(last_chunk.get("total_tokens", 0)),
            first_token_latency_ms=float(last_chunk.get("first_token_latency_ms", 0.0)),
            total_time_ms=float(last_chunk.get("total_time_ms", 0.0)),
            tokens_per_second=float(last_chunk.get("tokens_per_second", 0.0)),
            finish_reason=str(last_chunk.get("finish_reason", "length")),
            model_name=self.metadata.model_name,
            checkpoint_name=self.metadata.checkpoint_name,
            maturity_level=self.metadata.maturity_level,
        )
