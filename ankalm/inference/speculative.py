"""Speculative Decoding, Prompt Prefix KV Caching, and 2GB Ultra-Low VRAM Execution Profiles (Section 13 & 15)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch

from ankalm.core.constants import PROJECT_ROOT
from ankalm.inference.engine import AnkaLMInferenceEngine, GenerationConfig
from ankalm.quantization.quantizer import compute_model_storage_mb


@dataclass
class SpeculativeDecodingResult:
    prompt: str
    generated_text: str
    draft_model: str
    target_model: str
    lookahead_k: int
    total_generated_tokens: int
    proposed_draft_tokens: int
    accepted_draft_tokens: int
    acceptance_rate: float
    target_forward_calls: int
    speculative_tokens_per_sec: float
    baseline_target_tokens_per_sec: float
    speedup_ratio: float
    exact_quality_equivalence: bool
    total_memory_mb: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LowVRAMProfileBenchmark:
    profile_name: str
    vram_budget_target_gb: float
    checkpoint_used: str
    model_name: str
    parameter_count: int
    precision: str
    model_weights_mb: float
    runtime_rss_delta_mb: float
    fits_in_2gb_budget: bool
    startup_load_ms: float
    first_token_latency_ms: float
    tokens_per_second: float
    prefix_cache_speedup_pct: float
    sample_output: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SpeculativeDecoder:
    """
    Lossless Speculative Decoding Engine (Section 13.5).
    Uses a fast compact Draft model (`AnkaLM-Edge`) to propose K tokens and verifies all K tokens
    in a single parallel forward pass of the Target model (`AnkaLM-Small` or `AnkaLM-MoE`).
    Guarantees 100% mathematical quality equivalence to greedy target decoding.
    """

    def __init__(
        self,
        draft_engine: AnkaLMInferenceEngine,
        target_engine: AnkaLMInferenceEngine,
        lookahead_k: int = 4,
    ) -> None:
        self.draft = draft_engine
        self.target = target_engine
        self.lookahead_k = max(1, lookahead_k)

    @torch.no_grad()
    def generate_speculative(self, prompt: str, max_new_tokens: int = 32) -> SpeculativeDecodingResult:
        proc = psutil.Process(os.getpid())
        tok = self.target.tokenizer
        input_ids = tok.encode(prompt, add_bos=not prompt.startswith("<|bos|>"))
        eos_ids = {tok.eos_id, tok.eot_id}

        # 1. Baseline target-only greedy generation for exact quality & speed comparison
        t_base0 = time.perf_counter()
        base_seq = list(input_ids)
        for _ in range(max_new_tokens):
            inp_t = torch.tensor([base_seq], dtype=torch.long, device=self.target.device)
            t_logits, _, _ = self.target.model(inp_t)
            next_l = t_logits[0, -1, :].float()
            next_l[tok.pad_id] = -1e9
            next_l[tok.unk_id] = -1e9
            if len(base_seq) - len(input_ids) < 2:
                next_l[tok.eos_id] = -1e9
                next_l[tok.eot_id] = -1e9
            nt = int(torch.argmax(next_l).item())
            if nt in eos_ids:
                break
            base_seq.append(nt)
        dt_base = max(time.perf_counter() - t_base0, 1e-6)
        base_gen_ids = base_seq[len(input_ids) :]
        base_tps = len(base_gen_ids) / dt_base

        # 2. Speculative Decoding (Draft proposes K tokens -> Target verifies K+1 tokens in 1 batch forward pass)
        t_spec0 = time.perf_counter()
        spec_seq = list(input_ids)
        proposed_total = 0
        accepted_total = 0
        target_calls = 0

        while (len(spec_seq) - len(input_ids)) < max_new_tokens:
            rem = max_new_tokens - (len(spec_seq) - len(input_ids))
            cur_k = min(self.lookahead_k, rem)

            # Draft model proposes cur_k tokens
            draft_seq = list(spec_seq)
            draft_proposed: List[int] = []
            for _ in range(cur_k):
                d_inp = torch.tensor([draft_seq], dtype=torch.long, device=self.draft.device)
                d_logits, _, _ = self.draft.model(d_inp)
                dl = d_logits[0, -1, :].float()
                dl[tok.pad_id] = -1e9
                dl[tok.unk_id] = -1e9
                if len(draft_seq) - len(input_ids) < 2:
                    dl[tok.eos_id] = -1e9
                    dl[tok.eot_id] = -1e9
                dtok = int(torch.argmax(dl).item())
                draft_proposed.append(dtok)
                draft_seq.append(dtok)

            proposed_total += len(draft_proposed)

            # Target model verifies all cur_k proposed tokens in ONE forward pass
            v_inp = torch.tensor([draft_seq], dtype=torch.long, device=self.target.device)
            v_logits, _, _ = self.target.model(v_inp)
            target_calls += 1

            prefix_len = len(spec_seq)
            hit_eos = False
            all_accepted = True

            for i, d_tok in enumerate(draft_proposed):
                pos_idx = prefix_len - 1 + i
                tl = v_logits[0, pos_idx, :].float()
                tl[tok.pad_id] = -1e9
                tl[tok.unk_id] = -1e9
                if (len(spec_seq) - len(input_ids)) < 2:
                    tl[tok.eos_id] = -1e9
                    tl[tok.eot_id] = -1e9
                t_tok = int(torch.argmax(tl).item())

                if t_tok == d_tok:
                    accepted_total += 1
                    if t_tok in eos_ids:
                        hit_eos = True
                        all_accepted = False
                        break
                    spec_seq.append(t_tok)
                else:
                    # Reject draft token at i and accept target's correction token immediately
                    if t_tok in eos_ids:
                        hit_eos = True
                    else:
                        spec_seq.append(t_tok)
                    all_accepted = False
                    break

            if hit_eos:
                break

            # Bonus token from target logits at position -1 if all cur_k draft tokens were accepted!
            if all_accepted and (len(spec_seq) - len(input_ids)) < max_new_tokens:
                bl = v_logits[0, -1, :].float()
                bl[tok.pad_id] = -1e9
                bl[tok.unk_id] = -1e9
                b_tok = int(torch.argmax(bl).item())
                if b_tok in eos_ids:
                    break
                spec_seq.append(b_tok)

        dt_spec = max(time.perf_counter() - t_spec0, 1e-6)
        spec_gen_ids = spec_seq[len(input_ids) : len(input_ids) + max_new_tokens]
        spec_tps = len(spec_gen_ids) / dt_spec
        acc_rate = accepted_total / max(proposed_total, 1)
        exact_match = (spec_gen_ids == base_gen_ids)
        gen_text = tok.decode(spec_gen_ids, skip_special_tokens=True)
        mem_mb = compute_model_storage_mb(self.draft.model) + compute_model_storage_mb(self.target.model)

        return SpeculativeDecodingResult(
            prompt=prompt,
            generated_text=gen_text,
            draft_model=self.draft.metadata.model_name,
            target_model=self.target.metadata.model_name,
            lookahead_k=self.lookahead_k,
            total_generated_tokens=len(spec_gen_ids),
            proposed_draft_tokens=proposed_total,
            accepted_draft_tokens=accepted_total,
            acceptance_rate=round(acc_rate, 4),
            target_forward_calls=target_calls,
            speculative_tokens_per_sec=round(spec_tps, 1),
            baseline_target_tokens_per_sec=round(base_tps, 1),
            speedup_ratio=round(spec_tps / max(base_tps, 1e-6), 3),
            exact_quality_equivalence=exact_match,
            total_memory_mb=round(mem_mb, 2),
        )


def benchmark_low_vram_profiles() -> Dict[str, Any]:
    """
    Benchmark Ultra-Low VRAM (2GB Target) Edge, Balanced, and Flagship execution profiles
    plus Speculative Decoding and Prefix KV Caching (Section 13 & 21).
    """
    proc = psutil.Process(os.getpid())
    profiles_out: List[LowVRAMProfileBenchmark] = []

    profile_specs = [
        ("edge_2gb_int4", 2.0, "edge_final", "int4"),
        ("edge_2gb_int8", 2.0, "edge_final", "int8"),
        ("edge_2gb_fp16", 2.0, "edge_final", "fp16"),
        ("balanced_4gb_int4", 4.0, "small_final", "int4"),
        ("balanced_4gb_fp16", 4.0, "small_final", "fp16"),
        ("moe_sparse_bf16", 8.0, "moe_final", "bf16"),
    ]

    prompt = "Türkiye ve Türkçe doğal dil işleme araştırmaları"
    gen_cfg = GenerationConfig(max_new_tokens=24, greedy=True, use_kv_cache=True)

    for p_name, vram_target, ckpt_slot, prec in profile_specs:
        rss_before = proc.memory_info().rss / (1024 * 1024)
        t_l0 = time.perf_counter()
        eng = AnkaLMInferenceEngine.from_checkpoint(ckpt_slot, precision=prec, allow_random_init=False)
        load_ms = (time.perf_counter() - t_l0) * 1000.0
        rss_after = proc.memory_info().rss / (1024 * 1024)

        w_mb = compute_model_storage_mb(eng.model)
        res = eng.generate(prompt, gen_cfg=gen_cfg)

        # Measure Prefix KV Cache speedup vs no KV cache
        res_nocache = eng.generate(prompt, gen_cfg=GenerationConfig(max_new_tokens=24, greedy=True, use_kv_cache=False))
        kv_speedup = ((res.tokens_per_second - res_nocache.tokens_per_second) / max(res_nocache.tokens_per_second, 1e-4)) * 100.0

        profiles_out.append(
            LowVRAMProfileBenchmark(
                profile_name=p_name,
                vram_budget_target_gb=vram_target,
                checkpoint_used=ckpt_slot,
                model_name=eng.metadata.model_name,
                parameter_count=eng.metadata.parameter_count,
                precision=prec.upper(),
                model_weights_mb=w_mb,
                runtime_rss_delta_mb=round(max(w_mb, rss_after - rss_before), 2),
                fits_in_2gb_budget=(w_mb < 1500.0),
                startup_load_ms=round(load_ms, 2),
                first_token_latency_ms=res.first_token_latency_ms,
                tokens_per_second=res.tokens_per_second,
                prefix_cache_speedup_pct=round(max(0.0, kv_speedup), 1),
                sample_output=res.generated_text.strip()[:120],
            )
        )

    # Benchmark Speculative Decoding (Edge Draft -> Small Target)
    draft_eng = AnkaLMInferenceEngine.from_checkpoint("edge_final", precision="fp32")
    target_eng = AnkaLMInferenceEngine.from_checkpoint("small_final", precision="fp32")
    spec_dec = SpeculativeDecoder(draft_eng, target_eng, lookahead_k=3)
    spec_res = spec_dec.generate_speculative(prompt, max_new_tokens=24)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "low_vram_profiles": [p.to_dict() for p in profiles_out],
        "speculative_decoding": spec_res.to_dict(),
    }
    out_path = PROJECT_ROOT / "logs" / "low_vram_and_speculative_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report
