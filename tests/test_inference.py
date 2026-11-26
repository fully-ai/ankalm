"""Automated tests for AnkaLM Inference Engine and Quantization (Section 21: Inference)."""
from __future__ import annotations

import pytest

from ankalm.core.constants import PROJECT_ROOT
from ankalm.inference.engine import AnkaLMInferenceEngine, GenerationConfig


def test_inference_rejects_random_init_checkpoint() -> None:
    """Section 1.3 & 13: Never load random weights and present them as a trained model."""
    with pytest.raises(RuntimeError, match="randomly initialized weights"):
        AnkaLMInferenceEngine.from_checkpoint("micro_init", allow_random_init=False)


def test_inference_generation_streaming_and_sampling() -> None:
    engine = AnkaLMInferenceEngine.from_checkpoint("small_final", allow_random_init=False)
    cfg = GenerationConfig(
        max_new_tokens=20,
        temperature=0.6,
        top_k=30,
        top_p=0.9,
        repetition_penalty=1.15,
        seed=42,
        use_kv_cache=True,
    )
    prompt = "Türkiye ve Türkçe doğal dil işleme"
    res1 = engine.generate(prompt, gen_cfg=cfg)
    res2 = engine.generate(prompt, gen_cfg=cfg)

    assert res1.completion_tokens > 0
    assert len(res1.generated_text) > 0
    assert res1.tokens_per_second > 0
    # Deterministic seed check
    assert res1.generated_text == res2.generated_text

    # Streaming check
    chunks = list(engine.generate_stream(prompt, gen_cfg=cfg))
    assert len(chunks) >= 2
    assert chunks[-1]["finished"] is True


def test_inference_invalid_empty_prompt_and_long_context() -> None:
    engine = AnkaLMInferenceEngine.from_checkpoint("micro_final", allow_random_init=False)
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        engine.generate("   ")

    # Context window truncation check (prompt longer than max_seq_len)
    long_prompt = "Türkçe dil modeli testi " * 80
    res = engine.generate(long_prompt, gen_cfg=GenerationConfig(max_new_tokens=8, greedy=True))
    assert res.completion_tokens > 0
