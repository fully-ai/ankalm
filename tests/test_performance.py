"""Automated performance tests measuring tokens/sec, memory, model load, first-token latency, and checkpoint save time."""
from __future__ import annotations

from pathlib import Path
import time

from ankalm.core.config import AnkaLMConfig, ModelProfileConfig, TrainingProfileConfig
from ankalm.hardware.detector import estimate_training_memory
from ankalm.inference.engine import AnkaLMInferenceEngine, GenerationConfig
from ankalm.model.transformer import AnkaLMForCausalLM
from ankalm.storage.checkpoint import CheckpointManager
from ankalm.tokenizer.tokenizer import AnkaLMTokenizer


def test_performance_metrics_and_latencies(tmp_path: Path, tokenizer: AnkaLMTokenizer) -> None:
    # 1. Model loading time
    t0 = time.perf_counter()
    engine = AnkaLMInferenceEngine.from_checkpoint("small_final", allow_random_init=False)
    load_time_ms = (time.perf_counter() - t0) * 1000.0
    assert load_time_ms < 10000.0

    # 2. First-token latency & tokens per second
    res = engine.generate("Türkiye ve bilim", gen_cfg=GenerationConfig(max_new_tokens=16, greedy=True))
    assert res.first_token_latency_ms > 0.0
    assert res.tokens_per_second > 5.0

    # 3. Memory estimation
    est = estimate_training_memory(engine.model.config, TrainingProfileConfig(batch_size=8, precision="bf16"))
    assert est.total_estimated_mb > est.weights_mb
    assert est.fits_in_ram is True

    # 4. Atomic checkpoint save time
    ckpt_mgr = CheckpointManager(tmp_path / "perf_ckpts")
    cfg = AnkaLMConfig(
        model=ModelProfileConfig(
            name="AnkaLM-Perf",
            profile="micro",
            vocab_size=tokenizer.vocab_size,
            n_layers=2,
            d_model=64,
            d_ff=128,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=64,
        )
    )
    m = AnkaLMForCausalLM(cfg.model)
    t_save0 = time.perf_counter()
    ckpt_mgr.save_checkpoint(
        slot_name="perf_slot",
        model=m,
        optimizer=None,
        scheduler=None,
        scaler=None,
        config=cfg,
        global_step=1,
        epoch=0,
        processed_tokens=64,
        train_loss=5.0,
        val_loss=5.0,
        val_perplexity=148.4,
        initial_fingerprint="init_fp",
        param_l2_delta=1.0,
        tokenizer_meta={"version": "1.0.0", "vocab_size": tokenizer.vocab_size, "vocab_fingerprint": "fp"},
        dataset_manifest_summary={},
        dataset_sampling_state={},
        maturity_level="smoke_trained",
    )
    save_time_ms = (time.perf_counter() - t_save0) * 1000.0
    assert save_time_ms < 5000.0
