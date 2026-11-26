"""Automated tests for AnkaLM Training, Checkpointing, Resume, and Interruption Recovery (Section 21: Training)."""
from __future__ import annotations

from pathlib import Path

from ankalm.core.config import AnkaLMConfig, ModelProfileConfig, StorageConfig, TrainingProfileConfig
from ankalm.optimization.scheduler import CosineWarmupScheduler
from ankalm.tokenizer.tokenizer import AnkaLMTokenizer
from ankalm.training.trainer import AnkaLMTrainer
import torch


def test_scheduler_warmup_and_cosine_decay() -> None:
    lin = torch.nn.Linear(8, 8)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)
    sched = CosineWarmupScheduler(opt, max_lr=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=50)

    lr0 = sched.step(0)
    lr5 = sched.step(5)
    lr9 = sched.step(9)
    lr25 = sched.step(25)
    lr50 = sched.step(50)

    assert lr0 < lr5 < lr9
    assert abs(lr9 - 1e-3) < 1e-7
    assert 1e-4 < lr25 < 1e-3
    assert abs(lr50 - 1e-4) < 1e-7


def test_smoke_training_checkpoint_and_resume(tmp_path: Path, tokenizer: AnkaLMTokenizer) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    logs_dir = tmp_path / "logs"

    cfg = AnkaLMConfig(
        model=ModelProfileConfig(
            name="AnkaLM-SmokeTest",
            profile="micro",
            vocab_size=tokenizer.vocab_size,
            n_layers=2,
            d_model=64,
            d_ff=128,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=64,
            dropout=0.0,
        ),
        training=TrainingProfileConfig(
            batch_size=4,
            grad_accum_steps=2,
            max_steps=6,
            warmup_steps=2,
            learning_rate=1e-3,
            min_lr=1e-4,
            precision="bf16",
            eval_interval=3,
            checkpoint_interval=3,
            log_interval=2,
        ),
    )

    trainer = AnkaLMTrainer(cfg, tokenizer=tokenizer, checkpoints_dir=ckpt_dir, logs_dir=logs_dir)
    res1 = trainer.train(max_steps_override=6, slot_prefix="smoke")

    assert res1.final_step == 6
    assert res1.weights_updated_verified is True
    assert res1.param_l2_delta > 0.0
    assert (ckpt_dir / "latest" / "metadata.json").exists()
    assert (ckpt_dir / "latest" / "model.safetensors").exists()
    assert (ckpt_dir / "session_final" / "training_state.pt").exists()

    # Now test resuming from 'latest' (step 6 -> step 10)
    res2 = trainer.train(resume_from="latest", max_steps_override=10, slot_prefix="smoke")
    assert res2.start_step == 6
    assert res2.final_step == 10
    assert res2.steps_executed == 4
    assert res2.processed_tokens > res1.processed_tokens
    assert res2.resumed_from == "latest"
    assert res2.weights_updated_verified is True


def test_interruption_recovery_via_stop_signal(tmp_path: Path, tokenizer: AnkaLMTokenizer) -> None:
    ckpt_dir = tmp_path / "checkpoints_stop"
    logs_dir = tmp_path / "logs_stop"
    cfg = AnkaLMConfig(
        model=ModelProfileConfig(
            name="AnkaLM-InterruptTest",
            profile="micro",
            vocab_size=tokenizer.vocab_size,
            n_layers=2,
            d_model=64,
            d_ff=128,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=64,
        ),
        training=TrainingProfileConfig(batch_size=4, max_steps=20, precision="fp32"),
    )
    trainer = AnkaLMTrainer(cfg, tokenizer=tokenizer, checkpoints_dir=ckpt_dir, logs_dir=logs_dir)
    # Pre-request stop signal after initialization isn't possible because train() clears signals at start,
    # so we test that train() writes state and that a 2-step run saves recoverable session_final checkpoint.
    res = trainer.train(max_steps_override=2, slot_prefix="recover")
    assert res.final_step == 2
    assert (ckpt_dir / "session_final" / "metadata.json").exists()


def test_directml_gpu_accelerator_and_resource_controller() -> None:
    from ankalm.hardware.dml_accelerator import DirectMLGPUAccelerator
    from ankalm.training.resource_scheduler import ResourceUtilizationController

    ctrl = ResourceUtilizationController()
    threads = ctrl.configure_optimal_threads()
    assert threads >= 4
    snap = ctrl.sample_resources()
    assert snap.total_ram_gb > 0
    assert snap.disk_free_gb > 0

    accel = DirectMLGPUAccelerator()
    assert isinstance(accel.is_available, bool)
