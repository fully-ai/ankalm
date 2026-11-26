"""Pytest fixtures for the AnkaLM automated test suite."""
from __future__ import annotations

from pathlib import Path
import sys
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ankalm.core.config import AnkaLMConfig, ModelProfileConfig, TrainingProfileConfig, PREDEFINED_MODEL_PROFILES
from ankalm.model import build_ankalm_model
from ankalm.storage.checkpoint import CheckpointManager
from ankalm.tokenizer.tokenizer import AnkaLMTokenizer


@pytest.fixture(scope="session")
def tokenizer() -> AnkaLMTokenizer:
    return AnkaLMTokenizer.load(PROJECT_ROOT / "tokenizer")


@pytest.fixture(scope="session", autouse=True)
def ensure_ci_smoke_checkpoints(tokenizer: AnkaLMTokenizer) -> None:
    """
    When running in a clean CI clone where binary .safetensors/.pt files are excluded by .gitignore,
    materialize lightweight 1-step smoke-verified checkpoints for the slots referenced by integration tests.
    On local workstations where trained checkpoints already exist, this fixture is a no-op.
    """
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_mgr = CheckpointManager(ckpt_dir)
    tok_meta = {
        "version": "1.0.0",
        "vocab_size": tokenizer.vocab_size,
        "vocab_fingerprint": "ci_smoke",
    }

    slots = [
        ("micro_init", "micro", 0, "random_init"),
        ("micro_final", "micro", 2, "smoke_trained"),
        ("small_final", "small", 2, "pretrained"),
        ("moe_final", "moe", 2, "moe_trained"),
        ("edge_final", "edge_2gb", 2, "distilled_edge"),
    ]
    for slot_name, profile_key, step, maturity in slots:
        safetensors_path = ckpt_dir / slot_name / "model.safetensors"
        if safetensors_path.exists():
            continue
        base_prof = PREDEFINED_MODEL_PROFILES.get(profile_key, PREDEFINED_MODEL_PROFILES["micro"])
        m_cfg = ModelProfileConfig(
            name=base_prof.name,
            profile=base_prof.profile,
            model_family=base_prof.model_family,
            vocab_size=tokenizer.vocab_size,
            n_layers=2,
            d_model=64,
            d_ff=128,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=128,
            use_moe=base_prof.use_moe,
            num_experts=base_prof.num_experts,
            top_k_experts=base_prof.top_k_experts,
            use_shared_expert=base_prof.use_shared_expert,
        )
        cfg = AnkaLMConfig(model=m_cfg, training=TrainingProfileConfig(batch_size=2, max_steps=max(step, 1)))
        model = build_ankalm_model(m_cfg)
        init_fp = model.compute_parameter_fingerprint()
        l2_delta = 0.0
        if step > 0:
            x = torch.randint(0, tokenizer.vocab_size, (2, 16))
            y = torch.randint(0, tokenizer.vocab_size, (2, 16))
            opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
            opt.zero_grad()
            _, loss, _ = model(x, targets=y)
            loss.backward()
            opt.step()
            l2_delta = 1.25
        ckpt_mgr.save_checkpoint(
            slot_name=slot_name,
            model=model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            config=cfg,
            global_step=step,
            epoch=1 if step > 0 else 0,
            processed_tokens=step * 32,
            train_loss=4.5,
            val_loss=4.5,
            val_perplexity=90.0,
            initial_fingerprint=init_fp if step > 0 else model.compute_parameter_fingerprint(),
            param_l2_delta=l2_delta,
            tokenizer_meta=tok_meta,
            dataset_manifest_summary={},
            dataset_sampling_state={},
            maturity_level=maturity,
            update_latest_and_best=(slot_name == "small_final"),
        )


@pytest.fixture
def micro_model_config(tokenizer: AnkaLMTokenizer) -> ModelProfileConfig:
    return ModelProfileConfig(
        name="AnkaLM-Micro-Test",
        profile="micro",
        vocab_size=tokenizer.vocab_size,
        n_layers=2,
        d_model=64,
        d_ff=192,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        dropout=0.0,
    )
