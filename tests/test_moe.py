"""Automated tests for AnkaLM-MoE (Sparse Mixture-of-Experts), Low-Memory Optimizers, and Speculative Decoding (Section 20)."""
from __future__ import annotations

import torch

from ankalm.core.config import ModelProfileConfig, PREDEFINED_MODEL_PROFILES
from ankalm.inference.engine import AnkaLMInferenceEngine
from ankalm.inference.speculative import SpeculativeDecoder
from ankalm.model.moe import AnkaLMMoEForCausalLM
from ankalm.optimization import AdamW8bit, Adafactor, measure_optimizer_state_mb


def test_moe_active_vs_total_parameters_and_routing_gradients() -> None:
    cfg = ModelProfileConfig(
        name="AnkaLM-MoE-UnitTest",
        profile="moe",
        model_family="moe",
        vocab_size=256,
        n_layers=2,
        d_model=64,
        d_ff=128,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        use_moe=True,
        num_experts=4,
        top_k_experts=2,
        use_shared_expert=True,
        moe_aux_loss_coef=0.02,
    )
    model = AnkaLMMoEForCausalLM(cfg)
    total_stored, active_per_tok = model.count_total_and_active_parameters()
    actual_params = model.count_parameters()

    assert total_stored == actual_params
    assert active_per_tok < total_stored

    x = torch.randint(0, 256, (6, 16))
    y = torch.randint(0, 256, (6, 16))
    logits, loss, _ = model(x, targets=y)
    assert logits.shape == (6, 16, 256)
    assert loss is not None and torch.isfinite(loss)
    assert model.last_aux_loss > 0.0

    loss.backward()
    # Verify router gate, shared expert, and routed experts receive gradients
    for layer in model.layers:
        assert layer.moe.router.gate.weight.grad is not None
        assert layer.moe.shared_expert.gate_proj.weight.grad is not None
        grads_present = sum(1 for exp in layer.moe.routed_experts if exp.gate_proj.weight.grad is not None)
        assert grads_present >= 2

    summary = model.get_routing_summary()
    assert summary["total_stored_parameters"] == total_stored
    assert summary["active_parameters_per_token"] == active_per_tok
    assert len(summary["aggregate_expert_counts"]) == 4
    assert sum(summary["aggregate_expert_counts"]) > 0


def test_trained_moe_checkpoint_inference_and_no_collapse() -> None:
    eng = AnkaLMInferenceEngine.from_checkpoint("moe_final", allow_random_init=False)
    res = eng.generate("Türkçe doğal dil işleme")
    assert res.completion_tokens > 0
    assert isinstance(eng.model, AnkaLMMoEForCausalLM)
    diag = eng.model.get_routing_summary()
    assert diag["collapse_detected_any_layer"] is False
    assert len(diag["dead_experts_global"]) == 0


def test_low_memory_optimizers_8bit_adamw_and_adafactor() -> None:
    cfg = ModelProfileConfig(vocab_size=256, n_layers=2, d_model=64, d_ff=128, n_heads=4, n_kv_heads=2, max_seq_len=16)
    x = torch.randint(0, 256, (4, 16))
    y = torch.randint(0, 256, (4, 16))

    torch.manual_seed(7)
    m_adamw = AnkaLMMoEForCausalLM(cfg)
    opt_fp32 = torch.optim.AdamW(m_adamw.parameters(), lr=2e-3)
    _, l0, _ = m_adamw(x, targets=y)
    l0.backward()
    opt_fp32.step()
    mb_fp32 = measure_optimizer_state_mb(opt_fp32)

    torch.manual_seed(7)
    m_8bit = AnkaLMMoEForCausalLM(cfg)
    opt_8bit = AdamW8bit(m_8bit.parameters(), lr=2e-3)
    for _ in range(4):
        opt_8bit.zero_grad()
        _, l8, _ = m_8bit(x, targets=y)
        l8.backward()
        opt_8bit.step()
    mb_8bit = measure_optimizer_state_mb(opt_8bit)
    assert mb_8bit < mb_fp32 * 0.35

    torch.manual_seed(7)
    m_ada = AnkaLMMoEForCausalLM(cfg)
    opt_ada = Adafactor(m_ada.parameters(), lr=2e-3)
    _, la, _ = m_ada(x, targets=y)
    la.backward()
    opt_ada.step()
    mb_ada = measure_optimizer_state_mb(opt_ada)
    assert mb_ada < mb_fp32 * 0.15


def test_speculative_decoding_exact_quality_equivalence() -> None:
    draft_eng = AnkaLMInferenceEngine.from_checkpoint("edge_final", allow_random_init=False)
    target_eng = AnkaLMInferenceEngine.from_checkpoint("small_final", allow_random_init=False)
    spec = SpeculativeDecoder(draft_eng, target_eng, lookahead_k=3)
    res = spec.generate_speculative("Türkiye'nin başkenti", max_new_tokens=16)
    assert res.total_generated_tokens > 0
    assert res.exact_quality_equivalence is True
    assert res.target_forward_calls <= res.total_generated_tokens
