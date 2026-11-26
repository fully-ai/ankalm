"""Automated tests for AnkaLM Decoder-Only Transformer Architecture (Section 21: Model)."""
from __future__ import annotations

import torch

from ankalm.core.config import ModelProfileConfig
from ankalm.model.transformer import AnkaLMForCausalLM
from ankalm.optimization.scheduler import build_adamw_optimizer


def test_model_random_initialization_and_param_count(micro_model_config: ModelProfileConfig) -> None:
    m1 = AnkaLMForCausalLM(micro_model_config)
    m2 = AnkaLMForCausalLM(micro_model_config)
    assert m1.count_parameters() == micro_model_config.estimate_parameters()
    # Two independent random initializations without identical seeds must differ
    assert m1.compute_parameter_fingerprint() != m2.compute_parameter_fingerprint()
    # Weight tying check
    assert m1.lm_head.weight.data_ptr() == m1.tok_embeddings.weight.data_ptr()


def test_model_forward_backward_loss_and_weight_update(micro_model_config: ModelProfileConfig) -> None:
    torch.manual_seed(123)
    model = AnkaLMForCausalLM(micro_model_config)
    init_snapshot = model.snapshot_weights()
    init_fp = model.compute_parameter_fingerprint()

    optimizer = build_adamw_optimizer(model, lr=1e-2, weight_decay=0.01)
    x = torch.randint(0, micro_model_config.vocab_size, (4, 16))
    y = torch.randint(0, micro_model_config.vocab_size, (4, 16))

    logits, loss, _ = model(x, targets=y)
    assert logits.shape == (4, 16, micro_model_config.vocab_size)
    assert loss is not None and torch.isfinite(loss)

    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    optimizer.step()
    optimizer.zero_grad()

    after_fp = model.compute_parameter_fingerprint()
    l2_delta = model.compute_l2_delta(init_snapshot)
    assert after_fp != init_fp
    assert l2_delta > 0.0

    # Verify loss decreases on repeated optimization steps over the same batch
    for _ in range(8):
        _, l_step, _ = model(x, targets=y)
        l_step.backward()
        optimizer.step()
        optimizer.zero_grad()
    _, final_loss, _ = model(x, targets=y)
    assert float(final_loss.item()) < float(loss.item())


def test_causal_mask_invariance(micro_model_config: ModelProfileConfig) -> None:
    """Verify causal attention: modifying future tokens (t > k) must never alter logits at position k."""
    model = AnkaLMForCausalLM(micro_model_config)
    model.eval()

    x1 = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]], dtype=torch.long)
    x2 = torch.tensor([[10, 20, 30, 40, 999, 888, 777, 666]], dtype=torch.long)

    with torch.no_grad():
        logits1, _, _ = model(x1)
        logits2, _, _ = model(x2)

    # Positions 0, 1, 2, 3 are identical in x1 and x2; their output logits must match within floating point tolerance
    assert torch.allclose(logits1[:, :4, :], logits2[:, :4, :], atol=1e-5)
    # Position 4 sees different tokens and must differ
    assert not torch.allclose(logits1[:, 4:, :], logits2[:, 4:, :], atol=1e-3)
