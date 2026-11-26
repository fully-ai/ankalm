"""AnkaLM Independent Decoder-Only Transformer & Sparse MoE Architecture."""
from ankalm.core.config import ModelProfileConfig
from ankalm.model.layers import RMSNorm, RotaryEmbedding, apply_rotary_emb, SwiGLUFFN, GroupedQueryCausalAttention
from ankalm.model.transformer import AnkaLMTransformerBlock, AnkaLMForCausalLM
from ankalm.model.moe import MoERouter, SparseMoELayer, AnkaLMMoEBlock, AnkaLMMoEForCausalLM, MoERoutingDiagnostics


def build_ankalm_model(config: ModelProfileConfig) -> AnkaLMForCausalLM | AnkaLMMoEForCausalLM:
    """Factory that constructs either a Dense AnkaLMForCausalLM or a Sparse AnkaLMMoEForCausalLM."""
    if config.use_moe and config.num_experts > 1:
        return AnkaLMMoEForCausalLM(config)
    return AnkaLMForCausalLM(config)


__all__ = [
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary_emb",
    "SwiGLUFFN",
    "GroupedQueryCausalAttention",
    "AnkaLMTransformerBlock",
    "AnkaLMForCausalLM",
    "MoERouter",
    "SparseMoELayer",
    "AnkaLMMoEBlock",
    "AnkaLMMoEForCausalLM",
    "MoERoutingDiagnostics",
    "build_ankalm_model",
]
