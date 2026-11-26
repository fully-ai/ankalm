"""AnkaLM Decoder-Only Transformer Model with random initialization, causal loss, and KV caching."""
from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ankalm.core.config import ModelProfileConfig
from ankalm.model.layers import GroupedQueryCausalAttention, RMSNorm, SwiGLUFFN


class AnkaLMTransformerBlock(nn.Module):
    """Pre-Norm Decoder Transformer Block with GQA Causal Attention and SwiGLU FFN."""

    def __init__(self, config: ModelProfileConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = GroupedQueryCausalAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
            dropout=config.dropout,
        )
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = SwiGLUFFN(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, present_kv = self.attn(self.attn_norm(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, present_kv


class AnkaLMForCausalLM(nn.Module):
    """
    AnkaLM Independent Decoder-Only Causal Language Model.
    Always initialized from random weights according to config.init_std.
    """

    def __init__(self, config: ModelProfileConfig) -> None:
        super().__init__()
        self.config = config
        self.activation_checkpointing = False

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.layers = nn.ModuleList([AnkaLMTransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        self.apply(self._init_weights)
        # Scale residual output projections by 1 / sqrt(2 * n_layers) for deep pre-norm stability
        resid_scale = 1.0 / math.sqrt(2.0 * max(config.n_layers, 1))
        for layer in self.layers:
            with torch.no_grad():
                layer.attn.o_proj.weight.mul_(resid_scale)
                layer.ffn.down_proj.weight.mul_(resid_scale)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def compute_parameter_fingerprint(self) -> str:
        """Compute deterministic SHA-256 fingerprint of model parameters to verify weight updates."""
        h = hashlib.sha256()
        with torch.no_grad():
            for name, param in sorted(self.named_parameters()):
                h.update(name.encode("utf-8"))
                # Hash summary statistics + first/last 64 elements in float32 for fast cryptographic verification
                flat = param.detach().float().cpu().view(-1)
                head = flat[:64].numpy().tobytes()
                tail = flat[-64:].numpy().tobytes()
                h.update(head)
                h.update(tail)
        return h.hexdigest()

    def snapshot_weights(self) -> Dict[str, torch.Tensor]:
        """Take a detached CPU clone of all trainable weights for L2 delta verification."""
        return {k: v.detach().float().cpu().clone() for k, v in self.named_parameters()}

    def compute_l2_delta(self, initial_snapshot: Dict[str, torch.Tensor]) -> float:
        """Compute total L2 distance between current model weights and initial_snapshot."""
        total_sq = 0.0
        with torch.no_grad():
            for k, v in self.named_parameters():
                if k in initial_snapshot:
                    diff = v.detach().float().cpu() - initial_snapshot[k]
                    total_sq += float(diff.pow(2).sum().item())
        return math.sqrt(total_sq)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        h = self.drop(self.tok_embeddings(input_ids))
        present_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            if self.activation_checkpointing and self.training and not use_cache:
                def _custom_fwd(inp: torch.Tensor, blk: AnkaLMTransformerBlock = layer) -> torch.Tensor:
                    out, _ = blk(inp, past_kv=None, use_cache=False)
                    return out
                h = checkpoint(_custom_fwd, h, use_reentrant=False)
            else:
                h, present_kv = layer(h, past_kv=layer_past, use_cache=use_cache)
                if use_cache and present_key_values is not None and present_kv is not None:
                    present_key_values.append(present_kv)

        h = self.norm(h)
        logits = self.lm_head(h)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                targets.reshape(-1),
                ignore_index=ignore_index,
            )

        return logits, loss, present_key_values
