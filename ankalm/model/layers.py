"""Core neural network building blocks for AnkaLM: RMSNorm, RoPE, GQA Causal Attention, and SwiGLU."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Pre-Norm stability)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return (self.weight * normed).to(orig_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE) for relative position encoding in attention."""

    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        self.max_seq_len = max(self.max_seq_len, seq_len)
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        total_len = seq_len + offset
        if total_len > self.cos_cached.size(0):
            self._build_cache(total_len * 2)
        cos = self.cos_cached[offset:total_len]
        sin = self.sin_cached[offset:total_len]
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors of shape (B, H, T, D)."""
    cos_u = cos.unsqueeze(0).unsqueeze(0).to(q.dtype)
    sin_u = sin.unsqueeze(0).unsqueeze(0).to(q.dtype)
    q_rot = (q * cos_u) + (_rotate_half(q) * sin_u)
    k_rot = (k * cos_u) + (_rotate_half(k) * sin_u)
    return q_rot, k_rot


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network: down_proj(SiLU(gate_proj(x)) * up_proj(x))."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(activated))


class GroupedQueryCausalAttention(nn.Module):
    """
    Causal Multi-Head / Grouped-Query Attention (GQA) with RoPE, SDPA acceleration,
    and optional KV Cache for fast autoregressive generation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int = 512,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.num_queries_per_kv = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max(max_seq_len * 2, 1024), theta=rope_theta)
        self.resid_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = x.shape
        offset = past_kv[0].size(2) if past_kv is not None else 0

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(seq_len, offset=offset)
        q, k = apply_rotary_emb(q, k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        present_kv = (k, v) if use_cache else None

        # Expand KV heads if Grouped-Query Attention (n_kv_heads < n_heads)
        if self.num_queries_per_kv > 1:
            k_attn = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v_attn = v.repeat_interleave(self.num_queries_per_kv, dim=1)
        else:
            k_attn, v_attn = k, v

        # Use PyTorch optimized scaled_dot_product_attention
        if offset == 0 and seq_len > 1:
            attn_out = F.scaled_dot_product_attention(
                q,
                k_attn,
                v_attn,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        elif seq_len == 1 and offset > 0:
            # Single token decoding with past KV cache attends to all cached positions
            attn_out = F.scaled_dot_product_attention(
                q,
                k_attn,
                v_attn,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            # Chunked prefill with existing offset: construct explicit causal mask
            total_k = k_attn.size(2)
            mask = torch.ones(seq_len, total_k, dtype=torch.bool, device=x.device).tril(diagonal=offset)
            attn_out = F.scaled_dot_product_attention(
                q,
                k_attn,
                v_attn,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )

        out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.resid_dropout(self.o_proj(out)), present_kv
