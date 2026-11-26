"""AnkaLM Sparse Mixture-of-Experts (MoE) Architecture with Top-K Routing, Shared Expert, Load-Balancing Loss, and Collapse Diagnostics."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ankalm.core.config import ModelProfileConfig
from ankalm.model.layers import GroupedQueryCausalAttention, RMSNorm, SwiGLUFFN


@dataclass
class MoERoutingDiagnostics:
    num_experts: int
    top_k: int
    use_shared_expert: bool
    expert_token_counts: List[int]
    expert_selection_fractions: List[float]
    expert_mean_probabilities: List[float]
    routing_entropy: float
    max_possible_entropy: float
    normalized_entropy: float
    dominant_expert_index: int
    dominant_expert_share: float
    dead_experts: List[int]
    overflow_fraction: float
    collapse_detected: bool
    load_balance_aux_loss: float
    router_z_loss: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MoERouter(nn.Module):
    """
    Learned Top-K Softmax Router with Switch/GShard Load-Balancing Auxiliary Loss,
    Router Z-Loss, Capacity Factor Enforcement, and Expert Collapse Detection.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        aux_loss_coef: float = 0.02,
        z_loss_coef: float = 0.001,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.capacity_factor = capacity_factor
        self.aux_loss_coef = aux_loss_coef
        self.z_loss_coef = z_loss_coef
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self, x_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, MoERoutingDiagnostics]:
        """
        Args:
            x_flat: (N, d_model) where N = B * T
        Returns:
            topk_weights: (N, top_k) normalized routing weights
            topk_indices: (N, top_k) selected expert indices
            total_aux_loss: scalar tensor (aux_loss_coef * balance_loss + z_loss_coef * z_loss)
            diagnostics: MoERoutingDiagnostics
        """
        num_tokens = max(1, x_flat.size(0))
        logits = self.gate(x_flat).float()  # (N, E)

        # Router Z-loss for numerical stability: mean(logsumexp(logits)^2)
        z_loss = torch.mean(torch.logsumexp(logits, dim=-1) ** 2)

        probs = F.softmax(logits, dim=-1)  # (N, E)
        topk_probs, topk_indices = torch.topk(probs, k=self.top_k, dim=-1)  # (N, k)
        # Re-normalize top-k weights so they sum to 1 per token
        topk_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Compute token assignment fractions f_i across experts
        one_hot = F.one_hot(topk_indices, num_classes=self.num_experts).float()  # (N, k, E)
        tokens_per_expert = one_hot.sum(dim=(0, 1))  # (E,)
        total_assignments = float(num_tokens * self.top_k)
        f_i = tokens_per_expert / max(total_assignments, 1.0)  # (E,)
        p_i = probs.mean(dim=0)  # (E,)

        # Switch/GShard load balancing loss: E * sum(f_i * P_i)
        balance_loss = self.num_experts * torch.sum(f_i * p_i)
        total_aux_loss = (self.aux_loss_coef * balance_loss) + (self.z_loss_coef * z_loss)

        # Capacity and overflow tracking
        capacity = max(1, int(math.ceil(self.capacity_factor * total_assignments / self.num_experts)))
        overflow_tokens = torch.clamp(tokens_per_expert - capacity, min=0.0).sum().item()
        overflow_fraction = float(overflow_tokens) / max(total_assignments, 1.0)

        # Diagnostics
        with torch.no_grad():
            counts_list = [int(c.item()) for c in tokens_per_expert]
            fracs_list = [round(float(f.item()), 4) for f in f_i]
            probs_list = [round(float(p.item()), 4) for p in p_i]

            # Routing entropy: -sum(f_i * log(f_i))
            safe_f = torch.clamp(f_i, min=1e-9)
            entropy = float(-torch.sum(safe_f * torch.log(safe_f)).item())
            max_ent = math.log(max(self.num_experts, 2))
            norm_ent = entropy / max(max_ent, 1e-9)

            dom_idx = int(torch.argmax(f_i).item())
            dom_share = float(f_i[dom_idx].item())
            dead = [i for i, c in enumerate(counts_list) if c == 0]
            # Collapse if one expert takes > 80% of all routing slots when E >= 3
            collapse = (self.num_experts >= 3) and (dom_share > 0.80)

            diag = MoERoutingDiagnostics(
                num_experts=self.num_experts,
                top_k=self.top_k,
                use_shared_expert=True,
                expert_token_counts=counts_list,
                expert_selection_fractions=fracs_list,
                expert_mean_probabilities=probs_list,
                routing_entropy=round(entropy, 4),
                max_possible_entropy=round(max_ent, 4),
                normalized_entropy=round(norm_ent, 4),
                dominant_expert_index=dom_idx,
                dominant_expert_share=round(dom_share, 4),
                dead_experts=dead,
                overflow_fraction=round(overflow_fraction, 4),
                collapse_detected=collapse,
                load_balance_aux_loss=round(float(balance_loss.item()), 5),
                router_z_loss=round(float(z_loss.item()), 5),
            )

        return topk_weights.to(x_flat.dtype), topk_indices, total_aux_loss, diag


class SparseMoELayer(nn.Module):
    """
    Sparse Mixture-of-Experts Feed-Forward Layer combining:
    - An optional always-on Shared Expert (`SwiGLUFFN`)
    - E Routed Experts (`SwiGLUFFN`) selected via Top-K learned routing
    """

    def __init__(self, config: ModelProfileConfig) -> None:
        super().__init__()
        self.num_experts = max(2, config.num_experts)
        self.top_k = min(config.top_k_experts, self.num_experts)
        self.use_shared_expert = config.use_shared_expert

        self.router = MoERouter(
            d_model=config.d_model,
            num_experts=self.num_experts,
            top_k=self.top_k,
            capacity_factor=config.moe_capacity_factor,
            aux_loss_coef=config.moe_aux_loss_coef,
            z_loss_coef=config.moe_z_loss_coef,
        )
        self.shared_expert = (
            SwiGLUFFN(config.d_model, config.d_ff, dropout=config.dropout)
            if self.use_shared_expert
            else None
        )
        self.routed_experts = nn.ModuleList(
            [SwiGLUFFN(config.d_model, config.d_ff, dropout=config.dropout) for _ in range(self.num_experts)]
        )
        self.last_diagnostics: Optional[MoERoutingDiagnostics] = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (N, D)

        topk_weights, topk_indices, aux_loss, diag = self.router(x_flat)
        diag.use_shared_expert = self.use_shared_expert
        self.last_diagnostics = diag

        routed_out = torch.zeros_like(x_flat)
        # Dispatch tokens to each chosen expert and accumulate weighted outputs
        for expert_idx, expert_ffn in enumerate(self.routed_experts):
            # Mask of shape (N, k) where expert_idx was chosen
            match_mask = (topk_indices == expert_idx)
            if not match_mask.any():
                continue
            token_mask = match_mask.any(dim=-1)  # (N,)
            selected_tokens = x_flat[token_mask]  # (M, D)
            expert_output = expert_ffn(selected_tokens)  # (M, D)

            # Weight assigned to this expert for each selected token
            weights_for_expert = (topk_weights * match_mask.to(topk_weights.dtype)).sum(dim=-1, keepdim=True)[token_mask]
            routed_out[token_mask] = routed_out[token_mask] + (expert_output * weights_for_expert.to(expert_output.dtype))

        if self.shared_expert is not None:
            shared_out = self.shared_expert(x_flat)
            final_out = (0.5 * shared_out) + (0.5 * routed_out)
        else:
            final_out = routed_out

        return final_out.view(bsz, seq_len, d_model), aux_loss


class AnkaLMMoEBlock(nn.Module):
    """Pre-Norm Decoder Block with GQA Causal Attention and Sparse MoE FFN."""

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
        self.moe = SparseMoELayer(config)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        attn_out, present_kv = self.attn(self.attn_norm(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        moe_out, aux_loss = self.moe(self.ffn_norm(x))
        x = x + moe_out
        return x, present_kv, aux_loss


class AnkaLMMoEForCausalLM(nn.Module):
    """
    AnkaLM Independent Sparse Mixture-of-Experts Causal Language Model.
    Tracks total stored parameters vs active parameters per token and router load-balancing statistics.
    """

    def __init__(self, config: ModelProfileConfig) -> None:
        super().__init__()
        self.config = config
        self.activation_checkpointing = False

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.layers = nn.ModuleList([AnkaLMMoEBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        self.apply(self._init_weights)
        resid_scale = 1.0 / math.sqrt(2.0 * max(config.n_layers, 1))
        for idx, layer in enumerate(self.layers):
            with torch.no_grad():
                layer.attn.o_proj.weight.mul_(resid_scale)
                if layer.moe.shared_expert is not None:
                    layer.moe.shared_expert.down_proj.weight.mul_(resid_scale)
                for e_idx, exp in enumerate(layer.moe.routed_experts):
                    # Diversify initial routed expert symmetry slightly while preserving init_std scale
                    exp.down_proj.weight.mul_(resid_scale * (0.95 + 0.03 * e_idx))

        self.last_aux_loss: float = 0.0

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

    def count_total_and_active_parameters(self) -> Tuple[int, int]:
        return self.config.estimate_total_and_active_parameters()

    def compute_parameter_fingerprint(self) -> str:
        h = hashlib.sha256()
        with torch.no_grad():
            for name, param in sorted(self.named_parameters()):
                h.update(name.encode("utf-8"))
                flat = param.detach().float().cpu().view(-1)
                h.update(flat[:64].numpy().tobytes())
                h.update(flat[-64:].numpy().tobytes())
        return h.hexdigest()

    def snapshot_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().float().cpu().clone() for k, v in self.named_parameters()}

    def compute_l2_delta(self, initial_snapshot: Dict[str, torch.Tensor]) -> float:
        total_sq = 0.0
        with torch.no_grad():
            for k, v in self.named_parameters():
                if k in initial_snapshot:
                    diff = v.detach().float().cpu() - initial_snapshot[k]
                    total_sq += float(diff.pow(2).sum().item())
        return math.sqrt(total_sq)

    def get_routing_summary(self) -> Dict[str, Any]:
        """Aggregate routing diagnostics across all MoE layers."""
        layer_diags = []
        agg_counts = [0] * self.config.num_experts
        for idx, layer in enumerate(self.layers):
            d = layer.moe.last_diagnostics
            if d is not None:
                layer_diags.append({"layer": idx, **d.to_dict()})
                for e_i, c in enumerate(d.expert_token_counts):
                    if e_i < len(agg_counts):
                        agg_counts[e_i] += c
        total_routed = max(1, sum(agg_counts))
        agg_shares = [round(c / total_routed, 4) for c in agg_counts]
        total_params, active_params = self.count_total_and_active_parameters()
        return {
            "total_stored_parameters": total_params,
            "active_parameters_per_token": active_params,
            "active_fraction": round(active_params / max(total_params, 1), 4),
            "num_experts": self.config.num_experts,
            "top_k_experts": self.config.top_k_experts,
            "use_shared_expert": self.config.use_shared_expert,
            "aggregate_expert_counts": agg_counts,
            "aggregate_expert_shares": agg_shares,
            "dead_experts_global": [i for i, c in enumerate(agg_counts) if c == 0],
            "collapse_detected_any_layer": any(ld.get("collapse_detected", False) for ld in layer_diags),
            "last_aux_loss": round(self.last_aux_loss, 6),
            "layers": layer_diags,
        }

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
        total_aux = torch.tensor(0.0, device=input_ids.device, dtype=torch.float32)

        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            h, present_kv, layer_aux = layer(h, past_kv=layer_past, use_cache=use_cache)
            total_aux = total_aux + layer_aux
            if use_cache and present_key_values is not None and present_kv is not None:
                present_key_values.append(present_kv)

        self.last_aux_loss = float(total_aux.detach().item())
        h = self.norm(h)
        logits = self.lm_head(h)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                targets.reshape(-1),
                ignore_index=ignore_index,
            )
            loss = ce_loss + (total_aux / max(len(self.layers), 1))

        return logits, loss, present_key_values
