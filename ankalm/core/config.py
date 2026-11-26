"""Configuration management for AnkaLM model family (8B, 4B, Edge, Dense, MoE), training, and storage."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import yaml

from ankalm.core.constants import PROJECT_ROOT


@dataclass
class ModelProfileConfig:
    name: str = "AnkaLM-Small"
    profile: str = "small"
    arch_version: str = "2.0"
    model_family: str = "dense"  # dense, moe, edge, balanced_4b, flagship_8b
    vocab_size: int = 8192
    n_layers: int = 8
    d_model: int = 384
    d_ff: int = 1024
    n_heads: int = 6
    n_kv_heads: int = 2
    max_seq_len: int = 256
    dropout: float = 0.05
    activation: str = "swiglu"
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-5
    pos_encoding: str = "rope"
    rope_theta: float = 10000.0
    rope_scaling_factor: float = 1.0
    sliding_window: Optional[int] = None
    tie_word_embeddings: bool = True
    init_std: float = 0.02
    # Sparse Mixture-of-Experts (MoE) settings
    use_moe: bool = False
    num_experts: int = 1
    top_k_experts: int = 1
    use_shared_expert: bool = True
    moe_capacity_factor: float = 1.25
    moe_aux_loss_coef: float = 0.01
    moe_z_loss_coef: float = 0.001

    def __init__(self, **kwargs: Any) -> None:
        # Backward-compatible initializer ignoring unknown legacy keys if any
        defaults = {
            "name": "AnkaLM-Small",
            "profile": "small",
            "arch_version": "2.0",
            "model_family": "dense",
            "vocab_size": 8192,
            "n_layers": 8,
            "d_model": 384,
            "d_ff": 1024,
            "n_heads": 6,
            "n_kv_heads": 2,
            "max_seq_len": 256,
            "dropout": 0.05,
            "activation": "swiglu",
            "norm_type": "rmsnorm",
            "norm_eps": 1e-5,
            "pos_encoding": "rope",
            "rope_theta": 10000.0,
            "rope_scaling_factor": 1.0,
            "sliding_window": None,
            "tie_word_embeddings": True,
            "init_std": 0.02,
            "use_moe": False,
            "num_experts": 1,
            "top_k_experts": 1,
            "use_shared_expert": True,
            "moe_capacity_factor": 1.25,
            "moe_aux_loss_coef": 0.01,
            "moe_z_loss_coef": 0.001,
        }
        defaults.update({k: v for k, v in kwargs.items() if k in defaults})
        for k, v in defaults.items():
            setattr(self, k, v)

    def estimate_total_and_active_parameters(self) -> Tuple[int, int]:
        """
        Accurately calculate (total_stored_parameters, active_parameters_per_token).
        For dense models, total == active.
        For MoE models, total includes all E routed experts (+ shared expert + router),
        while active includes only top_k routed experts (+ shared expert + router).
        """
        head_dim = self.d_model // self.n_heads
        embed_params = self.vocab_size * self.d_model
        kv_dim = self.n_kv_heads * head_dim
        attn_params = (self.d_model * self.d_model) + 2 * (self.d_model * kv_dim) + (self.d_model * self.d_model)
        single_ffn_params = (3 if self.activation.lower() == "swiglu" else 2) * self.d_model * self.d_ff
        norm_params = 2 * self.d_model

        if not self.use_moe or self.num_experts <= 1:
            layer_total = attn_params + single_ffn_params + norm_params
            layer_active = layer_total
        else:
            router_params = self.d_model * self.num_experts
            shared_params = single_ffn_params if self.use_shared_expert else 0
            total_ffn = shared_params + (self.num_experts * single_ffn_params) + router_params
            active_ffn = shared_params + (min(self.top_k_experts, self.num_experts) * single_ffn_params) + router_params
            layer_total = attn_params + total_ffn + norm_params
            layer_active = attn_params + active_ffn + norm_params

        head_extra = 0 if self.tie_word_embeddings else (self.vocab_size * self.d_model)
        total_params = embed_params + (self.n_layers * layer_total) + self.d_model + head_extra
        active_params = embed_params + (self.n_layers * layer_active) + self.d_model + head_extra
        return (total_params, active_params)

    def estimate_parameters(self) -> int:
        return self.estimate_total_and_active_parameters()[0]


@dataclass
class TrainingProfileConfig:
    batch_size: int = 16
    grad_accum_steps: int = 2
    max_steps: int = 600
    warmup_steps: int = 40
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    precision: str = "bf16"  # fp32, fp16, bf16
    optimizer_type: str = "adamw"  # adamw, adamw_8bit, adafactor
    eval_interval: int = 50
    checkpoint_interval: int = 100
    log_interval: int = 10
    num_workers: int = 0
    pin_memory: bool = False
    activation_checkpointing: bool = False
    compile_model: bool = False
    pack_sequences: bool = True
    seed: int = 42


@dataclass
class StorageConfig:
    raw_data_dir: str = "datasets/raw"
    processed_data_dir: str = "datasets/processed"
    tokenized_data_dir: str = "datasets/tokenized"
    manifests_dir: str = "datasets/manifests"
    validation_data_dir: str = "datasets/validation"
    evaluation_data_dir: str = "datasets/evaluation"
    tokenizer_dir: str = "tokenizer"
    checkpoints_dir: str = "checkpoints"
    models_dir: str = "models"
    exports_dir: str = "exports"
    logs_dir: str = "logs"
    cache_dir: str = "cache"
    conversations_db: str = "cache/conversations.sqlite3"
    rag_dir: str = "cache/rag"
    documents_dir: str = "cache/documents"
    min_free_disk_gb: float = 2.0

    def resolve(self, root: Path = PROJECT_ROOT) -> Dict[str, Path]:
        return {
            k: (root / v).resolve()
            for k, v in asdict(self).items()
            if isinstance(v, str)
        }


@dataclass
class AnkaLMConfig:
    model: ModelProfileConfig = field(default_factory=ModelProfileConfig)
    training: TrainingProfileConfig = field(default_factory=TrainingProfileConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    hardware_profile: str = "auto"
    api_host: str = "127.0.0.1"
    api_port: int = 8420

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AnkaLMConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        model_cfg = ModelProfileConfig(**raw.get("model", {}))
        train_raw = {k: v for k, v in (raw.get("training", {}) or {}).items() if k in TrainingProfileConfig.__dataclass_fields__}
        storage_raw = {k: v for k, v in (raw.get("storage", {}) or {}).items() if k in StorageConfig.__dataclass_fields__}
        train_cfg = TrainingProfileConfig(**train_raw)
        storage_cfg = StorageConfig(**storage_raw)
        return cls(
            model=model_cfg,
            training=train_cfg,
            storage=storage_cfg,
            hardware_profile=raw.get("hardware_profile", "auto"),
            api_host=raw.get("api_host", "127.0.0.1"),
            api_port=int(raw.get("api_port", 8420)),
        )

    def to_yaml(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False, allow_unicode=True)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


PREDEFINED_MODEL_PROFILES: Dict[str, ModelProfileConfig] = {
    "micro": ModelProfileConfig(
        name="AnkaLM-Micro",
        profile="micro",
        model_family="dense",
        vocab_size=8192,
        n_layers=2,
        d_model=128,
        d_ff=384,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=128,
        dropout=0.0,
    ),
    "edge": ModelProfileConfig(
        name="AnkaLM-Edge",
        profile="edge",
        model_family="edge",
        vocab_size=8192,
        n_layers=4,
        d_model=192,
        d_ff=512,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=256,
        dropout=0.02,
    ),
    "tiny": ModelProfileConfig(
        name="AnkaLM-Tiny",
        profile="tiny",
        model_family="dense",
        vocab_size=8192,
        n_layers=4,
        d_model=256,
        d_ff=768,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=256,
        dropout=0.02,
    ),
    "small": ModelProfileConfig(
        name="AnkaLM-Small",
        profile="small",
        model_family="dense",
        vocab_size=8192,
        n_layers=8,
        d_model=384,
        d_ff=1024,
        n_heads=6,
        n_kv_heads=2,
        max_seq_len=256,
        dropout=0.05,
    ),
    "moe": ModelProfileConfig(
        name="AnkaLM-MoE",
        profile="moe",
        model_family="moe",
        vocab_size=8192,
        n_layers=6,
        d_model=256,
        d_ff=512,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=256,
        dropout=0.03,
        use_moe=True,
        num_experts=4,
        top_k_experts=2,
        use_shared_expert=True,
        moe_capacity_factor=1.25,
        moe_aux_loss_coef=0.02,
        moe_z_loss_coef=0.001,
    ),
    "extended": ModelProfileConfig(
        name="AnkaLM-Extended",
        profile="extended",
        model_family="dense",
        vocab_size=8192,
        n_layers=12,
        d_model=512,
        d_ff=1536,
        n_heads=8,
        n_kv_heads=4,
        max_seq_len=512,
        dropout=0.05,
    ),
    "edge_300m": ModelProfileConfig(
        name="AnkaLM-Edge-300M",
        profile="edge_300m",
        model_family="edge",
        vocab_size=8192,
        n_layers=16,
        d_model=1024,
        d_ff=3584,
        n_heads=16,
        n_kv_heads=4,
        max_seq_len=1024,
        dropout=0.05,
    ),
    "balanced_4b": ModelProfileConfig(
        name="AnkaLM-4B",
        profile="balanced_4b",
        model_family="balanced_4b",
        vocab_size=8192,
        n_layers=28,
        d_model=3072,
        d_ff=10240,
        n_heads=24,
        n_kv_heads=8,
        max_seq_len=4096,
        dropout=0.0,
    ),
    "flagship_8b": ModelProfileConfig(
        name="AnkaLM-8B",
        profile="flagship_8b",
        model_family="flagship_8b",
        vocab_size=8192,
        n_layers=32,
        d_model=4096,
        d_ff=15360,
        n_heads=32,
        n_kv_heads=8,
        max_seq_len=4096,
        dropout=0.0,
    ),
}
