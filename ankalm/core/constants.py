"""Project-wide constants, special token definitions, and maturity levels."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Mandatory special tokens with fixed IDs that never change once pretraining starts
SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|eot|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|unk|>",
]

SPECIAL_TOKEN_IDS = {
    "PAD": 0,
    "BOS": 1,
    "EOS": 2,
    "EOT": 3,
    "SYSTEM": 4,
    "USER": 5,
    "ASSISTANT": 6,
    "TOOL": 7,
    "UNK": 8,
}

TOKEN_TO_ID = {
    "<|pad|>": 0,
    "<|bos|>": 1,
    "<|eos|>": 2,
    "<|eot|>": 3,
    "<|system|>": 4,
    "<|user|>": 5,
    "<|assistant|>": 6,
    "<|tool|>": 7,
    "<|unk|>": 8,
}

ID_TO_TOKEN = {v: k for k, v in TOKEN_TO_ID.items()}

TURKISH_CHARS = ("ç", "Ç", "ğ", "Ğ", "ı", "I", "İ", "i", "ö", "Ö", "ş", "Ş", "ü", "Ü")
TURKISH_SPECIFIC_CHARS = set("çÇğĞıİöÖşŞüÜ")

# Mandatory distinct model maturity stages (Section 0.1 & 1.3)
CHECKPOINT_MATURITY_LEVELS = {
    "architecture_config_only": "Architecture configuration only (unallocated/untrained large-scale blueprint - NOT trained)",
    "random_init": "Random initialization (untrained weights - NOT inference ready)",
    "smoke_trained": "Smoke-trained checkpoint (pipeline verification only)",
    "experimental_base": "Experimental base model (early pretraining checkpoint)",
    "pretrained": "Pretrained base model (trained on Turkish pretraining corpus)",
    "instruction_tuned": "Instruction-tuned model (SFT on Turkish instruction dataset)",
    "distilled_edge": "Distilled Edge model (trained via knowledge distillation from AnkaLM teacher)",
    "moe_trained": "Trained Sparse Mixture-of-Experts (AnkaLM-MoE) model",
    "vision_trained": "Trained AnkaLM-Vision multimodal encoder and projector model",
    "evaluated": "Evaluated model (verified across Turkish Evaluation Lab suite)",
}

DATASET_CATEGORIES = [
    "general_turkish",
    "daily_conversation",
    "formal_writing",
    "education",
    "technology",
    "programming",
    "science",
    "mathematics",
    "explanations",
    "question_answer",
    "literature",
    "bilingual_tr_en",
]
