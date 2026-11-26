"""Turkish Dataset Engine: acquisition, cleaning, deduplication, manifests, and PyTorch datasets."""
from ankalm.data.cleaning import (
    clean_document,
    identify_turkish_language,
    score_document_quality,
    turkish_lower,
    turkish_upper,
)
from ankalm.data.dedupe import DeduplicationEngine
from ankalm.data.manifest import DatasetManifestEntry, DatasetManifestReport, build_and_save_manifests
from ankalm.data.acquisition import acquire_all_turkish_datasets, prepare_corpus_splits
from ankalm.data.dataset import PackedTokenDataset, CausalTextDataset
from ankalm.data.instruction_data import build_turkish_instruction_dataset

__all__ = [
    "clean_document",
    "identify_turkish_language",
    "score_document_quality",
    "turkish_lower",
    "turkish_upper",
    "DeduplicationEngine",
    "DatasetManifestEntry",
    "DatasetManifestReport",
    "build_and_save_manifests",
    "acquire_all_turkish_datasets",
    "prepare_corpus_splits",
    "PackedTokenDataset",
    "CausalTextDataset",
    "build_turkish_instruction_dataset",
]
