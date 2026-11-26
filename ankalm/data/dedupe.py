"""Exact and near-duplicate filtering engine using SHA-256 and character/word n-gram Jaccard shingles."""
from __future__ import annotations

import hashlib
import re
from typing import Dict, Iterable, List, Set, Tuple

from ankalm.data.cleaning import turkish_lower

WORD_SPLIT_RE = re.compile(r"\w+", re.UNICODE)


def compute_exact_hash(text: str) -> str:
    norm = " ".join(turkish_lower(text).split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def compute_shingles(text: str, k: int = 4) -> Set[int]:
    """Compute hashed k-word shingles (or character 4-grams for short texts) for near-duplicate detection."""
    norm = turkish_lower(text)
    words = WORD_SPLIT_RE.findall(norm)
    if len(words) < 20:
        chars = "".join(words)
        grams = [chars[i : i + 4] for i in range(max(1, len(chars) - 3))]
    else:
        grams = [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]
    # Use deterministic 64-bit hash via blake2b
    return {
        int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "little")
        for g in grams
    }


def jaccard_similarity(set_a: Set[int], set_b: Set[int]) -> float:
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


class DeduplicationEngine:
    """Stateful deduplication engine supporting exact, near-duplicate, and cross-split leakage checks."""

    def __init__(self, near_dup_threshold: float = 0.82) -> None:
        self.near_dup_threshold = near_dup_threshold
        self.seen_hashes: Set[str] = set()
        self.seen_sources: Set[str] = set()
        self.shingle_index: List[Tuple[str, Set[int]]] = []
        self.exact_duplicates_removed = 0
        self.near_duplicates_removed = 0

    def is_duplicate(self, text: str, doc_id: str = "") -> Tuple[bool, str]:
        """Return (is_dup, reason) and register the document if it is unique."""
        exact_h = compute_exact_hash(text)
        if exact_h in self.seen_hashes:
            self.exact_duplicates_removed += 1
            return (True, "exact_duplicate")

        shingles = compute_shingles(text)
        for existing_id, existing_shingles in self.shingle_index:
            # Fast size ratio pruning before set intersection
            la, lb = len(shingles), len(existing_shingles)
            if la == 0 or lb == 0:
                continue
            if min(la, lb) / max(la, lb) < self.near_dup_threshold:
                continue
            sim = jaccard_similarity(shingles, existing_shingles)
            if sim >= self.near_dup_threshold:
                self.near_duplicates_removed += 1
                return (True, f"near_duplicate_of:{existing_id}:{sim:.2f}")

        self.seen_hashes.add(exact_h)
        self.shingle_index.append((doc_id or exact_h[:10], shingles))
        return (False, "")

    def check_leakage_against(self, candidate_text: str) -> bool:
        """Check whether candidate_text leaks any document already indexed in this engine (without adding it)."""
        exact_h = compute_exact_hash(candidate_text)
        if exact_h in self.seen_hashes:
            return True
        shingles = compute_shingles(candidate_text)
        for _, existing_shingles in self.shingle_index:
            la, lb = len(shingles), len(existing_shingles)
            if la == 0 or lb == 0:
                continue
            if min(la, lb) / max(la, lb) < self.near_dup_threshold:
                continue
            if jaccard_similarity(shingles, existing_shingles) >= self.near_dup_threshold:
                return True
        return False
