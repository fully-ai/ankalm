"""Turkish text normalization, cleaning, language identification, and quality scoring."""
from __future__ import annotations

from dataclasses import dataclass
import html
import re
import unicodedata
from typing import Dict, Tuple

from ankalm.core.constants import TURKISH_SPECIFIC_CHARS

TURKISH_STOPWORDS = {
    "ve", "bir", "bu", "için", "ile", "olarak", "olan", "daha", "çok", "gibi",
    "kadar", "sonra", "ancak", "veya", "ama", "hem", "ise", "göre", "üzerine",
    "tarafından", "arasında", "büyük", "yeni", "önemli", "فقط", "değil", "var",
    "yok", "nasıl", "neden", "nedir", "Türkiye", "türkçe", "dil", "bilim", "veri",
    "sistem", "model", "üzerinde", "içinde", "olarak", "bulunur", "yapılır", "olur",
}

ENGLISH_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "can", "will", "into", "between", "system", "model", "data", "language",
}

HTML_TAG_RE = re.compile(r"<[^>]+>")
BOILERPLATE_PATTERNS = [
    re.compile(r"(?i)tüm hakları saklıdır\.?"),
    re.compile(r"(?i)çerez politikası|gizlilik sözleşmesi|kullanım koşulları"),
    re.compile(r"(?i)abone olmayı unutmayın|bizi takip edin|reklam alanı"),
    re.compile(r"(?i)copyright\s*©?\s*\d{4}"),
]
SPAM_PATTERNS = [
    re.compile(r"(?i)(hemen tıkla|bedava kazan|bahis|casino|viagra|kripto fırsat)"),
    re.compile(r"(https?://\S+\s*){4,}"),
]
MULTI_SPACE_RE = re.compile(r"[ \t\f\v]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def turkish_lower(text: str) -> str:
    """Lowercase text respecting Turkish dotted İ -> i and dotless I -> ı."""
    if not text:
        return text
    return text.replace("İ", "i").replace("I", "ı").lower()


def turkish_upper(text: str) -> str:
    """Uppercase text respecting Turkish dotted i -> İ and dotless ı -> I."""
    if not text:
        return text
    return text.replace("i", "İ").replace("ı", "I").upper()


def verify_valid_utf8(text: str) -> bool:
    """Detect invalid encoding or replacement characters."""
    if "\ufffd" in text or "\x00" in text:
        return False
    try:
        text.encode("utf-8", errors="strict")
        return True
    except UnicodeEncodeError:
        return False


def normalize_turkish_utf8(text: str) -> str:
    """Normalize UTF-8 in NFC form while strictly preserving Turkish İ/ı/i/I distinctions."""
    if not text:
        return ""
    # NFC preserves precomposed Turkish characters (ç, ğ, ı, İ, ö, ş, ü)
    normalized = unicodedata.normalize("NFC", text)
    # Fix accidental combining dot above after Latin capital I
    normalized = normalized.replace("I\u0307", "İ")
    return normalized


def clean_html_and_markdown(text: str, preserve_code_blocks: bool = True) -> str:
    """Strip HTML tags/entities and clean noisy markdown artifacts while preserving structure."""
    unescaped = html.unescape(text)
    if preserve_code_blocks and "```" in unescaped:
        parts = unescaped.split("```")
        cleaned_parts = []
        for idx, part in enumerate(parts):
            if idx % 2 == 0:
                cleaned_parts.append(HTML_TAG_RE.sub(" ", part))
            else:
                cleaned_parts.append(part)
        cleaned = "```".join(cleaned_parts)
    else:
        cleaned = HTML_TAG_RE.sub(" ", unescaped)

    for bp in BOILERPLATE_PATTERNS:
        cleaned = bp.sub("", cleaned)

    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = MULTI_SPACE_RE.sub(" ", cleaned)
    cleaned = MULTI_NEWLINE_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def detect_excessive_repetition(text: str) -> float:
    """Return repetition ratio (0.0 = unique, 1.0 = completely repetitive)."""
    words = text.split()
    if len(words) < 6:
        return 0.0
    # Check character run repetition (e.g. 'aaaaaaaaaa')
    if re.search(r"(.)\1{12,}", text):
        return 0.9
    # Check 3-gram repetition
    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 0.0
    unique_ratio = len(set(trigrams)) / len(trigrams)
    return round(1.0 - unique_ratio, 4)


def is_spam_document(text: str) -> bool:
    for pat in SPAM_PATTERNS:
        if pat.search(text):
            return True
    return False


def identify_turkish_language(text: str) -> Tuple[str, float]:
    """Identify whether a document is Turkish ('tr'), Bilingual ('tr-en'), or Other ('other')."""
    if not text or not text.strip():
        return ("other", 0.0)
    words = [turkish_lower(w.strip(".,!?;:()[]\"'")) for w in text.split() if w.strip()]
    if not words:
        return ("other", 0.0)

    tr_char_count = sum(1 for c in text if c in TURKISH_SPECIFIC_CHARS)
    alpha_count = max(sum(1 for c in text if c.isalpha()), 1)
    tr_char_density = tr_char_count / alpha_count

    tr_word_hits = sum(1 for w in words if w in TURKISH_STOPWORDS)
    en_word_hits = sum(1 for w in words if w in ENGLISH_STOPWORDS)

    # Check common Turkish morphological suffixes
    tr_suffixes = ("lar", "ler", "dir", "dır", "dur", "dür", "tir", "tır", "dan", "den", "tan", "ten", "인", "lık", "lik", "luk", "lük", "ması", "mesi", "arak", "erek")
    suffix_hits = sum(1 for w in words if len(w) >= 5 and w.endswith(tr_suffixes))

    tr_score = min(
        1.0,
        (tr_char_density * 14.0)
        + ((tr_word_hits / len(words)) * 3.5)
        + ((suffix_hits / len(words)) * 2.5),
    )
    en_score = min(1.0, (en_word_hits / len(words)) * 4.5)

    if tr_score >= 0.18 and en_score >= 0.12:
        return ("tr-en", round((tr_score + en_score) / 2, 3))
    if tr_score >= 0.15 or tr_char_count >= 2:
        return ("tr", round(max(tr_score, 0.5), 3))
    if en_score >= 0.2:
        return ("en", round(en_score, 3))
    return ("other", round(tr_score, 3))


@dataclass
class CleanedDocumentResult:
    cleaned_text: str
    is_valid: bool
    language: str
    language_confidence: float
    quality_score: float
    repetition_ratio: float
    rejection_reason: str = ""


def score_document_quality(text: str) -> float:
    """Compute a 0.0 - 1.0 quality score without penalizing rare Turkish or technical vocabulary."""
    if not text or len(text) < 25:
        return 0.1
    words = text.split()
    if len(words) < 4:
        return 0.2

    rep = detect_excessive_repetition(text)
    alpha_chars = sum(1 for c in text if c.isalpha() or c in " \n.,!?;:-()[]{}\"'0123456789+-*/=<>_")
    printable_ratio = alpha_chars / max(len(text), 1)
    has_sentence_end = any(text.rstrip().endswith(p) for p in (".", "!", "?", "```", "}", "]")) or "." in text

    score = 0.55
    if len(words) >= 12:
        score += 0.15
    if has_sentence_end:
        score += 0.15
    if printable_ratio >= 0.92:
        score += 0.15
    score -= rep * 0.6
    return round(max(0.0, min(1.0, score)), 4)


def clean_document(
    raw_text: str,
    allow_bilingual: bool = True,
    min_quality: float = 0.35,
    min_chars: int = 25,
) -> CleanedDocumentResult:
    """Run full cleaning pipeline on a raw document."""
    if not raw_text or not verify_valid_utf8(raw_text):
        return CleanedDocumentResult("", False, "invalid", 0.0, 0.0, 1.0, "invalid_encoding")

    norm = normalize_turkish_utf8(raw_text)
    cleaned = clean_html_and_markdown(norm)
    if len(cleaned) < min_chars:
        return CleanedDocumentResult(cleaned, False, "short", 0.0, 0.1, 0.0, "too_short")

    if is_spam_document(cleaned):
        return CleanedDocumentResult(cleaned, False, "spam", 0.0, 0.0, 0.0, "spam_detected")

    rep_ratio = detect_excessive_repetition(cleaned)
    if rep_ratio > 0.55:
        return CleanedDocumentResult(cleaned, False, "repetitive", 0.0, 0.1, rep_ratio, "excessive_repetition")

    lang, lang_conf = identify_turkish_language(cleaned)
    allowed_langs = {"tr", "tr-en"} if allow_bilingual else {"tr"}
    if lang not in allowed_langs:
        return CleanedDocumentResult(cleaned, False, lang, lang_conf, 0.2, rep_ratio, f"unsupported_language:{lang}")

    q_score = score_document_quality(cleaned)
    if q_score < min_quality:
        return CleanedDocumentResult(cleaned, False, lang, lang_conf, q_score, rep_ratio, "low_quality_score")

    return CleanedDocumentResult(
        cleaned_text=cleaned,
        is_valid=True,
        language=lang,
        language_confidence=lang_conf,
        quality_score=q_score,
        repetition_ratio=rep_ratio,
        rejection_reason="",
    )
