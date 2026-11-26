"""Automated tests for AnkaLM Turkish Tokenizer (Section 21: Tokenizer)."""
from __future__ import annotations

from ankalm.core.constants import SPECIAL_TOKEN_IDS, TOKEN_TO_ID, TURKISH_CHARS
from ankalm.data.cleaning import turkish_lower, turkish_upper
from ankalm.tokenizer.tokenizer import AnkaLMTokenizer
from ankalm.tokenizer.turkish_normalizer import verify_turkish_character_roundtrip


def test_tokenizer_turkish_characters_and_unicode(tokenizer: AnkaLMTokenizer) -> None:
    results = verify_turkish_character_roundtrip(tokenizer.encode, tokenizer.decode)
    for key, passed in results.items():
        assert passed, f"Turkish Unicode roundtrip failed for {key}"

    for ch in TURKISH_CHARS:
        ids = tokenizer.encode(ch)
        assert len(ids) >= 1
        assert tokenizer.decode(ids) == ch


def test_dotted_and_dotless_i_preservation(tokenizer: AnkaLMTokenizer) -> None:
    assert turkish_lower("İSTANBUL IĞDIR İZMİR ISPARTA") == "istanbul ığdır izmir ısparta"
    assert turkish_upper("istanbul ığdır izmir ısparta") == "İSTANBUL IĞDIR İZMİR ISPARTA"

    text = "Işık ile İğne, Irmak ile İstanbul arasında ı ve i farkı korunmalıdır."
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text
    assert "Işık" in decoded and "İğne" in decoded and "İstanbul" in decoded


def test_special_tokens_fixed_ids(tokenizer: AnkaLMTokenizer) -> None:
    for tok_str, expected_id in TOKEN_TO_ID.items():
        assert tokenizer.token_to_id(tok_str) == expected_id
        assert tokenizer.id_to_token(expected_id) == tok_str

    assert tokenizer.pad_id == SPECIAL_TOKEN_IDS["PAD"] == 0
    assert tokenizer.bos_id == SPECIAL_TOKEN_IDS["BOS"] == 1
    assert tokenizer.eos_id == SPECIAL_TOKEN_IDS["EOS"] == 2
    assert tokenizer.eot_id == SPECIAL_TOKEN_IDS["EOT"] == 3


def test_encode_decode_with_bos_eos_and_long_text(tokenizer: AnkaLMTokenizer) -> None:
    paragraph = (
        "Türkçe sondan eklemeli bir dildir. Araştırmacılarımız bilgisayarlarımızın "
        "üretkenliğini gözlemlediklerimizle ölçerek yerel yapay zeka modelleri eğitti. "
    )
    long_text = paragraph * 40
    ids = tokenizer.encode(long_text, add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert len(ids) > 400

    clean_decoded = tokenizer.decode(ids, skip_special_tokens=True)
    assert clean_decoded.strip() == long_text.strip()
