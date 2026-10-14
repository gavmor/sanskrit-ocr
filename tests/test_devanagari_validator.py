"""
Unit tests for Devanagari Unicode Validator & Sanitizer
Tests cluster grammar, HarfBuzz compatibility, and OCR decoding.
"""

import pytest
from devanagari_validator import (
    sanitize_devanagari_unicode,
    validate_devanagari_line,
    INDEP_TO_DEP_VOWELS,
    DECOMPOSED_VOWELS,
    decode_output,
)
from ocr_pdf import clean_repetitions


class TestDevanagariSanitizer:
    def test_recompose_decomposed_vowels(self):
        # अ + ा -> आ
        assert sanitize_devanagari_unicode("\u0905\u093Eह्न") == "आह्न"
        # अ + ु -> उ
        assert sanitize_devanagari_unicode("\u0905\u0941पासन") == "उपासन"

    def test_strip_orphan_leading_virama(self):
        # Orphan virama at line/word start: ्वब्द -> वब्द
        assert sanitize_devanagari_unicode("्वब्दशमन्यंत्") == "वब्दशमन्यंत्"
        assert sanitize_devanagari_unicode("> ्वब्द") == "> वब्द"

    def test_virama_before_matra(self):
        # श् + ु -> शु
        assert sanitize_devanagari_unicode("श्ुद्रार्दुर्येष्टम्") == "शुद्रार्दुर्येष्टम्"
        assert sanitize_devanagari_unicode("क्ि") == "कि"
        assert sanitize_devanagari_unicode("त्े") == "ते"

    def test_virama_before_independent_vowel(self):
        # र् + इ -> रि
        assert sanitize_devanagari_unicode("सस्तिर्इच") == "सस्तिरिच"
        # क् + उ -> कु
        assert sanitize_devanagari_unicode("क्उ") == "कु"

    def test_inverted_anusvara_matra(self):
        # Anusvara followed by matra: दंु -> दुं
        assert sanitize_devanagari_unicode("दंुक्रो") == "दुंक्रो"

    def test_avagraha_combining_marks(self):
        # Avagraha cannot take virama or matras
        assert sanitize_devanagari_unicode("स्वाट्यिऽ्निय") == "स्वाट्यिऽनिय"
        assert sanitize_devanagari_unicode("शोर्णृदेष्टृषृणोऽे") == "शोर्णृदेष्टृषृणोऽ"

    def test_strip_orphan_trailing_virama(self):
        # Standalone consonant with virama at line end
        assert sanitize_devanagari_unicode("शुद्रार्दुर्येष्टम्। र्") == "शुद्रार्दुर्येष्टम्।"

    def test_normalize_dandas(self):
        # Consecutive dandas and spacing
        assert sanitize_devanagari_unicode("तत्राद्विन्न्न्प्या न्न्धा ॥।३॥") == "तत्राद्विन्न्न्प्या न्न्धा॥ ३॥"
        assert sanitize_devanagari_unicode("शोक्यते ।") == "शोक्यते।"


class TestDevanagariValidator:
    def test_clean_string_is_valid(self):
        clean = "आद्यः स्मृतो मेषवपुस्ततोऽथ कालस्य मूर्द्धा कथितः पुराणैः।"
        is_valid, issues, _ = validate_devanagari_line(clean)
        assert is_valid is True
        assert len(issues) == 0

    def test_flag_virama_before_matra(self):
        dirty = "श्ुद्र"
        is_valid, issues, _ = validate_devanagari_line(dirty)
        assert is_valid is False
        assert "Virama+VowelSign" in issues

    def test_flag_leading_orphan_virama(self):
        dirty = "्वब्द"
        is_valid, issues, _ = validate_devanagari_line(dirty)
        assert is_valid is False
        assert "OrphanLeadingMark" in issues


class TestOCRDecoding:
    def test_decode_sample_word(self):
        # 2343 (ध) 2352 (र्) 2381 (्) 2350 (म) 2366 (ा) ... -> धर्माब्धिस्थमुदाहरणम्
        codepoints = "234323522381235023662348238123432367236023812341235023692342236623612352233923502381"
        decoded = decode_output(codepoints)
        assert decoded == "धर्माब्धिस्थमुदाहरणम्"

    def test_clean_repetitions(self):
        looped = "कुषीवन्नात्राम् द्वीच्यि न्निल्टी तुन्ति तुन्ति तुन्ति तुन्ति तुन्ति"
        cleaned = clean_repetitions(looped)
        assert "तुन्ति तुन्ति तुन्ति" not in cleaned
        assert cleaned.startswith("कुषीवन्नात्राम्")
