"""
Unit & Integration Tests for ByT5 Post-OCR Correction Pipeline.

Covers:
- Metric calculations (CER, WER, Levenshtein distance).
- Devanagari UTF-8 byte budgeting and token length calculations.
- Repetition collapse & Unicode OpenType cluster preprocessing.
- Post-processing safety guardrails (byte length limits, drift detection, danda canonicalization).
- Page 57 Verse 1.14 verification dataset integrity.
- Live ByT5 model inference and error correction capabilities.
"""

import pytest
import unicodedata
from byt5_correction_pipeline import (
    ByT5SanskritCorrector,
    CorrectionResult,
    MetricsResult,
    compute_metrics,
    normalize_dandas,
    PINGREE_PAGE57_VERSE_14,
)


class TestMetricsAndNormalization:
    def test_compute_metrics_exact_match(self):
        m = compute_metrics("आद्यः स्मृतो मेषवपुस्ततोऽथ", "आद्यः स्मृतो मेषवपुस्ततोऽथ")
        assert m.cer == 0.0
        assert m.wer == 0.0
        assert m.edit_distance == 0
        assert m.exact_match is True

    def test_compute_metrics_substitution_and_deletion(self):
        # 1 substitution ('त' instead of 'थ') in 26 chars
        s1 = "आद्यः स्मृतो मेषवपुस्ततोऽत"
        gt = "आद्यः स्मृतो मेषवपुस्ततोऽथ"
        m = compute_metrics(s1, gt)
        assert m.edit_distance == 1
        assert m.exact_match is False
        assert 0.03 < m.cer < 0.05

    def test_normalize_dandas_pipe_conversion(self):
        # ASCII single and double pipes should become Devanagari dandas
        assert normalize_dandas("मेषवपुः |") == "मेषवपुः।"
        assert normalize_dandas("मेषवपुः ||") == "मेषवपुः॥"
        assert normalize_dandas("मेषवपुः || १४ ||") == "मेषवपुः॥ १४॥"

    def test_normalize_dandas_collapse_and_spacing(self):
        # Multiple dandas should collapse to ॥
        assert normalize_dandas("ततोऽथ ॥।") == "ततोऽथ॥"
        assert normalize_dandas("कालस्य मूर्द्धा ।कथितः") == "कालस्य मूर्द्धा। कथितः"
        # Leading danda should be stripped
        assert normalize_dandas("। स्थानान्यजाविक") == "स्थानान्यजाविक"


class TestByteBudgetingAndPreprocessing:
    def test_devanagari_utf8_byte_expansion(self):
        # Devanagari characters require 3 UTF-8 bytes each
        dev_str = "कालस्य मूर्द्धा कथितः पुराणैः ।"
        char_len = len(dev_str)
        byte_len = len(dev_str.encode("utf-8"))
        # Character count is 31, but byte count is ~79
        assert byte_len > (char_len * 2.2)

        corrector = ByT5SanskritCorrector()
        cleaned = corrector.preprocess(dev_str)
        assert "कालस्य" in cleaned and "पुराणैः" in cleaned

    def test_preprocess_collapses_attention_loops(self):
        corrector = ByT5SanskritCorrector()
        noisy = "सप्तछ्तहुट्यात्रस्यात्रस्त्रस्यस्त्रस्य"
        cleaned = corrector.preprocess(noisy)
        # Repetitions of length >=3 repeated 2+ times should collapse
        assert len(cleaned) <= len(noisy)

    def test_preprocess_strips_markdown_quotes(self):
        corrector = ByT5SanskritCorrector()
        line = "> मह्याक्रे मस्तम्नम्"
        assert corrector.preprocess(line) == "मह्याक्रे मस्तम्नम्"


class TestSafetyGuardrails:
    def test_safety_guardrail_catches_explosive_length(self):
        corrector = ByT5SanskritCorrector(max_drift_cer=1.25)
        raw_inp = "मह्याक्रे मस्तम्नम्"
        # Non-repetitive diverse runaway output exceeding 2.2x input bytes
        hallucinated = "कक्षायां प्रत्यहं याति समसूत्रानुसारतः तृतीयं लङ्काक्षितिजमुन्मण्डलम्"

        final_text, fallback, notes = corrector.postprocess(hallucinated, raw_inp)
        assert fallback is True
        assert final_text == raw_inp
        assert any("safety limit" in n for n in notes)

    def test_safety_guardrail_catches_extreme_drift(self):
        corrector = ByT5SanskritCorrector(max_drift_cer=0.9)
        raw_inp = "कालस्य मूर्द्धा कथितः पुराणैः ।"
        # Completely unrelated hallucinated sequence
        unrelated = "अथ योगानुशासनम् योगश्चित्तवृत्तिनिरोधः"

        final_text, fallback, notes = corrector.postprocess(unrelated, raw_inp)
        assert fallback is True
        assert final_text == raw_inp
        assert any("Model drift CER" in n for n in notes)

    def test_safety_guardrail_permits_reasonable_corrections(self):
        corrector = ByT5SanskritCorrector(max_drift_cer=1.25)
        raw_inp = "कान्यस्य सूघ्या राट: जराष्णैः।"
        good_correction = "कान्यस्य सूध्या राटः जराष्णैः।"

        final_text, fallback, notes = corrector.postprocess(good_correction, raw_inp)
        assert fallback is False
        assert final_text == good_correction
        assert len(notes) == 0


class TestPage57Verse14Dataset:
    def test_dataset_contains_all_four_quarters(self):
        quarters = PINGREE_PAGE57_VERSE_14["quarters"]
        assert isinstance(quarters, list)
        assert len(quarters) == 4
        quarter_ids = [q["id"] for q in quarters]  # type: ignore[index]
        assert quarter_ids == ["Q1", "Q2", "Q3", "Q4"]

    def test_dataset_quarter_attributes(self):
        quarters = PINGREE_PAGE57_VERSE_14["quarters"]
        assert isinstance(quarters, list)
        for q in quarters:
            assert "ground_truth" in q  # type: ignore[operator]
            assert "indicocr_v2" in q  # type: ignore[operator]
            assert "tesseract_san" in q  # type: ignore[operator]
            assert "tesseract_dev" in q  # type: ignore[operator]
            # Ground truth must be Devanagari
            gt = q["ground_truth"]  # type: ignore[index]
            assert any("\u0900" <= c <= "\u097F" for c in gt)


@pytest.mark.integration
class TestByT5LiveInference:
    @pytest.fixture(scope="module")
    def corrector(self):
        pytest.importorskip("transformers")
        pytest.importorskip("torch")
        corrector = ByT5SanskritCorrector()
        corrector.load_model()
        return corrector
    def test_live_single_line_correction(self, corrector):
        noisy = "कर्मण्येवाधिकारस्त मा फलषु कदाचन |"
        res = corrector.correct_line(noisy)
        assert res.raw_input == noisy
        assert len(res.corrected_text) > 0
        assert res.latency_sec > 0
        # Check that pipe was normalized to Devanagari danda
        assert "|" not in res.corrected_text
        assert "।" in res.corrected_text
        # Model should correct dropped matra 'फलषु' -> 'फलेषु'
        assert "फलेषु" in res.corrected_text

    def test_live_page57_verse14_tesseract_correction(self, corrector):
        # Quarter 2 from Tesseract Devanagari
        q2_raw = "कान्यस्य सूघ्या राट: जराष्णैः।"
        gt = "कालस्य मूर्द्धा कथितः पुराणैः ।"

        res = corrector.correct_line(q2_raw)
        m_raw = compute_metrics(q2_raw, gt)
        m_corr = compute_metrics(res.corrected_text, gt)

        # ByT5 correctly fixes 'सूघ्या' -> 'सूध्या' and ':' -> 'ः'
        assert "सूध्या" in res.corrected_text
        assert ":" not in res.corrected_text
        # CER should be improved or maintained
        assert m_corr.cer <= m_raw.cer
