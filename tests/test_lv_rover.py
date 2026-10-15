"""
Unit & Integration Tests for LV-ROVER Multi-OCR Alignment & Metrical Resolution
Tests:
- Needleman-Wunsch character & sequence alignment
- Vāgdhenu metrical constraint evaluation & 11-syllable Upajāti scoring
- Attention repetition loop pruning
- Sanskrit morphological lexicon verification
- Syllable deficit bridging from CTC span
- End-to-end CER reduction benchmark on Sphujidhvaja's Yavanajātaka 1.14
"""

import pytest
import lv_rover
from verify_accuracy import lev_distance, PINGREE_GT_V14


def test_needleman_wunsch_alignment():
    aligner = lv_rover.NeedlemanWunschAligner()
    s1 = "कालस्य"
    s2 = "कात्यस्य"
    al1, al2 = aligner.align_pair(s1, s2)
    # Ensure characters align and length is preserved
    assert len(al1) == len(al2)
    assert al1[0] == 'क' and al2[0] == 'क'
    assert al1[-1] == 'य' and al2[-1] == 'य'


def test_needleman_wunsch_gap_alignment():
    aligner = lv_rover.NeedlemanWunschAligner()
    full_text = "आद्यः स्मृतो मेषवपुस्ततोऽथ"
    dropout_text = "आद्यः स्मृतो"
    al1, al2 = aligner.align_pair(full_text, dropout_text)
    assert len(al1) == len(al2)
    # The tail of dropout_text should contain gaps (None)
    assert any(x is None for x in al2)


def test_metrical_evaluator_indravajra():
    evaluator = lv_rover.MetricalConstraintEvaluator(target_syllables=11)
    pada = "आद्यः स्मृतो मेषवपुस्ततोऽथ"
    w, n, meter = evaluator.scan(pada)
    assert n == 11
    assert "".join(w) == "GGLGGLLGLGG"
    assert meter == "indravajrā(11)"
    score = evaluator.score(pada)
    assert score > 100.0  # High positive score for exact Indravajrā match


def test_metrical_evaluator_penalizes_deficit_and_surplus():
    evaluator = lv_rover.MetricalConstraintEvaluator(target_syllables=11)
    dropout = "मह्याक्रे मस्तम्नम्"  # 6 syllables
    runaway = "कानास्य श्रा राट्रानैहः कुटादैस्त्यः कुराणान्त्यः कु"  # 17 syllables

    score_dropout = evaluator.score(dropout)
    score_runaway = evaluator.score(runaway)
    score_canonical = evaluator.score("कालस्य मूर्द्धा कथितः पुराणैः ।")

    # Canonical 11-syllable line must vastly outscore dropout and runaway
    assert score_canonical > score_dropout
    assert score_canonical > score_runaway


def test_prune_repetition():
    rover = lv_rover.LVROVER()
    loop_text = "कानास्य श्रा राट्रानैहः कुटादैस्त्यः कुराणान्त्यः कुराणान्त्यः कुराणान्त्यः कु"
    pruned = rover.prune_repetition(loop_text)
    assert "कुराणान्त्यः कुराणान्त्यः" not in pruned


def test_lexicon_verifier():
    verifier = lv_rover.LexiconVerifier()
    # Canonical valid Sanskrit words
    assert verifier.is_valid_word("कालस्य")
    assert verifier.is_valid_word("रत्नाकर")
    assert verifier.is_valid_word("पुराणैः")
    assert verifier.is_valid_word("आद्यः")
    assert verifier.is_valid_word("स्मृतो")
    # Common OCR corruption non-words rejected
    assert not verifier.is_valid_word("कानास्य")
    assert not verifier.is_valid_word("गदिप्त्वः")
    assert not verifier.is_valid_word("स््पृत्यो")
    assert not verifier.is_valid_word("ञताय्यः")

def test_end_to_end_verse_14_cer_reduction():
    rover = lv_rover.LVROVER()
    gt_lines = PINGREE_GT_V14["sanskrit_lines"]

    # OCR predictions from Chapter 1 Page 57 Plate
    sample_engine_data = [
        # Quarter 1
        {
            "indic": "मह्याक्रे मस्तम्नम्",
            "tess_san": "ञताय्यः स््पृत्यो सेष्वस्स्बा- तस्तः",
            "tess_dev": "अराय्यः स्ख्ूतो सेष्यस्तस्ब्रातस्लि ः"
        },
        # Quarter 2
        {
            "indic": "कानास्य श्रा राट्रानैहः कुटादैस्त्यः कुराणान्त्यः कु",
            "tess_san": "कात्यस्य श्वा गदिप्त्वः प्राणैः ।",
            "tess_dev": "कान्यस्य सूघ्या राट: जराष्णैः।"
        },
        # Quarter 3
        {
            "indic": "स्तो जातास्तंच्छलम्",
            "tess_san": "स्तो 5ज्नाधिकास्वैन्रकृन्यराद्धि--",
            "tess_dev": "स्तो ऽञजाजिकास्तंनरकुन्स्‌राद्रि-"
        },
        # Quarter 4
        {
            "indic": "स्नित्र गाकरस्रस्त्यस्त्राक्ष",
            "tess_san": "स्वेता र्ितिश्उात्व्वाकररत्तमजस्ति - ॥2४६॥",
            "tess_dev": "स्तेत्ास्नि्पात्याकर रन्नम्जस्मि - ॥१४॥"
        }
    ]

    res = rover.resolve_verse(sample_engine_data)
    assert res["verse_meter"].startswith("Indravajrā") or "Upajāti" in res["verse_meter"]
    assert res["total_syllables"] in (43, 44)

    total_gt = 0
    total_san_err = 0
    total_rover_err = 0

    for i in range(4):
        gt = gt_lines[i]
        san = sample_engine_data[i]["tess_san"]
        resolved = res["resolved_lines"][i]

        err_san, l = lev_distance(san, gt)
        err_rover, _ = lev_distance(resolved, gt)

        total_gt += l
        total_san_err += err_san
        total_rover_err += err_rover

    san_cer = total_san_err / total_gt
    rover_cer = total_rover_err / total_gt

    # Assert significant CER reduction
    assert rover_cer < 0.10  # Dramatic error reduction to < 10%
    assert rover_cer < san_cer
