"""
Unit tests for Vāgdhenu integration:
Metrical scansion, Chandaḥ identification, syllabification, and text preprocessing.
"""

import pytest
import vagdhenu


def test_transliteration_to_slp1():
    deva = "आद्यः स्मृतो मेषवपुस्ततोऽथ"
    slp1 = vagdhenu.to_slp1(deva)
    assert "AdyaH" in slp1
    assert "mezavapu" in slp1


def test_scan_indravajra_meter():
    # Yavanajataka 1.14 pada 1 is canonical Indravajra (11 syllables)
    pada = "आद्यः स्मृतो मेषवपुस्ततोऽथ"
    slp1 = vagdhenu.to_slp1(pada)
    weights, n_syl = vagdhenu.scan_meter(slp1, pada_final_guru=True)
    pattern = "".join(weights)
    assert n_syl == 11
    assert pattern == "GGLGGLLGLGG"
    assert vagdhenu.identify_meter(weights) == "indravajrā(11)"


def test_scan_anushtubh_pada():
    # Canonical 8-syllable Anushtubh pada
    pada = "शुक्लाम्बरधरं विष्णुं"
    slp1 = vagdhenu.to_slp1(pada)
    weights, n_syl = vagdhenu.scan_meter(slp1, pada_final_guru=True)
    assert n_syl == 8


def test_syllabification():
    slp1 = "DarmAbDisTamudAharaRam"
    syls = vagdhenu.syllabify(slp1)
    assert len(syls) == 9
    vagdhenu.tag_weights(syls)
    assert all("weight" in s for s in syls)


def test_prep_text_colon_to_visarga():
    # Latin colon typo converted to Sanskrit visarga
    raw = "गुरु:-"
    assert vagdhenu.fix_colon(raw) == "गुरुः"


def test_prep_text_strip_punct():
    raw = "धर्माब्धिस्थमुदाहरणम् ॥ १४ ॥"
    stripped = vagdhenu.strip_punct(raw)
    assert "॥" not in stripped
    assert "१४" not in stripped
    assert "धर्माब्धिस्थमुदाहरणम्" in stripped
