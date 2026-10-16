#!/home/user/Desktop/sanskrit-ocr/.venv/bin/python
"""
Accuracy Verification & Multi-Model Comparative Tool for Sanskrit OCR
Compares:
1. IndicOCR-v2 (Attention-LSTM)
2. Tesseract 5.5 (Sanskrit 'san' model)
3. Tesseract 5.5 (Devanagari script model)
4. Morphological validation via sanskrit_parser
5. Ground truth alignment with Pingree's critical edition (Vol. 1 & Vol. 2)
"""

import sys
import unicodedata
import subprocess
from pathlib import Path
import cv2
import numpy as np
import requests

from ocr_pdf import predict_line, decode_output, clean_repetitions, DEFAULT_MODEL_URL
from vagdhenu import to_slp1, scan_meter, identify_meter


# Pingree Volume 2 English Translation & Sanskrit Ground Truth for Chapter 1 (verses 11-14)
PINGREE_GT_V14 = {
    "verse_id": "1.14 (Page 57)",
    "english_translation": (
        "The first (sign) is traditionally said to have the shape of a ram; "
        "it is called by the ancients the head of Kala. "
        "Its places are the paths of goats and sheep, caves, mountains, "
        "(hideouts of) thieves, and (places where there are) fire, metals, mines, and jewels."
    ),
    "sanskrit_lines": [
        "आद्यः स्मृतो मेषवपुस्ततोऽथ",
        "कालस्य मूर्द्धा कथितः पुराणैः ।",
        "स्थानान्यजाविकगोधराद्रि-",
        "स्तेनाग्निरत्नाकरलोहभाञ्जि ॥१४॥"
    ]
}

def lev_distance(s1, s2):
    s1 = unicodedata.normalize("NFC", s1.strip())
    s2 = unicodedata.normalize("NFC", s2.strip())
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n], len(s2)

def tesseract_ocr_line(img_np, lang="san"):
    cmd = ["/home/linuxbrew/.linuxbrew/bin/tesseract", "stdin", "stdout", "-l", lang, "--psm", "7"]
    success, buf = cv2.imencode(".png", img_np)
    if not success:
        return ""
    res = subprocess.run(cmd, input=buf.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return res.stdout.decode("utf-8").strip()

def run_morphological_check(word):
    try:
        from sanskrit_parser.parser.sandhi_analyzer import LexicalSandhiAnalyzer
        from sanskrit_parser.base.sanskrit_base import SanskritObject
        analyzer = LexicalSandhiAnalyzer()
        obj = SanskritObject(word)
        tags = analyzer.getMorphologicalTags(obj)
        return tags is not None and len(tags) > 0
    except Exception:
        return False

def compare_verse_14():
    print("=" * 70)
    print("ACCURACY COMPARISON & VERIFICATION: YAVANAJĀTAKA 1.14 (PAGE 57)")
    print("=" * 70)
    print(f"English Translation (Pingree Vol. 2):\n  \"{PINGREE_GT_V14['english_translation']}\"\n")

    crops = [f"/tmp/v14_{i}.png" for i in range(4)]
    gt_lines = PINGREE_GT_V14["sanskrit_lines"]

    total_gt_chars = 0
    total_indic_err = 0
    total_tess_san_err = 0
    total_tess_dev_err = 0
    total_byt5_err = 0

    try:
        from byt5_correction_pipeline import ByT5SanskritCorrector
        byt5_corrector = ByT5SanskritCorrector()
    except Exception:
        byt5_corrector = None
    for idx, (crop_path, gt) in enumerate(zip(crops, gt_lines), 1):
        if not Path(crop_path).exists():
            print(f"[!] Crop file {crop_path} missing.")
            continue
        roi = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)

        # 1. IndicOCR
        indic_text, _, _, _ = predict_line(roi)

        # 2. Tesseract Sanskrit
        tess_san = tesseract_ocr_line(roi, lang="san")

        # 3. Tesseract Devanagari
        tess_dev = tesseract_ocr_line(roi, lang="Devanagari")

        # 4. ByT5 Byte-Level Post-Correction (applied to Tesseract candidate)
        byt5_text = ""
        err_byt5 = 0
        if byt5_corrector:
            res = byt5_corrector.correct_line(tess_dev)
            byt5_text = res.corrected_text
            err_byt5, _ = lev_distance(byt5_text, gt)
            total_byt5_err += err_byt5
        # Error rates
        err_indic, l_gt = lev_distance(indic_text, gt)
        err_san, _ = lev_distance(tess_san, gt)
        err_dev, _ = lev_distance(tess_dev, gt)

        total_gt_chars += l_gt
        total_indic_err += err_indic
        total_tess_san_err += err_san
        total_tess_dev_err += err_dev

        slp1_gt = to_slp1(gt)
        w_gt, n_gt = scan_meter(slp1_gt)
        slp1_ind = to_slp1(indic_text)
        w_ind, n_ind = scan_meter(slp1_ind)

        print(f"\n[Quarter {idx}]")
        print(f"  Ground Truth:         {gt}")
        print(f"  IndicOCR-v2:          {indic_text:<35} | CER: {err_indic / l_gt:.1%}")
        print(f"  Tesseract (san):      {tess_san:<35} | CER: {err_san / l_gt:.1%}")
        print(f"  Tesseract (Devanagari): {tess_dev:<35} | CER: {err_dev / l_gt:.1%}")
        if byt5_text:
            slp1_byt5 = to_slp1(byt5_text)
            w_byt5, n_byt5 = scan_meter(slp1_byt5)
            print(f"  ByT5 Post-Correction: {byt5_text:<35} | CER: {err_byt5 / l_gt:.1%}")
            print(f"  Vāgdhenu Meter (ByT5):{''.join(w_byt5)} ({n_byt5} syl) -> {identify_meter(w_byt5)}")
        print(f"  Vāgdhenu Meter (GT):  {''.join(w_gt)} ({n_gt} syl) -> {identify_meter(w_gt)}")
        print(f"  Vāgdhenu Meter (OCR): {''.join(w_ind)} ({n_ind} syl)")
    print("\n" + "-" * 70)
    print("OVERALL ACCURACY SUMMARY (VERSE 1.14):")
    print(f"  IndicOCR-v2 CER:           {total_indic_err / total_gt_chars:.1%}")
    print(f"  Tesseract 'san' CER:       {total_tess_san_err / total_gt_chars:.1%}")
    print(f"  Tesseract 'Devanagari' CER:{total_tess_dev_err / total_gt_chars:.1%}")
    if byt5_corrector:
        print(f"  ByT5 Post-Corrected CER:   {total_byt5_err / total_gt_chars:.1%}")
    print("-" * 70)

    print("\nMORPHOLOGICAL SPELLCHECK DEMONSTRATION:")
    test_words = ["कालस्य", "कानास्य", "पुराणैः", "मेषः", "रत्नाकर"]
    for w in test_words:
        valid = run_morphological_check(w)
        status = "VALID SANSKRIT WORD" if valid else "REJECTED (Possible OCR error)"
        print(f"  {w:<12} -> {status}")
    print("=" * 70)

if __name__ == "__main__":
    compare_verse_14()
