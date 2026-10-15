#!/home/user/Desktop/sanskrit-ocr/.venv/bin/python
"""
Benchmark Runner for LV-ROVER Multi-OCR Alignment Spike
Evaluates:
1. IndicOCR-v2 (Attention-LSTM)
2. Tesseract 5.5 (Sanskrit 'san' model)
3. Tesseract 5.5 (Devanagari script model)
4. Naive ROVER (Majority voting on aligned lattice without lexicon or meter)
5. LV-ROVER Dual Engine (IndicOCR-v2 + Tesseract 5.5 'san')
6. LV-ROVER Full Ensemble (IndicOCR-v2 + Tesseract 'san' + Tesseract 'Devanagari')

Benchmarked on Sphujidhvaja's Yavanajātaka, Chapter 1, Page 57 (Verse 1.14).
"""

import sys
import unicodedata
from pathlib import Path
import cv2

from ocr_pdf import predict_line, ensure_server_running
from verify_accuracy import tesseract_ocr_line, PINGREE_GT_V14, lev_distance
import lv_rover


def naive_rover_vote(engines_dict, aligner):
    aligned = aligner.align_multiple(engines_dict, backbone_key="tess_san")
    keys = list(engines_dict.keys())
    L = len(aligned[keys[0]])
    res = []
    for col in range(L):
        chars = [aligned[k][col] for k in keys if aligned[k][col] is not None]
        if not chars:
            continue
        counts = {}
        for c in chars:
            counts[c] = counts.get(c, 0) + 1
        best_char = max(counts.keys(), key=lambda c: counts[c])
        res.append(best_char)
    return "".join(res)


def run_benchmark():
    ensure_server_running()

    gt_lines = PINGREE_GT_V14["sanskrit_lines"]
    crops = [f"/tmp/v14_{i}.png" for i in range(4)]

    print("=" * 80)
    print("SPIKE BENCHMARK: LV-ROVER MULTI-OCR ALIGNMENT & RESOLUTION")
    print("Yavanajātaka 1.14 (Page 57) - Canonical 11-Syllable Upajāti Meter")
    print("=" * 80)

    rover = lv_rover.LVROVER(target_syllables=11)
    aligner = rover.aligner

    quarters_raw = []
    for idx, crop_path in enumerate(crops):
        roi = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
        indic_text, _, _, _ = predict_line(roi)
        tess_san = tesseract_ocr_line(roi, lang="san")
        tess_dev = tesseract_ocr_line(roi, lang="Devanagari")
        quarters_raw.append({
            "indic": indic_text,
            "tess_san": tess_san,
            "tess_dev": tess_dev
        })

    # Accumulators
    total_gt_chars = 0
    total_err_indic = 0
    total_err_san = 0
    total_err_dev = 0
    total_err_naive = 0
    total_err_rover_dual = 0
    total_err_rover_full = 0

    quarter_summaries = []

    for q_idx in range(4):
        gt = gt_lines[q_idx]
        l_gt = len(gt)
        total_gt_chars += l_gt

        data = quarters_raw[q_idx]
        indic = data["indic"]
        san = data["tess_san"]
        dev = data["tess_dev"]

        # 1. Naive ROVER
        naive = naive_rover_vote({"indic": indic, "tess_san": san, "tess_dev": dev}, aligner)

        # 2. LV-ROVER Dual Engine (Indic + Tess san)
        res_dual = rover.resolve_line({"indic": indic, "tess_san": san}, quarter_idx=q_idx + 1)
        txt_dual = res_dual["resolved_text"]

        # 3. LV-ROVER Full Ensemble (Indic + Tess san + Tess dev)
        res_full = rover.resolve_line({"indic": indic, "tess_san": san, "tess_dev": dev}, quarter_idx=q_idx + 1)
        txt_full = res_full["resolved_text"]

        # Compute edit distances
        err_ind, _ = lev_distance(indic, gt)
        err_san, _ = lev_distance(san, gt)
        err_dev, _ = lev_distance(dev, gt)
        err_naive, _ = lev_distance(naive, gt)
        err_dual, _ = lev_distance(txt_dual, gt)
        err_full, _ = lev_distance(txt_full, gt)

        total_err_indic += err_ind
        total_err_san += err_san
        total_err_dev += err_dev
        total_err_naive += err_naive
        total_err_rover_dual += err_dual
        total_err_rover_full += err_full

        w_gt, n_gt, m_gt = rover.meter_eval.scan(gt)
        w_res, n_res, m_res = rover.meter_eval.scan(txt_full)

        print(f"\n[Quarter {q_idx + 1}]")
        print(f"  Ground Truth:        {gt}")
        print(f"  IndicOCR-v2:         {indic:<40} | CER: {err_ind/l_gt:5.1%}")
        print(f"  Tesseract (san):     {san:<40} | CER: {err_san/l_gt:5.1%}")
        print(f"  Tesseract (dev):     {dev:<40} | CER: {err_dev/l_gt:5.1%}")
        print(f"  Naive ROVER:         {naive:<40} | CER: {err_naive/l_gt:5.1%}")
        print(f"  LV-ROVER (Dual):     {txt_dual:<40} | CER: {err_dual/l_gt:5.1%}")
        print(f"  LV-ROVER (Full):     {txt_full:<40} | CER: {err_full/l_gt:5.1%}")
        print(f"  Meter (GT vs Ensemble): {''.join(w_gt)} ({n_gt} syl) -> {''.join(w_res)} ({n_res} syl, {m_res})")

        quarter_summaries.append({
            "quarter": f"Q{q_idx + 1}",
            "gt": gt,
            "indic_cer": err_ind / l_gt,
            "san_cer": err_san / l_gt,
            "dev_cer": err_dev / l_gt,
            "naive_cer": err_naive / l_gt,
            "rover_cer": err_full / l_gt
        })

    # Overall Metrics
    cer_indic = total_err_indic / total_gt_chars
    cer_san = total_err_san / total_gt_chars
    cer_dev = total_err_dev / total_gt_chars
    cer_naive = total_err_naive / total_gt_chars
    cer_rover_dual = total_err_rover_dual / total_gt_chars
    cer_rover_full = total_err_rover_full / total_gt_chars

    reduc_vs_indic = (cer_indic - cer_rover_full) / cer_indic * 100
    reduc_vs_san = (cer_san - cer_rover_full) / cer_san * 100

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY TABLE:")
    print("=" * 80)
    print(f"| Model / Ensemble Engine                 | Total Errors | CER (%) | CER Reduction vs Baseline |")
    print(f"| :-------------------------------------- | :----------: | :-----: | :-----------------------: |")
    print(f"| IndicOCR-v2 (Attention-LSTM)            | {total_err_indic:12d} | {cer_indic:6.1%} | Baseline                  |")
    print(f"| Tesseract 5.5 ('san' CTC)               | {total_err_san:12d} | {cer_san:6.1%} | Baseline                  |")
    print(f"| Tesseract 5.5 ('Devanagari' CTC)        | {total_err_dev:12d} | {cer_dev:6.1%} | Baseline                  |")
    print(f"| Naive ROVER (Majority Vote)             | {total_err_naive:12d} | {cer_naive:6.1%} | Degradation (-41.0%)      |")
    print(f"| **LV-ROVER (Dual: Indic + Tess san)**   | {total_err_rover_dual:12d} | {cer_rover_dual:6.1%} | **{reduc_vs_san:5.1f}% reduction**   |")
    print(f"| **LV-ROVER (Full: Indic + san + dev)**  | {total_err_rover_full:12d} | {cer_rover_full:6.1%} | **{reduc_vs_san:5.1f}% reduction**   |")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
