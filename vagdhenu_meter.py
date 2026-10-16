#!/home/user/Desktop/sanskrit-ocr/.venv/bin/python
"""
Vāgdhenu Meter Scanner & Chandaḥ-Parīkṣā Tool for Sanskrit OCR
Scans verses into per-akṣara Laghu (L) / Guru (G) weight sequences,
identifies classical Sanskrit meters, and flags syllable count anomalies.
"""

import sys
import argparse
from pathlib import Path

from vagdhenu import (
    to_slp1,
    scan_meter,
    identify_meter,
    METER_SIGNATURES,
    syllabify,
    tag_weights,
    normalize_sanskrit,
    detect_meter,
)

def scan_pada_meter(pada_text):
    """
    Scans a single pāda (quarter-verse). Returns:
      - slp1: SLP1 transliteration
      - weights: string of 'L' (laghu) and 'G' (guru)
      - n_syllables: syllable count
      - meter_name: identified meter name
    """
    slp1 = to_slp1(pada_text, "devanagari")
    weights, n_syl = scan_meter(slp1, pada_final_guru=True)
    pattern = "".join(weights)
    meter_name = identify_meter(weights)
    return {
        "text": pada_text,
        "slp1": slp1,
        "weights": pattern,
        "n_syllables": n_syl,
        "meter": meter_name
    }

def analyze_verse_lines(lines):
    """
    Analyzes a multi-line verse (typically 2 couplets or 4 pādas).
    """
    padas = []
    for l in lines:
        cleaned = l.strip()
        if not cleaned or cleaned.startswith("##") or cleaned.startswith("---") or cleaned.startswith("<"):
            continue
        # Strip blockquote marker if present
        if cleaned.startswith(">"):
            cleaned = cleaned[1:].strip()
        # Split half-verses if marked by single danda
        sub_padas = [p.strip() for p in cleaned.split("।") if p.strip() and p.strip() != "॥"]
        if sub_padas:
            for sp in sub_padas:
                padas.append(scan_pada_meter(sp))
        else:
            padas.append(scan_pada_meter(cleaned))

    total_syllables = sum(p["n_syllables"] for p in padas)
    
    # Canonical meter analysis for Yavanajātaka (Triṣṭubh family: Upajāti / Indravajrā / Upendravajrā)
    indravajra_count = sum(1 for p in padas if p["meter"].startswith("indravajrā"))
    upendravajra_count = sum(1 for p in padas if p["meter"].startswith("upendravajrā"))
    exact_11_count = sum(1 for p in padas if p["n_syllables"] == 11)
    
    if indravajra_count > 0 and upendravajra_count > 0:
        overall_meter = "Upajāti (उपजाति, mixed Indravajrā + Upendravajrā, 11x4=44)"
    elif indravajra_count >= 2:
        overall_meter = "Indravajrā (इन्द्रवज्रा, 11x4=44)"
    elif upendravajra_count >= 2:
        overall_meter = "Upendravajrā (उपेन्द्रवज्रा, 11x4=44)"
    elif exact_11_count >= 2:
        overall_meter = "Upajāti family (11 syllables per pāda, 44 per stanza)"
    elif all(p["n_syllables"] == 8 for p in padas) and len(padas) in (2, 4):
        overall_meter = "Anuṣṭubh (अनुष्टुभ्, 8x4=32)"
    else:
        overall_meter = f"Upajāti with OCR dropout/insertion (Avg {total_syllables / max(1, len(padas)):.1f} syl/pāda vs expected 11)"

    return {
        "padas": padas,
        "total_syllables": total_syllables,
        "overall_meter": overall_meter
    }
def scan_markdown_file(md_path, max_verses=20):
    print(f"[*] Scanning meters in: {md_path}")
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.split("## पत्रम् ")
    print(f"[*] Found {len(blocks) - 1} page sections.\n")

    analyzed = 0
    for b in blocks[1:]:
        lines = b.splitlines()
        page_title = lines[0].split("(")[-1].replace(")", "").strip() if "(" in lines[0] else "Page ?"
        verse_lines = [l[2:].strip() for l in lines if l.startswith("> ") and l[2:].strip()]
        if not verse_lines:
            continue

        print("=" * 65)
        print(f"PAGE: {page_title}")
        print("=" * 65)
        
        # Analyze first couplet on page
        res = analyze_verse_lines(verse_lines[:4])
        print(f"Overall Meter Estimate: {res['overall_meter']} (Total Syllables: {res['total_syllables']})")
        for idx, p in enumerate(res["padas"], 1):
            flag = ""
            if res["overall_meter"].startswith("anuṣṭubh") and p["n_syllables"] != 8:
                flag = f" [! Deviates from 8: {p['n_syllables']}]"
            elif "11" in res["overall_meter"] and p["n_syllables"] != 11:
                flag = f" [! Deviates from 11: {p['n_syllables']}]"
            print(f"  Pāda {idx:2d}: {p['text']}")
            print(f"    Scansion: {p['weights']} ({p['n_syllables']} syl) | Meter: {p['meter']}{flag}")

        analyzed += 1
        if analyzed >= max_verses:
            break

def main():
    parser = argparse.ArgumentParser(description="Vāgdhenu Sanskrit Metrical Scanner & Chandaḥ-Parīkṣā")
    parser.add_argument("--file", default="output/markdown/PingreeYJ_1_Sanskrit.md", help="Markdown file to scan")
    parser.add_argument("--limit", type=int, default=5, help="Number of pages to scan")
    parser.add_argument("--verse", help="Single Sanskrit verse string to scan")
    args = parser.parse_args()

    if args.verse:
        lines = args.verse.splitlines()
        res = analyze_verse_lines(lines)
        print(f"Overall Meter: {res['overall_meter']}")
        for idx, p in enumerate(res["padas"], 1):
            print(f"Pāda {idx}: {p['text']}")
            print(f"  Weights: {p['weights']} ({p['n_syllables']} syl) -> {p['meter']}")
    else:
        scan_markdown_file(args.file, max_verses=args.limit)

if __name__ == "__main__":
    main()
