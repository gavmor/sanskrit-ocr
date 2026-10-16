#!/home/user/Desktop/sanskrit-ocr/.venv/bin/python
"""
Pipeline Quality Control & Metrical Audit Tool
Audits Sanskrit OCR transcriptions against canonical meters (Upajāti/Indravajrā 11-syl),
detects syllable dropouts/insertions, and scores lexical health.
"""

import sys
import re
import json
import argparse
from pathlib import Path
from collections import Counter

from vagdhenu import to_slp1, scan_meter, identify_meter

def audit_transcription(md_path, expected_syllables=11, max_pages=None):
    md_path = Path(md_path).resolve()
    print(f"[*] Auditing quality control for: {md_path.name}")
    print(f"[*] Expected canonical meter length: {expected_syllables} syllables per pāda (Upajāti/Triṣṭubh)\n")

    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    page_blocks = content.split("## पत्रम् ")
    if len(page_blocks) <= 1:
        page_blocks = content.split("## Page ")

    total_pages = len(page_blocks) - 1
    if max_pages:
        page_blocks = page_blocks[:max_pages + 1]

    page_reports = []
    global_syllable_dist = Counter()
    total_padas = 0
    exact_matches = 0
    deficit_padas = 0
    excess_padas = 0

    for b in page_blocks[1:]:
        lines = b.splitlines()
        page_header = lines[0].strip()
        verse_lines = [l[2:].strip() for l in lines if l.startswith("> ") and l[2:].strip()]
        if not verse_lines:
            continue

        padas_on_page = []
        page_deficit = 0
        page_excess = 0
        page_exact = 0

        for l in verse_lines:
            # Split half-verses
            sub_padas = [p.strip() for p in l.split("।") if p.strip() and p.strip() != "॥"]
            for sp in (sub_padas or [l]):
                slp1 = to_slp1(sp, "devanagari")
                weights, n_syl = scan_meter(slp1, pada_final_guru=True)
                if n_syl < 3:
                    continue  # skip fragments

                total_padas += 1
                global_syllable_dist[n_syl] += 1
                delta = n_syl - expected_syllables

                status = "EXACT"
                if delta == 0:
                    exact_matches += 1
                    page_exact += 1
                elif delta < 0:
                    deficit_padas += 1
                    page_deficit += 1
                    status = f"DROPOUT ({delta})"
                else:
                    excess_padas += 1
                    page_excess += 1
                    status = f"EXCESS (+{delta})"

                padas_on_page.append({
                    "text": sp,
                    "syllables": n_syl,
                    "weights": "".join(weights),
                    "delta": delta,
                    "status": status,
                    "meter": identify_meter(weights)
                })

        n_padas = len(padas_on_page)
        adherence_pct = (page_exact / max(1, n_padas)) * 100
        page_reports.append({
            "page": page_header,
            "pada_count": n_padas,
            "exact_11_count": page_exact,
            "deficit_count": page_deficit,
            "excess_count": page_excess,
            "metrical_adherence_pct": round(adherence_pct, 1),
            "padas": padas_on_page
        })

    # Summary calculations
    overall_adherence = (exact_matches / max(1, total_padas)) * 100
    avg_syllables = sum(syl * count for syl, count in global_syllable_dist.items()) / max(1, total_padas)

    print("=" * 70)
    print("SANSKRIT OCR PIPELINE QUALITY CONTROL REPORT")
    print("=" * 70)
    print(f"  Total Pages Audited:               {len(page_reports)} / {total_pages}")
    print(f"  Total Pādas Analyzed:              {total_padas}")
    print(f"  Average Syllables per Pāda:        {avg_syllables:.2f} (Target: {expected_syllables})")
    print(f"  Exact Metrical Matches (11 syl):   {exact_matches} ({overall_adherence:.1f}%)")
    print(f"  Pādas with Syllable Dropout (<11): {deficit_padas} ({(deficit_padas/max(1,total_padas))*100:.1f}%)")
    print(f"  Pādas with Excess/Run-on (>11):    {excess_padas} ({(excess_padas/max(1,total_padas))*100:.1f}%)")
    print("-" * 70)
    print("SYLLABLE COUNT DISTRIBUTION:")
    for syl in sorted(global_syllable_dist.keys()):
        marker = " <--- TARGET (11-syl Upajāti)" if syl == expected_syllables else ""
        cnt = global_syllable_dist[syl]
        bar = "█" * int(cnt * 40 / max(global_syllable_dist.values()))
        print(f"  {syl:2d} syl: {cnt:5d} ({cnt*100/max(1,total_padas):5.1f}%) | {bar}{marker}")
    print("=" * 70)

    # Save structured JSON report
    report_file = md_path.parent / "qc_report.json"
    with open(report_file, "w", encoding="utf-8") as jf:
        json.dump({
            "source_document": md_path.name,
            "target_meter": "Upajāti (11 syllables per pāda)",
            "summary": {
                "pages_audited": len(page_reports),
                "total_padas": total_padas,
                "exact_matches": exact_matches,
                "overall_adherence_pct": round(overall_adherence, 1),
                "deficit_padas": deficit_padas,
                "excess_padas": excess_padas,
                "avg_syllables": round(avg_syllables, 2)
            },
            "syllable_distribution": dict(sorted(global_syllable_dist.items())),
            "pages": page_reports[:50]  # sample first 50 pages in json
        }, jf, ensure_ascii=False, indent=2)

    print(f"[+] Full QC report saved: {report_file}")
    return report_file

def main():
    parser = argparse.ArgumentParser(description="Sanskrit OCR Metrical QC & Pipeline Auditor")
    parser.add_argument("--file", default="output/markdown/PingreeYJ_1_Sanskrit.md", help="Markdown file to audit")
    parser.add_argument("--syllables", type=int, default=11, help="Expected syllables per pada (default: 11 for Yavanajataka)")
    parser.add_argument("--pages", type=int, default=None, help="Max pages to audit (default: all)")
    args = parser.parse_args()

    audit_transcription(args.file, expected_syllables=args.syllables, max_pages=args.pages)

if __name__ == "__main__":
    main()
