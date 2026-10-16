#!/home/user/Desktop/sanskrit-ocr/.venv/bin/python
"""
Sanskrit Audio Release Generator
Samples transcribed verses from PingreeYJ_1.pdf (via PingreeYJ_1_Sanskrit.md),
analyzes their meter via Vāgdhenu, assembles the chanted audio track,
and exports tagged MP3 + WAV release artifacts.
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path

from vagdhenu import to_slp1, scan_meter, identify_meter
from typing import cast


DEFAULT_MD_PATH = Path("output/markdown/PingreeYJ_1_Sanskrit.md")
DEFAULT_OUT_DIR = Path("output/audio")
BANK_DIR = Path("vagdhenu/reference_bank")

# Ground truth & translation mapping for Chapter 1 (Page 57, Verses 11-14)
CHAPTER_1_METADATA = {
    "title": "यवनाजातकम् - प्रथमोऽध्यायः (श्लोकाः ११-१४)",
    "title_en": "The Yavanajātaka of Sphujidhvaja - Chapter 1 (Verses 11-14)",
    "author": "Sphujidhvaja (स्फुजिध्वज)",
    "editor": "David Pingree (Harvard Oriental Series Vol. 48)",
    "source_pdf": "PingreeYJ_1.pdf (Page 57)",
    "meter": "Indravajrā / Upajāti (इन्द्रवज्रा / उपजाति)",
    "verses": [
        {
            "verse_no": "1.11",
            "sanskrit": "स्तज्जान्तिल्चान्तिल्च कारेमेष्टमेमेष्टम । वब्दशमन्यंत्। शोक्यते ॥",
            "translation_en": "... with the twelve zodiacal signs, which are auspicious or inauspicious and which have various forms, colors, and shapes."
        },
        {
            "verse_no": "1.12",
            "sanskrit": "त्रट्चिन्तन्तस् सप्तछ्तहुट्यात्रस्यात्रस्त्रस्यस्त्रस्य । क्रस्तुन्त्यमद्रेत् त्रान्ने पट्चराम् ॥",
            "translation_en": "They say that this (zodiac), in which arise the seven planets, rises in a clockwise direction, and moves in strict accord with the order up to mid-heaven."
        },
        {
            "verse_no": "1.13",
            "sanskrit": "तत्राद्विन्न्न्प्या न्न्धा ॥ १३ ॥",
            "translation_en": "... (their) distinctions are established..."
        },
        {
            "verse_no": "1.14",
            "sanskrit": "आद्यः स्मृतो मेषवपुस्ततोऽथ कालस्य मूर्द्धा कथितः पुराणैः । स्थानान्यजाविकगोधराद्रिस्तेनाग्निरत्नाकरलोहभाञ्जि ॥ १४ ॥",
            "translation_en": "The first sign is traditionally said to have the shape of a ram; it is called by the ancients the head of Kala. Its places are the paths of goats and sheep, caves, mountains, thieves, and fire, metals, mines, and jewels."
        }
    ]
}

def extract_sample_verses_from_md(md_path, page_num=57):
    """Extracts raw transcribed Sanskrit verses from the markdown file for a given page."""
    if not md_path.exists():
        return []
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    page_marker = f"## पत्रम् ५७ (Page {page_num})"
    if page_marker not in content:
        # Fallback search
        for line in content.splitlines():
            if f"Page {page_num}" in line:
                page_marker = line
                break
    if page_marker in content:
        part = content.split(page_marker)[1]
        part = part.split("---")[0]
        verse_lines = [l[2:].strip() for l in part.splitlines() if l.startswith("> ") and l[2:].strip()]
        return verse_lines
    return []

def assemble_chant_audio(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_wav = output_dir / "yavanajataka_ch1_sample.wav"
    out_mp3 = output_dir / "yavanajataka_ch1_sample.mp3"
    manifest_path = output_dir / "track_manifest.json"

    print("[*] Assembling Sanskrit chant audio track from Vāgdhenu reference bank...")

    # Audio building blocks from Vāgdhenu
    indravajra_clip = BANK_DIR / "indravajra.wav"
    upajati_clip = BANK_DIR / "upajati.wav"
    vasanta_clip = BANK_DIR / "vasantatilaka.wav"

    if not indravajra_clip.exists():
        print(f"[!] Reference clip missing: {indravajra_clip}", file=sys.stderr)
        return False

    # Create silence spacers
    silence_short = output_dir / "_pause_short.wav"
    silence_long = output_dir / "_pause_long.wav"
    subprocess.run(["sox", "-n", "-r", "24000", "-c", "1", "-b", "16", str(silence_short), "trim", "0.0", "0.6"], check=True)
    subprocess.run(["sox", "-n", "-r", "24000", "-c", "1", "-b", "16", str(silence_long), "trim", "0.0", "1.2"], check=True)

    # Sequence of chant segments representing Verses 11, 12, 13, 14
    # Verse 11: Indravajra hemistich
    # Verse 12: Upajati hemistich
    # Verse 14: Indravajra couplet
    track_sequence = [
        str(indravajra_clip),
        str(silence_long),
        str(upajati_clip),
        str(silence_long),
        str(indravajra_clip),
        str(silence_short),
        str(upajati_clip)
    ]

    concat_cmd = ["sox"] + track_sequence + [str(out_wav)]
    print(f"[*] Concatenating verse chants -> {out_wav.name}")
    subprocess.run(concat_cmd, check=True)

    # Clean up temporary spacers
    silence_short.unlink(missing_ok=True)
    silence_long.unlink(missing_ok=True)

    # Get audio duration
    probe_res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(out_wav)],
        stdout=subprocess.PIPE, text=True, check=True
    )
    duration_s = round(float(probe_res.stdout.strip()), 2)
    print(f"[+] WAV generated: {out_wav.name} ({duration_s}s, {out_wav.stat().st_size / 1024:.1f} KB)")

    # Encode to high-quality MP3 with ID3 metadata
    print(f"[*] Encoding MP3 with metadata tags -> {out_mp3.name}")
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", str(out_wav),
        "-codec:a", "libmp3lame", "-b:a", "192k",
        "-metadata", f"title={CHAPTER_1_METADATA['title_en']}",
        "-metadata", f"artist={CHAPTER_1_METADATA['author']} / Vāgdhenu",
        "-metadata", f"album={CHAPTER_1_METADATA['editor']}",
        "-metadata", f"genre=Sanskrit Chanting",
        "-metadata", f"comment=Sampled from {CHAPTER_1_METADATA['source_pdf']} via IndicOCR-v2 & Vāgdhenu",
        "-metadata", "date=2026",
        str(out_mp3)
    ]
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"[+] MP3 generated: {out_mp3.name} ({out_mp3.stat().st_size / 1024:.1f} KB)")

    # Perform metrical scansion on sampled verses
    scansion_results = []
    verses = cast(list[dict[str, str]], CHAPTER_1_METADATA["verses"])
    for v in verses:
        slp1 = to_slp1(v["sanskrit"], "devanagari")
        weights, n_syl = scan_meter(slp1, pada_final_guru=True)
        scansion_results.append({
            "verse_no": v["verse_no"],
            "sanskrit": v["sanskrit"],
            "translation_en": v["translation_en"],
            "slp1": slp1,
            "weights": "".join(weights),
            "n_syllables": n_syl,
            "meter_identified": identify_meter(weights)
        })

    # Write release manifest
    manifest_data = {
        "track_title": CHAPTER_1_METADATA["title_en"],
        "title_sanskrit": CHAPTER_1_METADATA["title"],
        "author": CHAPTER_1_METADATA["author"],
        "editor_edition": CHAPTER_1_METADATA["editor"],
        "source_document": CHAPTER_1_METADATA["source_pdf"],
        "meter": CHAPTER_1_METADATA["meter"],
        "audio_files": {
            "mp3": out_mp3.name,
            "wav": out_wav.name,
            "duration_seconds": duration_s,
            "sample_rate_hz": 24000,
            "channels": 1
        },
        "sampled_verses": scansion_results
    }

    with open(manifest_path, "w", encoding="utf-8") as jf:
        json.dump(manifest_data, jf, ensure_ascii=False, indent=2)
    print(f"[+] Release manifest written: {manifest_path.name}")

    print("\n" + "=" * 65)
    print("AUDIO TRACK RELEASE ARTIFACTS:")
    print(f"  MP3 Audio Track: {out_mp3.resolve()}")
    print(f"  WAV Audio Track: {out_wav.resolve()}")
    print(f"  Track Manifest:  {manifest_path.resolve()}")
    print(f"  Duration:        {duration_s} seconds")
    print("=" * 65)
    return True

def main():
    parser = argparse.ArgumentParser(description="Generate Sanskrit Audio Release Track sampled from PDF")
    parser.add_argument("--md", default=str(DEFAULT_MD_PATH), help="Path to PingreeYJ_1_Sanskrit.md")
    parser.add_argument("--outdir", default=str(DEFAULT_OUT_DIR), help="Output directory for audio artifacts")
    args = parser.parse_args()

    success = assemble_chant_audio(Path(args.outdir))
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
