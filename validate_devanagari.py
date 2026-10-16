#!/home/user/Desktop/sanskrit-ocr/.venv/bin/python
"""
Unicode Devanagari Sequence Validator & Sanitizer for Sanskrit OCR
Validates against illegal Unicode clusters and HarfBuzz DottedCircle glyph errors.
"""

import sys
import re
import argparse
import subprocess
from pathlib import Path

HB_SHAPE_BIN = "/home/linuxbrew/.linuxbrew/bin/hb-shape"
DEFAULT_FONT = str(Path.home() / ".local/share/fonts/sanskrit/fonts/Shobhika-Regular.otf")

# Map independent vowels to dependent vowel signs (matras) when preceded by a virama (e.g. क् + इ -> कि)
INDEP_TO_DEP_VOWELS = {
    '\u0905': '',        # अ -> inherently inherent
    '\u0906': '\u093E',  # आ -> ा
    '\u0907': '\u093F',  # इ -> ि
    '\u0908': '\u0940',  # ई -> ी
    '\u0909': '\u0941',  # उ -> ु
    '\u090A': '\u0942',  # ऊ -> ू
    '\u090B': '\u0943',  # ऋ -> ृ
    '\u090C': '\u0944',  # ऌ -> ॢ
    '\u090F': '\u0947',  # ए -> े
    '\u0910': '\u0948',  # ऐ -> ै
    '\u0913': '\u094B',  # ओ -> ो
    '\u0914': '\u094C',  # औ -> ौ
}

from devanagari_validator import sanitize_devanagari_unicode as sanitize_devanagari_line


def check_harfbuzz_dotted_circles(text, font_path=DEFAULT_FONT):
    """
    Invokes HarfBuzz to shape text. Returns True if any DottedCircle glyph is generated.
    """
    if not Path(HB_SHAPE_BIN).exists() or not Path(font_path).exists():
        return False, []
    
    clean_sample = text.replace('>', '').strip()
    if not clean_sample:
        return False, []
        
    cmd = [HB_SHAPE_BIN, font_path, f"--text={clean_sample}"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = res.stdout
    has_dotted = ("DottedCircle" in out) or ("uni25CC" in out) or ("gid" in out and "25CC" in out)
    return has_dotted, out


def audit_file(file_path, font_path=DEFAULT_FONT, auto_fix=False):
    file_path = Path(file_path).resolve()
    print(f"[*] Auditing file: {file_path.name}")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    issues_found = 0
    hb_dotted_count = 0
    fixed_lines = []

    for l_idx, line in enumerate(lines, 1):
        raw = line.rstrip("\n")
        
        # Structural regex checks
        m_virama_matra = bool(re.search(r'\u094D[\u093E-\u094C\u0962\u0963]', raw))
        m_orphan_lead = bool(re.search(r'(?:^|[\s\(\[\{>.,!?;:\-–—|।॥])[\u0901-\u0903\u093E-\u094D\u0962\u0963]', raw))
        m_virama_indep = bool(re.search(r'\u094D[\u0905-\u0914]', raw))
        m_anusvara_virama = bool(re.search(r'[\u0901-\u0903]\u094D', raw))
        m_double_virama = bool(re.search(r'\u094D{2,}', raw))
        
        has_regex_issue = (m_virama_matra or m_orphan_lead or m_virama_indep or 
                           m_anusvara_virama or m_double_virama)

        # HarfBuzz OpenType shape check
        has_hb_dotted, glyphs = check_harfbuzz_dotted_circles(raw, font_path)
        if has_hb_dotted:
            hb_dotted_count += 1

        if has_regex_issue or has_hb_dotted:
            issues_found += 1
            if not auto_fix and issues_found <= 10:
                print(f"  [Line {l_idx:4d}] Violation detected:")
                print(f"    Text: {raw[:70]}")
                reasons = []
                if m_virama_matra: reasons.append("Virama+VowelSign")
                if m_orphan_lead: reasons.append("OrphanLeadingMark")
                if m_virama_indep: reasons.append("Virama+IndependentVowel")
                if m_anusvara_virama: reasons.append("Anusvara+Virama")
                if m_double_virama: reasons.append("DoubleVirama")
                if has_hb_dotted: reasons.append("HarfBuzz-DottedCircle")
                print(f"    Tags: {', '.join(reasons)}")

        if auto_fix:
            if raw.startswith('> '):
                prefix = '> '
                content = raw[2:]
            else:
                prefix = ''
                content = raw
            sanitized = prefix + sanitize_devanagari_line(content)
            fixed_lines.append(sanitized + "\n")

    print("\n" + "=" * 65)
    print("UNICODE AUDIT SUMMARY:")
    print(f"  Total Lines Analyzed:         {len(lines)}")
    print(f"  Lines with Structural Issues: {issues_found}")
    print(f"  Lines triggering DottedCircle:{hb_dotted_count}")
    print("=" * 65)

    if auto_fix:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(fixed_lines)
        print(f"[+] File automatically sanitized and re-written: {file_path}")
        
        # Verify post-fix
        print("[*] Re-auditing sanitized file...")
        audit_file(file_path, font_path, auto_fix=False)

    return issues_found == 0 and hb_dotted_count == 0


def main():
    parser = argparse.ArgumentParser(description="Devanagari Unicode & HarfBuzz Sequence Validator")
    parser.add_argument("file", help="Path to markdown or text file")
    parser.add_argument("--font", default=DEFAULT_FONT, help="Path to OpenType Devanagari font for HarfBuzz")
    parser.add_argument("--fix", action="store_true", help="Automatically sanitize and repair invalid clusters")
    args = parser.parse_args()

    success = audit_file(args.file, args.font, auto_fix=args.fix)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
