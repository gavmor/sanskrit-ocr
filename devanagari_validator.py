#!/home/user/Desktop/sanskrit-ocr/.venv/bin/python
"""
Devanagari Unicode Validator & Sanitizer Engine
Enforces Unicode Standard Section 12.1 and OpenType Universal Shaping Engine (USE) rules.
Directly integrated into the Sanskrit OCR pipeline at inference time.
"""

import re
import subprocess
from pathlib import Path

HB_SHAPE_BIN = "/home/linuxbrew/.linuxbrew/bin/hb-shape"
DEFAULT_FONT = str(Path.home() / ".local/share/fonts/sanskrit/fonts/Shobhika-Regular.otf")

# Independent to dependent vowel mapping when preceded by virama (e.g. क् + इ -> कि)
INDEP_TO_DEP_VOWELS = {
    '\u0905': '',        # अ -> inherent vowel
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

# Decomposed independent vowel mappings (अ + ा -> आ)
DECOMPOSED_VOWELS = {
    '\u0905\u093E': '\u0906',
    '\u0905\u093F': '\u0907',
    '\u0905\u0940': '\u0908',
    '\u0905\u0941': '\u0909',
    '\u0905\u0942': '\u090A',
    '\u0905\u0943': '\u090B',
    '\u0905\u0947': '\u090F',
    '\u0905\u0948': '\u0910',
    '\u0905\u094B': '\u0913',
    '\u0905\u094C': '\u0914',
}

# Structural regex error patterns
ILLEGAL_PATTERNS = {
    "Virama+VowelSign": re.compile(r'\u094D+[\u093E-\u094C\u0962\u0963]'),
    "VowelSign+Virama": re.compile(r'[\u093E-\u094C\u0962\u0963]\u094D+'),
    "OrphanLeadingMark": re.compile(r'(?:^|[\s\(\[\{>.,!?;:\-–—|।॥])[\u0901-\u0903\u093E-\u094D\u0962\u0963]'),
    "Virama+IndependentVowel": re.compile(r'\u094D+[\u0905-\u0914]'),
    "Anusvara+Virama": re.compile(r'[\u0901-\u0903]\u094D+'),
    "Anusvara+VowelSign": re.compile(r'[\u0901\u0902][\u093E-\u094C]'),
    "DoubleVirama": re.compile(r'\u094D{2,}'),
    "MarkOnAvagraha": re.compile(r'\u093D[\u093E-\u094D]+'),
}

def sanitize_devanagari_unicode(text):
    """
    Transforms raw OCR output into canonical, valid Unicode Devanagari.
    Eliminates all sequences that cause text shapers (Zed/HarfBuzz) to emit DottedCircle.
    """
    if not text:
        return ""

    # 1. Recompose decomposed independent vowels (अ + ा -> आ)
    for decomp, comp in DECOMPOSED_VOWELS.items():
        text = text.replace(decomp, comp)

    # 2. Strip non-Devanagari combining accents from OCR specks (e.g. U+0331 macron below)
    text = re.sub(r'[\u0300-\u036F]', '', text)

    # 3. Swap misplaced anusvara/candrabindu and dependent vowel (दंु -> दुं)
    text = re.sub(r'([\u0901\u0902])([\u093E-\u094C])', r'\2\1', text)

    # 4. Strip combining marks on avagraha (ऽ् -> ऽ)
    text = re.sub(r'\u093D[\u093E-\u094D]+', '\u093D', text)

    # 5. Strip dependent vowel marks attached to independent vowels (पाएे -> पाए)
    text = re.sub(r'([\u0904-\u0914])[\u093E-\u094C]+', r'\1', text)

    # 6. Map virama followed by independent vowel to dependent vowel (क् + इ -> कि, र् + इ -> रि)
    for indep, dep in INDEP_TO_DEP_VOWELS.items():
        text = re.sub(r'\u094D+' + indep, dep, text)

    # 7. Strip virama immediately before or after a dependent vowel sign (श् + ु -> शु)
    text = re.sub(r'\u094D+([\u093E-\u094C\u0962\u0963])', r'\1', text)
    text = re.sub(r'([\u093E-\u094C\u0962\u0963])\u094D+', r'\1', text)

    # 8. Collapse multiple consecutive viramas (्् -> ्)
    text = re.sub(r'\u094D{2,}', '\u094D', text)

    # 9. Collapse multiple identical matras
    text = re.sub(r'([\u093E-\u094C])\1+', r'\1', text)

    # 10. Strip viramas attached to anusvara/visarga (ं् -> ं, ः् -> ः, क् + ं -> कं)
    text = re.sub(r'([\u0901-\u0903])\u094D+', r'\1', text)
    text = re.sub(r'\u094D+([\u0901-\u0903])', r'\1', text)

    # 11. Strip Vedic pitch accents that OCR hallucinates from printed specks
    text = re.sub(r'[\u0951-\u0954]', '', text)

    # 12. Strip orphan leading combining marks at word boundary or after punctuation
    # (Removes orphan viramas like '> ्व' -> '> व' and ' ्या' -> ' या')
    text = re.sub(r'(^|[\s\(\[\{>.,!?;:\-–—|।॥])[\u0901-\u0903\u093E-\u094D\u0962\u0963]+', r'\1', text)
    text = re.sub(r'(^|[\s\(\[\{>.,!?;:\-–—|।॥])[\u0901-\u0903\u093E-\u094D\u0962\u0963]+', r'\1', text)

    # 13. Strip isolated single consonant with virama at end of word or line (e.g. '। र्' -> '।')
    text = re.sub(r'(\s+)[\u0915-\u0939]\u094D(?=[\s|।॥]|$)', r'', text)

    # 14. Normalize danda punctuation
    text = re.sub(r'[।॥]{2,}', '॥', text)
    text = re.sub(r'\s+([।॥])', r'\1', text)
    text = re.sub(r'([।॥])(?=[^\s\n|।॥])', r'\1 ', text)
    text = re.sub(r'^[।॥]\s*', '', text)

    # 15. Normalize spaces
    text = re.sub(r'[ \t]+', ' ', text).strip()

    return text


def validate_devanagari_line(text, font_path=DEFAULT_FONT, check_harfbuzz=False):
    """
    Validates a text line. Returns:
      - is_valid (bool): True if line has no structural violations and no HarfBuzz DottedCircle.
      - issues (list[str]): List of detected violation names.
      - has_dotted_circle (bool): True if HarfBuzz rendered a DottedCircle glyph.
    """
    issues = []
    for name, pat in ILLEGAL_PATTERNS.items():
        if pat.search(text):
            issues.append(name)

    has_dotted = False
    if check_harfbuzz and Path(HB_SHAPE_BIN).exists() and Path(font_path).exists():
        clean_sample = text.replace('>', '').strip()
        if clean_sample:
            cmd = [HB_SHAPE_BIN, font_path, f"--text={clean_sample}"]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out = res.stdout
            if ("DottedCircle" in out) or ("uni25CC" in out) or ("gid" in out and "25CC" in out):
                has_dotted = True
                issues.append("HarfBuzz-DottedCircle")

    is_valid = len(issues) == 0 and not has_dotted
    return is_valid, issues, has_dotted
