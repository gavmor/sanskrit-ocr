#!/home/user/Desktop/sanskrit-ocr/.venv/bin/python
"""
Clean Sanskrit-Only Markdown Generator
Strips all English sections (Pages 1-56 and critical apparatus)
Formats Pages 57-518 into clean, readable Sanskrit verse couplets.
"""

import re
from pathlib import Path

PAGES_DIR = Path("output/markdown/pages")
OUTPUT_MD = Path("output/markdown/PingreeYJ_1_Sanskrit.md")

DEV_NUMS = str.maketrans("0123456789", "०१२३४५६७८९")
from devanagari_validator import sanitize_devanagari_unicode


def clean_line(l):
    # Normalize whitespace
    l = re.sub(r'[ \t]+', ' ', l).strip()
    
    # Strip leading/trailing stray non-devanagari punctuation
    l = re.sub(r'^[#@$%&*~_+=<>{}\[\]\d\s.,\-/]+', '', l).strip()
    l = re.sub(r'[#@$%&*~_+=<>{}\[\]\s.,\-/]+$', '', l).strip()
    
    # Sanitize invalid Unicode sequences that cause dotted circle glyph errors in strict shapers (Zed/HarfBuzz)
    l = sanitize_devanagari_unicode(l)
    
    # Require at least 4 devanagari characters (filters out noise like '।', '४', '० ।', '{ओओऒ#')
    dev_chars = sum(1 for c in l if '\u0900' <= c <= '\u097F' and c not in '।॥०१२३४५६७८९')
    if dev_chars < 4:
        return None
        
    # Check that the line is predominantly Devanagari
    total_alpha = sum(1 for c in l if not c.isspace() and c not in '0123456789.,-—|/:;#')
    if total_alpha > 0 and (dev_chars / total_alpha) < 0.6:
        return None
        
    return l
def normalize_dandas(line):
    # Collapse multiple consecutive dandas like ॥। -> ॥
    line = re.sub(r'[।॥]{2,}', '॥', line)
    # Attach danda to preceding word (no space before)
    line = re.sub(r'\s+([।॥])', r'\1', line)
    # Ensure single space after danda if followed by characters
    line = re.sub(r'([।॥])(?=[^\s\n|।॥])', r'\1 ', line)
    # Strip leading stray danda at the very start of a line
    line = re.sub(r'^[।॥]\s*', '', line)
    return line.strip()

def format_couplets(lines):
    couplets = []
    i = 0
    n = len(lines)
    while i < n:
        l1 = normalize_dandas(lines[i])
        # Check if line 1 already ends with verse marker ॥
        if '॥' in l1 or i + 1 >= n:
            couplets.append(f"> {l1}")
            i += 1
            continue
            
        l2 = normalize_dandas(lines[i+1])
        couplets.append(f"> {l1}  \n> {l2}")
        i += 2
        
    return "\n\n".join(couplets)

def generate_clean_sanskrit_markdown():
    print(f"[*] Compiling clean Sanskrit-only Markdown from {PAGES_DIR}...")
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    
    header = (
        "# यवनाजातकम्\n"
        "### Sphujidhvaja's Yavanajātaka (Sanskrit Text)\n"
        "*Edition: David Pingree, Harvard Oriental Series, Volume 48 (1978)*  \n"
        "*Source: PingreeYJ_1.pdf (Pages 57–518)*  \n\n"
        "---\n\n"
    )
    
    sections = [header]
    total_verses = 0
    pages_included = 0
    
    for page_num in range(57, 519):
        p_file = PAGES_DIR / f"page_{page_num:04d}.md"
        if not p_file.exists():
            continue
            
        with open(p_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Extract only Sanskrit Verses section
        if "### Sanskrit Verses" not in content:
            continue
            
        s_part = content.split("### Sanskrit Verses")[1]
        # Nix English apparatus completely
        if "<details>" in s_part:
            s_part = s_part.split("<details>")[0]
            
        raw_lines = [l.strip() for l in s_part.splitlines() if l.strip()]
        clean_lines = []
        for l in raw_lines:
            cl = clean_line(l)
            if cl:
                clean_lines.append(cl)
                
        if not clean_lines:
            continue
            
        formatted_verses = format_couplets(clean_lines)
        dev_page = str(page_num).translate(DEV_NUMS)
        
        page_block = (
            f"## पत्रम् {dev_page} (Page {page_num})\n\n"
            f"{formatted_verses}\n\n"
            "---\n\n"
        )
        sections.append(page_block)
        total_verses += len(clean_lines)
        pages_included += 1
        
    final_text = "".join(sections)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(final_text)
        
    print(f"[+] Clean Sanskrit Markdown generated: {OUTPUT_MD}")
    print(f"[+] Included: {pages_included} pages, ~{total_verses} Sanskrit verse lines.")
    print(f"[+] File size: {OUTPUT_MD.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    generate_clean_sanskrit_markdown()
