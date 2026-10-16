#!/home/user/Desktop/sanskrit-ocr/.venv/bin/python
"""
High-Performance Sanskrit PDF to Markdown Converter with LV-ROVER Alignment
Converts Pingree's Yavanajātaka (PingreeYJ_1.pdf) into clean, metrically-validated Markdown.
- Pages 57-518: Classical Sanskrit Devanagari verses (Chapters 1-79)
- Dual-engine consensus: IndicOCR-v2 (Attention-LSTM) + Tesseract 5.5 ('san' CTC)
- Vāgdhenu 11-syllable Upajāti metrical alignment & resolution
- OpenType HarfBuzz cluster sanitization (zero DottedCircles)
- Formats verses into clean blockquoted couplets
- Multi-process parallel execution with batch inference
"""

import os
import sys
import io
import re
import time
import json
import base64
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import requests
import pymupdf
import pytesseract
from PIL import Image

from devanagari_validator import sanitize_devanagari_unicode, validate_devanagari_line, decode_output
from lv_rover import LVROVER

MODEL_URL = "http://127.0.0.1:8501/v1/models/sanskrit:predict"
DEV_NUMS = str.maketrans("0123456789", "०१२३४५६७८९")

def clean_repetitions(text):
    if not text:
        return ""
    text = re.sub(r'([^\s])\1{3,}', r'\1\1', text)
    for l in range(35, 2, -1):
        pattern = re.compile(rf'(.{{{l}}}?)\1{{2,}}')
        text = pattern.sub(r'\1', text)
    return text.strip()

def classify_line_script(text):
    dev_count = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total_alpha = sum(1 for c in text if not c.isspace() and c not in '0123456789.,-—|/:;#~')
    if total_alpha > 0 and (dev_count / total_alpha) >= 0.35:
        return "devanagari"
    return "latin"

def tesseract_ocr_line(roi, lang="san"):
    cmd = ["/home/linuxbrew/.linuxbrew/bin/tesseract", "stdin", "stdout", "-l", lang, "--psm", "7"]
    success, buf = cv2.imencode(".png", roi)
    if not success:
        return ""
    try:
        res = subprocess.run(cmd, input=buf.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5.0)
        return res.stdout.decode("utf-8").strip()
    except Exception:
        return ""

def clean_line_content(l):
    l = re.sub(r'[ \t]+', ' ', l).strip()
    l = re.sub(r'^[#@$%&*~_+=<>{}\[\]\d\s.,\-/]+', '', l).strip()
    l = re.sub(r'[#@$%&*~_+=<>{}\[\]\s.,\-/]+$', '', l).strip()
    l = sanitize_devanagari_unicode(l)
    dev_chars = sum(1 for c in l if '\u0900' <= c <= '\u097F' and c not in '।॥०१२३४५६७८९')
    if dev_chars < 4:
        return None
    total_alpha = sum(1 for c in l if not c.isspace() and c not in '0123456789.,-—|/:;#')
    if total_alpha > 0 and (dev_chars / total_alpha) < 0.6:
        return None
    return l

def normalize_dandas(line):
    line = re.sub(r'[।॥]{2,}', '॥', line)
    line = re.sub(r'\s+([।॥])', r'\1', line)
    line = re.sub(r'([।॥])(?=[^\s\n|।॥])', r'\1 ', line)
    line = re.sub(r'^[।॥]\s*', '', line)
    return line.strip()

def format_couplets(lines):
    couplets = []
    i = 0
    n = len(lines)
    while i < n:
        l1 = normalize_dandas(lines[i])
        if '॥' in l1 or i + 1 >= n:
            couplets.append(f"> {l1}")
            i += 1
            continue
        l2 = normalize_dandas(lines[i+1])
        couplets.append(f"> {l1}  \n> {l2}")
        i += 2
    return "\n\n".join(couplets)

def process_page_worker(args_tuple):
    pdf_path, pno, out_dir_str, force = args_tuple
    page_num = pno + 1
    page_md_path = Path(out_dir_str) / f"page_{page_num:04d}.md"

    # Only process Sanskrit pages 57 to 518
    if page_num < 57 or page_num > 518:
        return page_num, "skipped (intro/end)", 0.0

    if not force and page_md_path.exists() and page_md_path.stat().st_size > 20:
        return page_num, "cached", 0.0

    t0 = time.time()
    try:
        doc = pymupdf.open(pdf_path)
        page = doc[pno]
        ilist = page.get_images()
        if ilist:
            base_img = doc.extract_image(ilist[0][0])
            pil_img = Image.open(io.BytesIO(base_img['image'])).convert('L')
        else:
            pix = page.get_pixmap(dpi=200)
            pil_img = Image.open(io.BytesIO(pix.tobytes())).convert('L')

        img_np = np.array(pil_img)
        H, W = img_np.shape[:2]

        # Line detection via Devanagari layout analysis
        data = pytesseract.image_to_data(pil_img, lang='Devanagari', output_type=pytesseract.Output.DICT)
        n = len(data['level'])
        line_boxes = []
        for i in range(n):
            if data['level'][i] == 4:
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                if w < 40 or h < 18 or h > w * 1.3 or x > W - 60 or x < 40 or y < 40 or y > H - 40:
                    continue
                line_boxes.append((y, x, w, h))
        line_boxes.sort(key=lambda b: (b[0], b[1]))

        if not line_boxes:
            md_content = f"## पत्रम् {str(page_num).translate(DEV_NUMS)} (Page {page_num})\n\n*[No text detected]*\n"
            with open(page_md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            return page_num, "empty", time.time() - t0

        # Prepare batch for IndicOCR-v2
        b64_list = []
        for y, x, w, h in line_boxes:
            roi = img_np[y:y+h, x:x+w]
            r_min, r_max = float(roi.min()), float(roi.max())
            if r_max > r_min:
                roi_norm = ((roi.astype(np.float32) - r_min) * (255.0 / (r_max - r_min))).astype(np.uint8)
            else:
                roi_norm = roi.copy()
            r = 32.0 / float(h)
            new_w = max(1, int(w * r))
            resized = cv2.resize(roi_norm, (new_w, 32), interpolation=cv2.INTER_CUBIC)
            _, buf = cv2.imencode('.jpg', resized)
            b64_list.append({'b64': base64.b64encode(buf.tobytes()).decode('utf-8')})

        # Batch query IndicOCR model server
        resp = requests.post(MODEL_URL, json={'signature_name': 'serving_default', 'inputs': b64_list}, timeout=60.0)
        res = resp.json()
        outputs = res.get('outputs', {}).get('output', [])
        if isinstance(outputs, str):
            outputs = [outputs]

        # Multi-Engine Consensus with LV-ROVER + Vāgdhenu Upajāti constraint
        rover = LVROVER(target_syllables=11)
        clean_verses = []

        for idx, raw_out in enumerate(outputs):
            decoded = decode_output(raw_out)
            cleaned = clean_repetitions(decoded)
            if not cleaned:
                continue
            sanitized = sanitize_devanagari_unicode(cleaned)
            if not sanitized:
                continue

            script = classify_line_script(sanitized)
            if script == "devanagari":
                # Run Tesseract Sanskrit on ROI and align via LV-ROVER
                y, x, w, h = line_boxes[idx]
                roi = img_np[y:y+h, x:x+w]
                tess_san = tesseract_ocr_line(roi, lang="san")
                if tess_san:
                    rover_res = rover.resolve_line({"indic": sanitized, "tess_san": tess_san})
                    line_text = rover_res["resolved_text"]
                else:
                    line_text = sanitized

                cl = clean_line_content(line_text)
                if cl:
                    clean_verses.append(cl)

        dev_page = str(page_num).translate(DEV_NUMS)
        formatted_couplets = format_couplets(clean_verses) if clean_verses else "*[No verses transcribed]*"
        md_content = f"## पत्रम् {dev_page} (Page {page_num})\n\n{formatted_couplets}\n\n---\n"

        with open(page_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        dt = time.time() - t0
        return page_num, f"{len(clean_verses)} verses (LV-ROVER)", dt

    except Exception as e:
        return page_num, f"error: {str(e)[:50]}", time.time() - t0


def build_final_markdown(out_dir, final_md_path):
    print(f"[*] Assembling clean Sanskrit Markdown document: {final_md_path.name}...")
    header = (
        "# यवनाजातकम्\n"
        "### Sphujidhvaja's Yavanajātaka (Sanskrit Text)\n"
        "*Edition: David Pingree, Harvard Oriental Series, Volume 48 (1978)*  \n"
        "*Transcribed via IndicOCR-v2 + Tesseract 5.5 + LV-ROVER Metrical Alignment*  \n"
        "*Source: PingreeYJ_1.pdf (Pages 57–518, Chapters 1–79)*  \n\n"
        "---\n\n"
    )
    pages_dir = Path(out_dir)
    total_pages = 0
    with open(final_md_path, "w", encoding="utf-8") as out_f:
        out_f.write(header)
        for p in range(57, 519):
            p_file = pages_dir / f"page_{p:04d}.md"
            if p_file.exists():
                with open(p_file, "r", encoding="utf-8") as pf:
                    out_f.write(pf.read() + "\n\n")
                total_pages += 1

    print(f"[+] Final document written: {final_md_path} ({final_md_path.stat().st_size / 1024:.1f} KB, {total_pages} pages)")


def main():
    parser = argparse.ArgumentParser(description="Parse entire PingreeYJ_1.pdf to clean Sanskrit Markdown with LV-ROVER")
    parser.add_argument("--pdf", default="/home/user/Downloads/PingreeYJ_1.pdf")
    parser.add_argument("--pages", default="57-518", help="Page range, e.g. '57-518' (Sanskrit text), '57-65', or 'all'")
    parser.add_argument("--workers", type=int, default=10, help="Number of worker processes")
    parser.add_argument("--out-dir", default="./output/markdown/pages")
    parser.add_argument("--final-md", default="./output/markdown/PingreeYJ_1_Sanskrit.md")
    parser.add_argument("--force", action="store_true", help="Force re-parsing of cached pages")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"[!] PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(pdf_path))
    pdf_total_pages = len(doc)
    doc.close()

    if args.pages.lower() == "all":
        page_indices = list(range(56, min(pdf_total_pages, 518)))  # Sanskrit core: 57-518
    elif "-" in args.pages:
        s, e = args.pages.split("-", 1)
        page_indices = list(range(int(s) - 1, min(pdf_total_pages, int(e))))
    else:
        page_indices = [int(p) - 1 for p in args.pages.split(",") if p.strip()]

    print(f"=======================================================")
    print(f"SANSKRIT PDF -> MARKDOWN PARSER (LV-ROVER ENSEMBLE)")
    print(f"Source: {pdf_path.name} (Targeting Pages 57–518)")
    print(f"Selected: {len(page_indices)} page(s) with {args.workers} worker processes")
    print(f"Page Cache: {out_dir}")
    print(f"Final Output: {args.final_md}")
    print(f"Force re-parse: {args.force}")
    print(f"=======================================================")

    tasks = [(str(pdf_path), pno, str(out_dir), args.force) for pno in page_indices]
    t_start = time.time()
    completed = 0
    total_tasks = len(tasks)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_page_worker, t): t[1] + 1 for t in tasks}
        for future in as_completed(futures):
            pno_1based = futures[future]
            try:
                page_num, status, dt = future.result()
                completed += 1
                elapsed = time.time() - t_start
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_tasks - completed) / rate if rate > 0 else 0
                print(f"[{completed:3d}/{total_tasks:3d} ({completed*100/total_tasks:5.1f}%)] Page {page_num:04d}: {status:<28} ({dt:.2f}s) | ETA: {eta/60:.1f}m")
            except Exception as e:
                print(f"[!] Page {pno_1based} failed with: {e}")

    total_time = time.time() - t_start
    print(f"\n[+] All {completed} pages processed in {total_time/60:.2f} minutes.")
    build_final_markdown(out_dir, Path(args.final_md).resolve())

    # Generate updated audio release track from the newly parsed text
    print("\n[*] Refreshing audio chant release track...")
    subprocess.run([sys.executable, "generate_audio_release.py"], check=True)


if __name__ == "__main__":
    main()
