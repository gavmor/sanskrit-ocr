#!/home/user/Desktop/sanskrit-ocr/.venv/bin/python
"""
High-Performance Sanskrit PDF to Markdown Converter
Converts Pingree's Yavanajātaka (PingreeYJ_1.pdf) into clean Markdown.
- Pages 1-56: English introductory apparatus
- Pages 57-518: Classical Sanskrit Devanagari verses with critical apparatus
- Multi-process parallel execution with batch inference
- Resumable page-level checkpointing
"""

import os
import sys
import io
import re
import time
import json
import base64
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import requests
import pymupdf
import pytesseract
from PIL import Image
from devanagari_validator import sanitize_devanagari_unicode, validate_devanagari_line


MODEL_URL = "http://127.0.0.1:8501/v1/models/sanskrit:predict"

def decode_output(out_str):
    if not out_str:
        return ""
    out_word = ""
    i = 0
    n = len(out_str)
    while i < n:
        if out_str[i:i+2] in ("23", "24") and i + 4 <= n:
            try:
                code = int(out_str[i:i+4])
                out_word += chr(code)
                i += 4
                continue
            except ValueError:
                pass
        if out_str[i:i+2] == "12" and i + 3 <= n:
            try:
                code = int(out_str[i:i+3])
                out_word += chr(code)
                i += 3
                continue
            except ValueError:
                pass
        if i + 2 <= n and (out_str[i:i+2] in ("32", "35", "39", "44", "45", "46", "95") or ("48" <= out_str[i:i+2] <= "57")):
            try:
                code = int(out_str[i:i+2])
                out_word += chr(code)
                i += 2
                continue
            except ValueError:
                pass
        break
    return out_word

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

def process_page_worker(args_tuple):
    pdf_path, pno, out_dir_str = args_tuple
    page_num = pno + 1
    page_md_path = Path(out_dir_str) / f"page_{page_num:04d}.md"
    
    # Check if already processed
    if page_md_path.exists() and page_md_path.stat().st_size > 20:
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

        # Case 1: Pages 1-56 (English Introduction)
        if page_num < 57:
            text = pytesseract.image_to_string(pil_img, lang="eng")
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            body = "\n\n".join(lines) if lines else "*[Blank / Uninscribed Page]*"
            md_content = f"## Page {page_num} (Introduction)\n\n{body}\n"
            with open(page_md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            dt = time.time() - t0
            return page_num, f"intro ({len(lines)} lines)", dt

        # Case 2: Pages 57-520 (Sanskrit Verses & Apparatus)
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
            md_content = f"## Page {page_num}\n\n*[No text detected]*\n"
            with open(page_md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            return page_num, "empty", time.time() - t0

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

        # Batch request to model server
        resp = requests.post(MODEL_URL, json={'signature_name': 'serving_default', 'inputs': b64_list}, timeout=60.0)
        res = resp.json()
        outputs = res.get('outputs', {}).get('output', [])
        if isinstance(outputs, str):
            outputs = [outputs]

        verses = []
        apparatus = []
        for raw_out in outputs:
            decoded = decode_output(raw_out)
            cleaned = clean_repetitions(decoded)
            if not cleaned:
                continue
            sanitized = sanitize_devanagari_unicode(cleaned)
            if not sanitized:
                continue
            script = classify_line_script(sanitized)
            if script == "devanagari":
                verses.append(sanitized)
            else:
                apparatus.append(sanitized)

        # Build Markdown representation
        md_parts = [f"## Page {page_num}"]
        if verses:
            md_parts.append("\n### Sanskrit Verses\n")
            md_parts.append("\n\n".join(verses))
        if apparatus:
            md_parts.append(f"\n<details>\n<summary>Critical Apparatus (Page {page_num})</summary>\n\n" +
                            "\n\n".join(apparatus) + "\n</details>")

        md_content = "\n".join(md_parts) + "\n"
        with open(page_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        dt = time.time() - t0
        return page_num, f"{len(verses)} verses, {len(apparatus)} app", dt

    except Exception as e:
        return page_num, f"error: {str(e)[:50]}", time.time() - t0


def build_final_markdown(out_dir, total_pages, final_md_path):
    print(f"[*] Assembling final markdown document from {total_pages} pages...")
    header = (
        "# The Yavanajātaka of Sphujidhvaja\n\n"
        "**Volume I: Edited Text and Critical Apparatus**  \n"
        "*Edited, Translated, and Commented on by David Pingree*  \n"
        "*Harvard Oriental Series, Volume 48 (Harvard University Press, 1978)*  \n\n"
        "---\n\n"
        "## Table of Contents & Structure\n"
        "- **Part 1 (Pages 1–56)**: Introduction and Manuscript Witnesses\n"
        "- **Part 2 (Pages 57–518)**: Classical Sanskrit Text (Chapters 1–79)\n"
        "- **Part 3 (Pages 519–520)**: Final Colophon and Back Matter\n\n"
        "---\n\n"
    )
    pages_dir = Path(out_dir)
    with open(final_md_path, "w", encoding="utf-8") as out_f:
        out_f.write(header)
        for p in range(1, total_pages + 1):
            p_file = pages_dir / f"page_{p:04d}.md"
            if p_file.exists():
                with open(p_file, "r", encoding="utf-8") as pf:
                    out_f.write(pf.read() + "\n\n---\n\n")
            else:
                out_f.write(f"## Page {p}\n\n*[Page not available]*\n\n---\n\n")
    print(f"[+] Final document written: {final_md_path} ({final_md_path.stat().st_size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Convert PingreeYJ_1.pdf to Sanskrit Markdown")
    parser.add_argument("--pdf", default="/home/user/Downloads/PingreeYJ_1.pdf")
    parser.add_argument("--pages", default="all", help="'all', '57-65', '1-520', etc.")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--out-dir", default="./ocr_markdown/pages")
    parser.add_argument("--final-md", default="PingreeYJ_1_Sanskrit.md")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"[!] PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(pdf_path))
    total_pages = len(doc)
    doc.close()

    if args.pages.lower() == "all":
        page_indices = list(range(total_pages))
    elif "-" in args.pages:
        s, e = args.pages.split("-", 1)
        page_indices = list(range(int(s) - 1, min(total_pages, int(e))))
    else:
        page_indices = [int(p) - 1 for p in args.pages.split(",") if p.strip()]

    print(f"=======================================================")
    print(f"SANSKRIT PDF -> MARKDOWN CONVERTER")
    print(f"Source: {pdf_path.name} ({total_pages} pages)")
    print(f"Selected: {len(page_indices)} page(s) with {args.workers} worker processes")
    print(f"Page Cache: {out_dir}")
    print(f"Final Output: {args.final_md}")
    print(f"=======================================================")

    tasks = [(str(pdf_path), pno, str(out_dir)) for pno in page_indices]
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
                print(f"[{completed:3d}/{total_tasks:3d} ({completed*100/total_tasks:5.1f}%)] Page {page_num:04d}: {status:<24} ({dt:.2f}s) | ETA: {eta/60:.1f}m")
            except Exception as e:
                print(f"[!] Page {pno_1based} failed with: {e}")

    total_time = time.time() - t_start
    print(f"\n[+] All {completed} pages processed in {total_time/60:.2f} minutes.")
    build_final_markdown(out_dir, total_pages, Path(args.final_md).resolve())


if __name__ == "__main__":
    main()
