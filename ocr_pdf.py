#!/home/user/Desktop/sanskrit-ocr/.venv/bin/python
"""
Sanskrit OCR Runner for PDF Documents
Targeting Pingree's Yavanajātaka (Harvard Oriental Series Vol. 48)
and other classical Sanskrit PDF texts.

Uses the IndicOCR-v2 (Attention-LSTM) model from:
"An OCR for Classical Indic Documents Containing Arbitrarily Long Words" (CVPRW 2020)
by Dwivedi, Saluja, Sarvadevabhatla (CVIT, IIIT Hyderabad).
"""

import os
import sys
import argparse
import base64
import json
import re
import time
from pathlib import Path
import subprocess
import io

import cv2
import numpy as np
import requests
import pymupdf
import pytesseract
from PIL import Image
from devanagari_validator import sanitize_devanagari_unicode, validate_devanagari_line


DEFAULT_MODEL_URL = "http://127.0.0.1:8501/v1/models/sanskrit:predict"
CONTAINER_NAME = "sanskrit-ocr-server"

def ensure_server_running(server_url=DEFAULT_MODEL_URL, model_dir=None):
    """Checks if the TensorFlow Serving model server is active; if not, attempts to start it via Docker."""
    try:
        r = requests.get("http://127.0.0.1:8501/v1/models/sanskrit", timeout=1.5)
        if r.status_code == 200:
            return True
    except Exception:
        pass

    print("[*] Model server not detected at 127.0.0.1:8501. Attempting to start via Docker...")
    if model_dir is None:
        repo_root = Path(__file__).resolve().parent
        model_dir = repo_root / "get_predictions" / "c3_attention"

    if not model_dir.exists():
        print(f"[!] Model directory not found: {model_dir}", file=sys.stderr)
        return False

    cmd = [
        "docker", "run", "-d",
        "--name", CONTAINER_NAME,
        "-p", "127.0.0.1:8501:8501",
        "-v", f"{model_dir.resolve()}:/models/sanskrit",
        "-e", "MODEL_NAME=sanskrit",
        "tensorflow/serving:1.15.0"
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("[*] Started TensorFlow Serving container. Waiting for model initialization...")
        for _ in range(15):
            time.sleep(1)
            try:
                r = requests.get("http://127.0.0.1:8501/v1/models/sanskrit", timeout=1.0)
                if r.status_code == 200:
                    print("[+] Model server is ready.")
                    return True
            except Exception:
                pass
    except subprocess.CalledProcessError as e:
        print(f"[!] Failed to start container: {e.stderr.decode()}", file=sys.stderr)

    return False


def decode_output(out_str):
    """
    Decodes the IndicOCR decimal-encoded character prediction string into UTF-8 Devanagari text.
    Handles 4-digit Devanagari codepoints, 3-digit symbols (|, ~), and 2-digit ASCII characters.
    """
    if not out_str:
        return ""
    out_word = ""
    i = 0
    n = len(out_str)
    while i < n:
        # Devanagari Unicode block U+0900 (2304) to U+097F (2431)
        if out_str[i:i+2] in ("23", "24") and i + 4 <= n:
            try:
                code = int(out_str[i:i+4])
                out_word += chr(code)
                i += 4
                continue
            except ValueError:
                pass

        # 3-digit symbols: 124 ('|'), 126 ('~')
        if out_str[i:i+2] == "12" and i + 3 <= n:
            try:
                code = int(out_str[i:i+3])
                out_word += chr(code)
                i += 3
                continue
            except ValueError:
                pass

        # 2-digit ASCII: space (32), punctuation (35, 44, 45, 46, 95), digits (48-57)
        if i + 2 <= n and (out_str[i:i+2] in ("32", "35", "39", "44", "45", "46", "95") or ("48" <= out_str[i:i+2] <= "57")):
            try:
                code = int(out_str[i:i+2])
                out_word += chr(code)
                i += 2
                continue
            except ValueError:
                pass

        # Unrecognized prefix or end of valid stream
        break

    return out_word


def clean_repetitions(text):
    """
    Cleans runaway seq2seq attention repetitions at line ends or within lines.
    Trims repeating subword/token patterns (common artifact in attention OCR).
    """
    if not text:
        return ""
    # Collapse consecutive identical non-whitespace characters if repeated 4+ times
    text = re.sub(r'([^\s])\1{3,}', r'\1\1', text)
    # Collapse repeating phrases/substrings of lengths 35 down to 3
    for l in range(35, 2, -1):
        pattern = re.compile(rf'(.{{{l}}}?)\1{{2,}}')
        text = pattern.sub(r'\1', text)
    return text.strip()


def classify_line_script(text):
    """
    Determines if a line is Sanskrit Devanagari verse or Latin apparatus/header.
    """
    dev_count = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total_alpha = sum(1 for c in text if not c.isspace() and c not in '0123456789.,-—|/:;#~')
    if total_alpha > 0 and (dev_count / total_alpha) >= 0.35:
        return "devanagari"
    return "latin"


def parse_page_ranges(range_str, max_pages):
    """Parses page strings like '57', '57-60', '57,58,62', 'all' into 0-based page indices."""
    if range_str.lower() == "all":
        return list(range(max_pages))
    pages = set()
    parts = range_str.split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = max(1, int(start_s))
            end = min(max_pages, int(end_s))
            for p in range(start, end + 1):
                pages.add(p - 1)
        else:
            p = int(part)
            if 1 <= p <= max_pages:
                pages.add(p - 1)
    return sorted(list(pages))


def extract_page_image(doc, page_num):
    """
    Extracts the full-resolution scan image directly from the PDF page object.
    Falls back to rendering if embedded image extraction is unavailable.
    """
    page = doc[page_num]
    ilist = page.get_images()
    if ilist:
        xref = ilist[0][0]
        base_img = doc.extract_image(xref)
        img_bytes = base_img["image"]
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("L")
        return np.array(pil_img)

    # Fallback: rasterize page at 300 DPI
    pix = page.get_pixmap(dpi=300)
    pil_img = Image.open(io.BytesIO(pix.tobytes())).convert("L")
    return np.array(pil_img)


def detect_lines(img_np):
    """
    Detects text line bounding boxes using Tesseract Devanagari layout analysis.
    Filters scanner borders, vertical rules, and noise artifacts.
    """
    H, W = img_np.shape[:2]
    pil_img = Image.fromarray(img_np)
    data = pytesseract.image_to_data(pil_img, lang="Devanagari", output_type=pytesseract.Output.DICT)
    n = len(data["level"])
    lines = []
    for i in range(n):
        if data["level"][i] == 4:  # text line level
            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]

            # Filter scanner artifacts and marginalia
            if w < 40 or h < 18:
                continue
            if h > w * 1.3:  # vertical line / scanner artifact
                continue
            if x > W - 60 or x < 40:  # gutter border
                continue
            if y < 40 or y > H - 40:
                continue

            lines.append((y, x, w, h))

    # Sort strictly from top to bottom
    lines.sort(key=lambda b: (b[0], b[1]))
    return lines


def predict_line(roi, server_url=DEFAULT_MODEL_URL):
    """
    Normalizes contrast, resizes to height=32, and queries the IndicOCR model server.
    """
    h, w = roi.shape[:2]
    if h == 0 or w == 0:
        return "", 0.0

    # Pure numpy contrast stretch (stretches ink from gray to solid black, type-safe)
    r_min = float(roi.min())
    r_max = float(roi.max())
    if r_max > r_min:
        roi_norm = ((roi.astype(np.float32) - r_min) * (255.0 / (r_max - r_min))).astype(np.uint8)
    else:
        roi_norm = roi.copy()
    # Scale to fixed height 32 preserving aspect ratio
    r = 32.0 / float(h)
    new_w = max(1, int(w * r))
    resized = cv2.resize(roi_norm, (new_w, 32), interpolation=cv2.INTER_CUBIC)

    # Encode as JPEG
    success, buf = cv2.imencode(".jpg", resized)
    if not success:
        return "", 0.0

    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    body = {"signature_name": "serving_default", "inputs": [{"b64": b64}]}
    headers = {"cache-control": "no-cache", "content-type": "application/json"}

    try:
        resp = requests.post(server_url, json=body, headers=headers, timeout=10.0)
        res = resp.json()
        raw_out = res["outputs"]["output"]
        prob = res["outputs"].get("probability", 0.0)
        decoded = decode_output(raw_out)
        cleaned = clean_repetitions(decoded)
        sanitized = sanitize_devanagari_unicode(cleaned)
        is_valid, issues, _ = validate_devanagari_line(sanitized)
        return sanitized, prob, is_valid, issues
    except Exception:
        return "", 0.0, False, ["InferenceError"]


def process_pdf(pdf_path, pages_arg="57-60", output_dir="./ocr_results", server_url=DEFAULT_MODEL_URL, save_vis=False):
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        print(f"[!] PDF file not found: {pdf_path}", file=sys.stderr)
        return False

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    pages_out_dir = out_path / "pages"
    pages_out_dir.mkdir(exist_ok=True)
    if save_vis:
        (out_path / "visualizations").mkdir(exist_ok=True)

    print(f"[*] Opening document: {pdf_path.name}")
    doc = pymupdf.open(str(pdf_path))
    total_pages = len(doc)
    page_indices = parse_page_ranges(pages_arg, total_pages)
    print(f"[*] Total PDF pages: {total_pages}. Processing {len(page_indices)} page(s): {[p+1 for p in page_indices]}")

    full_text_entries = []
    all_sanskrit_verses = []
    all_metadata = []
    start_total_time = time.time()
    for idx, pno in enumerate(page_indices, 1):
        page_num_1based = pno + 1
        t0 = time.time()
        print(f"\n--- [Page {page_num_1based}/{total_pages} ({idx}/{len(page_indices)})] ---")

        # 1. Extract image
        img_np = extract_page_image(doc, pno)
        h_page, w_page = img_np.shape[:2]
        print(f"  Extracted page image: {w_page}x{h_page}")

        # 2. Line detection
        lines = detect_lines(img_np)
        print(f"  Found {len(lines)} text lines")

        verse_lines = []
        apparatus_lines = []
        page_lines_meta = []
        vis_img = None
        if save_vis:
            vis_img = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

        for l_idx, (y, x, w, h) in enumerate(lines, 1):
            roi = img_np[y:y+h, x:x+w]
            text, prob, is_valid, issues = predict_line(roi, server_url)
            script = classify_line_script(text)
            if text:
                if script == "devanagari":
                    verse_lines.append(text)
                    print(f"    [Sanskrit L{l_idx:02d}]: {text}")
                else:
                    apparatus_lines.append(text)
                    print(f"    [Apparatus L{l_idx:02d}]: {text}")

            page_lines_meta.append({
                "line_idx": l_idx,
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "text": text,
                "script": script,
                "unicode_valid": is_valid,
                "unicode_issues": issues,
                "probability": float(prob)
            })

            if save_vis and vis_img is not None:
                color = (0, 200, 0) if script == "devanagari" else (200, 100, 0)
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(vis_img, str(l_idx), (max(0, x - 30), y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        page_content_blocks = []
        if verse_lines:
            page_content_blocks.append("--- SANSKRIT VERSES ---\n" + "\n".join(verse_lines))
        if apparatus_lines:
            page_content_blocks.append("--- APPARATUS / NOTES ---\n" + "\n".join(apparatus_lines))
        page_str = "\n\n".join(page_content_blocks)

        # Save individual page text
        page_txt_file = pages_out_dir / f"page_{page_num_1based:04d}.txt"
        with open(page_txt_file, "w", encoding="utf-8") as pf:
            pf.write(page_str + "\n")

        if save_vis and vis_img is not None:
            vis_file = out_path / "visualizations" / f"page_{page_num_1based:04d}_annotated.jpg"
            cv2.imwrite(str(vis_file), vis_img)

        full_text_entries.append(f"=== PAGE {page_num_1based} ===\n{page_str}\n")
        if verse_lines:
            all_sanskrit_verses.append(f"--- Page {page_num_1based} ---\n" + "\n".join(verse_lines))
        all_metadata.append({
            "page": page_num_1based,
            "dimensions": {"width": w_page, "height": h_page},
            "line_count": len(lines),
            "lines": page_lines_meta
        })

        dt = time.time() - t0
        print(f"  Page {page_num_1based} completed in {dt:.2f}s -> {page_txt_file.name}")

    # Write combined text file
    combined_file = out_path / "full_transcription.txt"
    with open(combined_file, "w", encoding="utf-8") as cf:
        cf.write("\n".join(full_text_entries))
    # Write pure Sanskrit verses file
    sanskrit_file = out_path / "sanskrit_verses_only.txt"
    with open(sanskrit_file, "w", encoding="utf-8") as sf:
        sf.write("\n\n".join(all_sanskrit_verses) + "\n")


    # Write JSON metadata
    json_file = out_path / "ocr_results.json"
    with open(json_file, "w", encoding="utf-8") as jf:
        json.dump({
            "source_pdf": str(pdf_path),
            "processed_pages": [p+1 for p in page_indices],
            "total_time_seconds": round(time.time() - start_total_time, 2),
            "pages": all_metadata
        }, jf, ensure_ascii=False, indent=2)

    total_time = time.time() - start_total_time
    print(f"\n=======================================================")
    print(f"[+] OCR Complete for {len(page_indices)} page(s) in {total_time:.2f} seconds.")
    print(f"[+] Full transcription : {combined_file}")
    print(f"[+] Individual pages   : {pages_out_dir}/")
    print(f"[+] Structured JSON    : {json_file}")
    print(f"=======================================================")
    return True


def main():
    parser = argparse.ArgumentParser(description="Local Sanskrit OCR for Classical Indic PDFs (Yavanajātaka)")
    parser.add_argument("--pdf", default="/home/user/Downloads/PingreeYJ_1.pdf", help="Target PDF path")
    parser.add_argument("--pages", default="57-60", help="Page range(s) to process, e.g. '57', '57-60', '57,58,62', or 'all'")
    parser.add_argument("--output-dir", default="./ocr_results", help="Directory to save transcription files")
    parser.add_argument("--server-url", default=DEFAULT_MODEL_URL, help="TensorFlow Serving REST URL")
    parser.add_argument("--visualize", action="store_true", help="Save annotated visualization images")
    args = parser.parse_args()

    # Ensure server is running
    if not ensure_server_running(args.server_url):
        print("[!] Could not connect to or start model server.", file=sys.stderr)
        sys.exit(1)

    process_pdf(
        pdf_path=args.pdf,
        pages_arg=args.pages,
        output_dir=args.output_dir,
        server_url=args.server_url,
        save_vis=args.visualize
    )


if __name__ == "__main__":
    main()
