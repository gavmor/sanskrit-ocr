#!/usr/bin/env python3
"""
ByT5 Byte-Level Sequence-to-Sequence Post-OCR Correction Pipeline for Sanskrit.

Context & Motivation:
- Historical Sanskrit editions such as David Pingree's Yavanajātaka (1978) exhibit
  complex vertical conjunct ligatures, floating virāmas, dropped mātrās, visual
  consonant confusions (e.g. n/l, kh/m, gh/dh), and attention repetition loops.
- Standard word-level spellcheckers fail due to Sanskrit's agglutinative morphology
  and sandhi rules.
- ByT5 (Xue et al., 2022) operates directly on UTF-8 bytes without a subword tokenizer,
  avoiding the ~90% subword fragmentation rate of SentencePiece on Indic conjuncts
  and eliminating out-of-vocabulary (<unk>) tokens.

Prior Art:
- 'A Benchmark and Dataset for Post-OCR text correction in Sanskrit' (EMNLP 2022, ayushbits/pe-ocr-sanskrit)
- 'One Model is All You Need: ByT5-Sanskrit, a Unified Model for Sanskrit NLP Tasks' (EMNLP 2024)
- 'Upcycle Your OCR: Reusing OCRs for Post-OCR Text Correction in Romanised Sanskrit' (CoNLL 2018)
"""

import os
import sys
import re
import time
import argparse
import unicodedata
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

# Ensure C_INCLUDE_PATH is present for Triton JIT compiler on environments without python3-dev in /usr/include
DEFAULT_UV_INCLUDE = "/home/user/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/include/python3.12"
if os.path.exists(DEFAULT_UV_INCLUDE) and "C_INCLUDE_PATH" not in os.environ:
    os.environ["C_INCLUDE_PATH"] = DEFAULT_UV_INCLUDE

try:
    import editdistance as ed  # type: ignore
except ImportError:
    class _EdFallback:
        @staticmethod
        def eval(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m + 1): dp[i][0] = i
            for j in range(n + 1): dp[0][j] = j
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    cost = 0 if s1[i-1] == s2[j-1] else 1
                    dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
            return dp[m][n]
    ed = _EdFallback()  # type: ignore

try:
    import torch  # type: ignore
except ImportError:
    torch = None

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # type: ignore
except ImportError:
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None

# Local repo imports for validation and scansion
try:
    from devanagari_validator import sanitize_devanagari_unicode
except ImportError:
    def sanitize_devanagari_unicode(text: str) -> str:
        return text

try:
    from ocr_pdf import clean_repetitions
except ImportError:
    def clean_repetitions(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'([^\s])\1{3,}', r'\1\1', text)
        for l in range(35, 2, -1):
            pattern = re.compile(rf'(.{{{l}}}?)\1{{2,}}')
            text = pattern.sub(r'\1', text)
        return text.strip()


# Canonical Ground Truth for Yavanajātaka 1.14 (Page 57) from Pingree Volume 2 (1978)
PINGREE_PAGE57_VERSE_14 = {
    "verse_id": "1.14 (Page 57)",
    "chapter": 1,
    "meter": "Indravajrā / Upajāti (11 syllables per pāda)",
    "translation": (
        "The first (sign) is traditionally said to have the shape of a ram; "
        "it is called by the ancients the head of Kala. "
        "Its places are the paths of goats and sheep, caves, mountains, "
        "(hideouts of) thieves, and (places where there are) fire, metals, mines, and jewels."
    ),
    "quarters": [
        {
            "id": "Q1",
            "name": "Aries (मेषवपुः)",
            "ground_truth": "आद्यः स्मृतो मेषवपुस्ततोऽथ",
            "indicocr_v2": "मह्याक्रे मस्तम्नम्",
            "tesseract_san": "ञताय्यः स््पृत्यो सेष्वस्स्बा- तस्तः",
            "tesseract_dev": "अराय्यः स्ख्ूतो सेष्यस्तस्ब्रातस्लि ः",
        },
        {
            "id": "Q2",
            "name": "Head of Kāla (कालस्य मूर्द्धा)",
            "ground_truth": "कालस्य मूर्द्धा कथितः पुराणैः ।",
            "indicocr_v2": "कानास्य श्रा राट्रानैहः कुटादैस्त्यः कुराणान्त्यः कु",
            "tesseract_san": "कात्यस्य श्वा गदिप्त्वः प्राणैः ।",
            "tesseract_dev": "कान्यस्य सूघ्या राट: जराष्णैः।",
        },
        {
            "id": "Q3",
            "name": "Pastures and Caves (अजाविक)",
            "ground_truth": "स्थानान्यजाविकगोधराद्रि-",
            "indicocr_v2": "स्तो जातास्तंच्छलम्",
            "tesseract_san": "स्तो 5ज्नाधिकास्वैन्रकृन्यराद्धि--",
            "tesseract_dev": "स्तो ऽञजाजिकास्तंनरकुन्स्‌राद्रि-",
        },
        {
            "id": "Q4",
            "name": "Mines and Gems (रत्नाकर)",
            "ground_truth": "स्तेनाग्निरत्नाकरलोहभाञ्जि ॥१४॥",
            "indicocr_v2": "स्नित्र गाकरस्रस्त्यस्त्राक्ष",
            "tesseract_san": "स्वेता र्ितिश्उात्व्वाकररत्तमजस्ति - ॥2४६॥",
            "tesseract_dev": "स्तेत्ास्नि्पात्याकर रन्नम्जस्मि - ॥१४॥",
        },
    ],
}


@dataclass
class CorrectionResult:
    raw_input: str
    cleaned_input: str
    corrected_text: str
    model_raw_output: str
    latency_sec: float
    fallback_applied: bool = False
    notes: List[str] = field(default_factory=list)


@dataclass
class MetricsResult:
    cer: float
    wer: float
    edit_distance: int
    gt_length: int
    exact_match: bool


def compute_metrics(predicted: str, ground_truth: str) -> MetricsResult:
    """Computes CER, WER, and exact match against ground truth using NFC normalization."""
    p_norm = unicodedata.normalize("NFC", str(predicted).strip())
    gt_norm = unicodedata.normalize("NFC", str(ground_truth).strip())

    char_dist = ed.eval(p_norm, gt_norm)
    gt_len = max(len(gt_norm), 1)
    cer = char_dist / gt_len

    p_words = p_norm.split()
    gt_words = gt_norm.split()
    word_dist = ed.eval(p_words, gt_words)
    gt_word_len = max(len(gt_words), 1)
    wer = word_dist / gt_word_len

    return MetricsResult(
        cer=cer,
        wer=wer,
        edit_distance=char_dist,
        gt_length=len(gt_norm),
        exact_match=(p_norm == gt_norm),
    )


def normalize_dandas(text: str) -> str:
    """Canonicalizes ASCII pipe characters and spaces around Devanagari dandas."""
    if not text:
        return ""
    # Normalize ASCII pipe or double pipe to Devanagari dandas
    text = text.replace("||", "॥").replace("|", "।")
    # Collapse multiple consecutive dandas
    text = re.sub(r'[।॥]{2,}', '॥', text)
    # Remove space preceding danda
    text = re.sub(r'\s+([।॥])', r'\1', text)
    # Ensure space following danda if followed by alphabetic characters
    text = re.sub(r'([।॥])(?=[^\s\n|।॥०-९0-9])', r'\1 ', text)
    # Clean leading stray dandas
    text = re.sub(r'^[।॥]\s*', '', text)
    return text.strip()


class ByT5SanskritCorrector:
    """
    Lightweight Post-OCR Byte-Level Seq2Seq Corrector for Classical Sanskrit.
    
    Handles:
    1. Attention repetition collapse and Unicode canonicalization.
    2. Proper UTF-8 byte budgeting during beam search generation.
    3. Conservative drift filtering (safeguards against seq2seq hallucinations).
    4. Post-generation danda and ligature cleanup.
    """

    DEFAULT_MODEL = "tusharislampure29/byt5-sanskrit-ocr"

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        dtype: Optional[Any] = None,
        max_drift_cer: float = 1.25,
        prefix: str = "correct: ",
    ):
        self.model_name = model_name_or_path
        self.max_drift_cer = max_drift_cer
        self.prefix = prefix

        if torch is not None:
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            if dtype is None:
                self.dtype = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
            else:
                self.dtype = dtype
        else:
            self.device = "cpu"
            self.dtype = None
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Loads ByT5 tokenizer and model onto target device."""
        if self.model is not None:
            return

        if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
            raise ImportError("Hugging Face `transformers` is required to run ByT5SanskritCorrector.")

        print(f"[*] Loading ByT5 tokenizer & model '{self.model_name}' on {self.device} ({self.dtype})...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        print(f"[+] Model loaded in {time.time() - t0:.2f}s.")

    def preprocess(self, raw_line: str) -> str:
        """Cleans input line: collapses attention loops and canonicalizes Unicode."""
        # 1. Strip markdown quote indicators and edge whitespace
        line = re.sub(r'^[>\s]+', '', raw_line).strip()
        # 2. Collapse runaway attention repetitions
        line = clean_repetitions(line)
        # 3. Canonicalize Devanagari Unicode OpenType clusters
        line = sanitize_devanagari_unicode(line)
        # 4. Normalize NFC
        line = unicodedata.normalize("NFC", line)
        return line

    def postprocess(self, raw_pred: str, cleaned_input: str) -> Tuple[str, bool, List[str]]:
        """
        Applies conservative post-filtering:
        - Detects runaway repetitions in model output.
        - Checks character drift against raw input.
        - Canonicalizes dandas.
        """
        notes = []
        fallback = False

        # Trim repeating generation phrases
        text = clean_repetitions(raw_pred)
        text = sanitize_devanagari_unicode(text)
        text = normalize_dandas(text)

        # Check runaway length ratio: Devanagari bytes vs input bytes
        input_bytes = len(cleaned_input.encode("utf-8"))
        output_bytes = len(text.encode("utf-8"))

        if input_bytes > 0 and output_bytes > (input_bytes * 2.2):
            notes.append(f"Output byte length ({output_bytes}) exceeded safety limit (input {input_bytes}); fallback applied.")
            return cleaned_input, True, notes

        # Drift distance check: if output drifts wildly from raw OCR (CER > max_drift_cer)
        drift_dist = ed.eval(cleaned_input, text)
        drift_cer = drift_dist / max(len(cleaned_input), 1)

        if drift_cer > self.max_drift_cer and len(cleaned_input) >= 8:
            notes.append(f"Model drift CER {drift_cer:.1%} exceeded threshold {self.max_drift_cer:.1%}; keeping cleaned raw input.")
            return cleaned_input, True, notes

        return text, fallback, notes

    def correct_line(
        self,
        line: str,
        num_beams: int = 4,
        byte_expansion_ratio: float = 1.35,
        min_extra_bytes: int = 32,
    ) -> CorrectionResult:
        """
        Corrects a single OCR line using byte-level ByT5 generation.
        
        Calculates byte budgeting based on UTF-8 byte length to prevent premature
        truncation on Devanagari text (which uses 3 bytes per character).
        """
        self.load_model()
        t0 = time.time()

        cleaned = self.preprocess(line)
        if not cleaned:
            return CorrectionResult(
                raw_input=line,
                cleaned_input="",
                corrected_text="",
                model_raw_output="",
                latency_sec=0.0,
                fallback_applied=False,
                notes=["Empty input"],
            )

        prompt = f"{self.prefix}{cleaned}" if self.prefix else cleaned
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model or tokenizer failed to initialize.")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # In ByT5, 1 token == 1 byte. Devanagari characters are 3 bytes each.
        input_byte_len = len(cleaned.encode("utf-8"))
        max_new_bytes = int(input_byte_len * byte_expansion_ratio) + min_extra_bytes

        with torch.no_grad():
            output_ids = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_bytes,
                num_beams=num_beams,
                length_penalty=1.0,
                early_stopping=True,
            )

        raw_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        final_text, fallback, notes = self.postprocess(raw_output, cleaned)
        latency = time.time() - t0

        return CorrectionResult(
            raw_input=line,
            cleaned_input=cleaned,
            corrected_text=final_text,
            model_raw_output=raw_output,
            latency_sec=latency,
            fallback_applied=fallback,
            notes=notes,
        )

    def correct_lines(self, lines: List[str], num_beams: int = 4) -> List[CorrectionResult]:
        """Corrects a batch of Sanskrit OCR lines."""
        return [self.correct_line(l, num_beams=num_beams) for l in lines]

    def correct_markdown_page(
        self,
        md_content: str,
        page_num: int = 57,
        num_beams: int = 4,
    ) -> Tuple[str, List[CorrectionResult]]:
        """
        Extracts verse lines for a target page from markdown, corrects each line,
        and returns updated markdown content and correction traces.
        """
        page_header = f"## पत्रम् ५७ (Page {page_num})"
        if page_header not in md_content:
            page_header_alt = f"Page {page_num}"
            if page_header_alt not in md_content:
                raise ValueError(f"Page marker for page {page_num} not found in markdown.")

        lines = md_content.splitlines()
        in_target_page = False
        new_lines = []
        results = []

        for line in lines:
            if f"Page {page_num}" in line and line.startswith("##"):
                in_target_page = True
                new_lines.append(line)
                continue
            elif in_target_page and line.startswith("##") and f"Page {page_num}" not in line:
                in_target_page = False
                new_lines.append(line)
                continue

            if in_target_page and line.strip().startswith(">"):
                raw_verse = line.strip().lstrip(">").strip()
                res = self.correct_line(raw_verse, num_beams=num_beams)
                results.append(res)
                new_lines.append(f"> {res.corrected_text}")
            else:
                new_lines.append(line)

        return "\n".join(new_lines), results


def evaluate_page57_verse14(corrector: ByT5SanskritCorrector) -> Dict[str, Any]:
    """
    Rigorously evaluates ByT5 post-OCR correction on Verse 1.14 of Page 57
    across three OCR engines: IndicOCR-v2, Tesseract Sanskrit, and Tesseract Devanagari.
    """
    data = PINGREE_PAGE57_VERSE_14
    ocr_engines = ["indicocr_v2", "tesseract_san", "tesseract_dev"]
    engine_labels = {
        "indicocr_v2": "IndicOCR-v2 (Raw Attention-LSTM)",
        "tesseract_san": "Tesseract 5.5 ('san' model)",
        "tesseract_dev": "Tesseract 5.5 ('Devanagari' model)",
    }

    summary = {}

    print("=" * 80)
    print("BYT5 BYTE-LEVEL POST-OCR CORRECTION BENCHMARK: YAVANAJĀTAKA 1.14 (PAGE 57)")
    print(f"Author: David Pingree (Harvard Oriental Series Vol. 48, Page 57)")
    print(f"Translation: \"{data['translation']}\"")
    print("=" * 80)

    for eng in ocr_engines:
        print(f"\nEvaluating Engine: {engine_labels[eng]}")
        print("-" * 80)
        tot_gt_chars = 0
        tot_raw_ed = 0
        tot_corr_ed = 0
        tot_gt_words = 0
        tot_raw_wed = 0
        tot_corr_wed = 0

        quarter_reports = []

        quarters: List[Dict[str, Any]] = data["quarters"]  # type: ignore[assignment]
        for q in quarters:
            qid = str(q["id"])
            gt = str(q["ground_truth"])
            raw_ocr = str(q[eng])

            res = corrector.correct_line(raw_ocr)
            corr_text = res.corrected_text

            raw_m = compute_metrics(raw_ocr, gt)
            corr_m = compute_metrics(corr_text, gt)

            tot_gt_chars += raw_m.gt_length
            tot_raw_ed += raw_m.edit_distance
            tot_corr_ed += corr_m.edit_distance

            tot_gt_words += len(gt.split())
            tot_raw_wed += ed.eval(raw_ocr.split(), gt.split())
            tot_corr_wed += ed.eval(corr_text.split(), gt.split())

            delta_cer = corr_m.cer - raw_m.cer
            delta_str = f"{delta_cer:+.1%}" if delta_cer != 0 else "0.0%"

            print(f"[{qid} - {q.get('name', '')}]")
            print(f"  Ground Truth:    {gt}")
            print(f"  Raw OCR:         {raw_ocr:<38} | CER: {raw_m.cer:.1%} | WER: {raw_m.wer:.1%}")
            print(f"  ByT5 Corrected:  {corr_text:<38} | CER: {corr_m.cer:.1%} | WER: {corr_m.wer:.1%} (Δ: {delta_str})")
            if res.notes:
                print(f"  Safety Notes:    {', '.join(res.notes)}")
            print()

            quarter_reports.append({
                "quarter": qid,
                "ground_truth": gt,
                "raw_ocr": raw_ocr,
                "corrected": corr_text,
                "raw_cer": raw_m.cer,
                "corrected_cer": corr_m.cer,
                "delta_cer": delta_cer,
                "raw_wer": raw_m.wer,
                "corrected_wer": corr_m.wer,
            })

        overall_raw_cer = tot_raw_ed / tot_gt_chars
        overall_corr_cer = tot_corr_ed / tot_gt_chars
        overall_raw_wer = tot_raw_wed / max(tot_gt_words, 1)
        overall_corr_wer = tot_corr_wed / max(tot_gt_words, 1)

        summary[eng] = {
            "label": engine_labels[eng],
            "raw_cer": overall_raw_cer,
            "corrected_cer": overall_corr_cer,
            "raw_wer": overall_raw_wer,
            "corrected_wer": overall_corr_wer,
            "delta_cer": overall_corr_cer - overall_raw_cer,
            "delta_wer": overall_corr_wer - overall_raw_wer,
            "quarters": quarter_reports,
        }

        print(f"  >> Overall {engine_labels[eng]}:")
        print(f"     CER: {overall_raw_cer:.1%} -> {overall_corr_cer:.1%} (Δ: {overall_corr_cer - overall_raw_cer:+.1%})")
        print(f"     WER: {overall_raw_wer:.1%} -> {overall_corr_wer:.1%} (Δ: {overall_corr_wer - overall_raw_wer:+.1%})")

    print("\n" + "=" * 80)
    print("ACCURACY COMPARISON MATRIX (VERSE 1.14):")
    print("=" * 80)
    print(f"{'OCR Engine':<32} | {'Raw CER':<9} | {'ByT5 CER':<9} | {'Δ CER':<8} | {'Raw WER':<9} | {'ByT5 WER':<9}")
    print("-" * 80)
    for eng, res in summary.items():
        print(f"{res['label']:<32} | {res['raw_cer']:<9.1%} | {res['corrected_cer']:<9.1%} | {res['delta_cer']:<+8.1%} | {res['raw_wer']:<9.1%} | {res['corrected_wer']:<9.1%}")
    print("=" * 80)

    return summary


def evaluate_emnlp2022_benchmark(corrector: ByT5SanskritCorrector, max_samples: int = 25) -> Dict[str, float]:
    """
    Evaluates ByT5 corrector on out-of-domain historical/astronomical Sanskrit texts
    from the EMNLP 2022 benchmark dataset (ayushbits/pe-ocr-sanskrit).
    """
    import urllib.request
    import io
    import pandas as pd  # type: ignore

    url = "https://raw.githubusercontent.com/ayushbits/pe-ocr-sanskrit/main/test_devnagari.csv"
    print(f"\n[*] Fetching EMNLP 2022 benchmark samples from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "curl/7.81.0"})
    with urllib.request.urlopen(req) as resp:
        content = resp.read().decode("utf-8")

    df = pd.read_csv(io.StringIO(content)).dropna().head(max_samples)
    print(f"[+] Loaded {len(df)} evaluation samples.")

    tot_gt_chars = 0
    tot_raw_ed = 0
    tot_corr_ed = 0

    print("-" * 80)
    print(f"{'Sample #':<10} | {'Raw CER':<10} | {'ByT5 CER':<10} | {'Input Preview':<40}")
    print("-" * 80)

    for idx, row in df.iterrows():
        raw = row["input_text"].strip()
        gt = row["target_text"].strip()

        res = corrector.correct_line(raw)
        corr = res.corrected_text

        raw_m = compute_metrics(raw, gt)
        corr_m = compute_metrics(corr, gt)

        tot_gt_chars += raw_m.gt_length
        tot_raw_ed += raw_m.edit_distance
        tot_corr_ed += corr_m.edit_distance

        preview = raw[:38] + ("..." if len(raw) > 38 else "")
        print(f"{idx+1:<10} | {raw_m.cer:<10.1%} | {corr_m.cer:<10.1%} | {preview}")

    mean_raw_cer = tot_raw_ed / max(tot_gt_chars, 1)
    mean_corr_cer = tot_corr_ed / max(tot_gt_chars, 1)

    print("-" * 80)
    print(f"EMNLP 2022 Benchmark Aggregate ({len(df)} samples):")
    print(f"  Raw OCR CER:     {mean_raw_cer:.2%}")
    print(f"  ByT5 Corrected:  {mean_corr_cer:.2%} (Δ: {mean_corr_cer - mean_raw_cer:+.2%})")
    print("-" * 80)

    return {
        "samples": len(df),
        "raw_cer": mean_raw_cer,
        "corrected_cer": mean_corr_cer,
        "delta_cer": mean_corr_cer - mean_raw_cer,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="ByT5 Sanskrit Post-OCR Correction Pipeline")
    parser.add_argument("--model", type=str, default=ByT5SanskritCorrector.DEFAULT_MODEL, help="Model name or path")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument("--evaluate-v14", action="store_true", help="Run Verse 1.14 accuracy evaluation")
    parser.add_argument("--evaluate-emnlp2022", action="store_true", help="Run EMNLP 2022 benchmark test")
    parser.add_argument("--samples", type=int, default=25, help="Number of benchmark samples to evaluate")
    parser.add_argument("--correct-text", type=str, default=None, help="Correct a single Sanskrit OCR line")
    parser.add_argument("--markdown-file", type=str, default=None, help="Path to markdown file (e.g. PingreeYJ_1_Sanskrit.md)")
    parser.add_argument("--page", type=int, default=57, help="Page number to correct in markdown file")
    return parser.parse_args()


def main():
    args = parse_args()

    corrector = ByT5SanskritCorrector(
        model_name_or_path=args.model,
        device=args.device,
    )

    if args.evaluate_v14:
        evaluate_page57_verse14(corrector)

    if args.evaluate_emnlp2022:
        evaluate_emnlp2022_benchmark(corrector, max_samples=args.samples)

    if args.correct_text:
        res = corrector.correct_line(args.correct_text)
        print("\n--- ByT5 Single Line Correction ---")
        print(f"Raw Input:       {res.raw_input}")
        print(f"Preprocessed:    {res.cleaned_input}")
        print(f"Corrected:       {res.corrected_text}")
        print(f"Latency:         {res.latency_sec:.3f}s")
        if res.notes:
            print(f"Notes:           {', '.join(res.notes)}")

    if args.markdown_file:
        md_path = Path(args.markdown_file)
        if not md_path.exists():
            print(f"Error: Markdown file {md_path} not found.")
            sys.exit(1)
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"\n[*] Processing Page {args.page} in {md_path}...")
        updated_content, results = corrector.correct_markdown_page(content, page_num=args.page)
        print(f"[+] Corrected {len(results)} lines on Page {args.page}.")
        for idx, r in enumerate(results, 1):
            print(f"  Line {idx}: {r.raw_input} -> {r.corrected_text}")

    if not (args.evaluate_v14 or args.evaluate_emnlp2022 or args.correct_text or args.markdown_file):
        # Default behavior: run Verse 1.14 evaluation
        evaluate_page57_verse14(corrector)


if __name__ == "__main__":
    main()
