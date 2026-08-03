#!/usr/bin/env python3
"""
parser.py — Phase 1 of skill-matrix

Scans the local certs/ folder for PDF certificates, extracts useful fields
(name, issuing org, date, hours, skills), infers a category/subcategory for
each one, and writes everything out to output/skills_<model>.csv — one CSV
per vision model, so you can compare accuracy before picking one.

Nothing leaves your computer. AI extraction runs through Ollama on
localhost — no cloud API, no account, no network required once the models
are pulled. Nothing in certs/, data/, or output/ is committed to git except
the CSVs themselves — see .gitignore.

Requires Ollama running locally (https://ollama.com) with vision models
pulled, e.g.:
    ollama pull qwen2.5vl
    ollama pull moondream

Usage:
    python parser/parser.py
    python parser/parser.py --certs-dir /path/to/certs
    python parser/parser.py --models qwen2.5vl,moondream
    python parser/parser.py --ollama-host http://localhost:11434
    python parser/parser.py --no-ai     # skip AI entirely, heuristics-only CSV
"""

import argparse
import base64
import csv
import io
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pdfplumber

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CERTS_DIR = REPO_ROOT / "certs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"
CATEGORIES_PATH = Path(__file__).resolve().parent / "categories.json"

DEFAULT_MODELS = ["qwen2.5vl", "moondream"]
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

CSV_FIELDS = [
    "cert_name",
    "issuing_org",
    "date_completed",
    "hours",
    "category",
    "subcategories",
    "skills",
    "source_file",
    "needs_review",
]

# ---------------------------------------------------------------------------
# PDF text extraction (fast heuristic first pass + hint text for the AI step)
# ---------------------------------------------------------------------------

GARBLED_MARKER = "(cid:"  # pdfplumber's literal placeholder for unmappable font glyphs


def extract_pages(pdf_path: Path) -> list:
    """
    Return a dict per page: {"text", "method", "image_b64"}. One page = one
    cert, so this handles both single-cert-per-file PDFs and multi-cert
    PDFs (several certs merged into one file, e.g. via Canva).

    Text extraction tries the normal text layer first, then OCR (as a text
    hint only — the vision models below read the actual image directly, so
    OCR quality matters less than it used to).
    """
    results = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            method = "text"
            if not text.strip() or GARBLED_MARKER in text:
                method = "none"
                if OCR_AVAILABLE:
                    try:
                        img_for_ocr = page.to_image(resolution=300).original
                        ocr_text = pytesseract.image_to_string(img_for_ocr)
                        if ocr_text.strip():
                            text = ocr_text
                            method = "ocr"
                    except Exception as e:
                        print(f"  [!] OCR failed: {e}", file=sys.stderr)

            # Always render an image — the vision models read this directly,
            # regardless of whether text extraction/OCR worked.
            image = page.to_image(resolution=200).original
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            results.append({"text": text, "method": method, "image_b64": image_b64})
    return results


# ---------------------------------------------------------------------------
# Heuristic field extraction (fast, free, offline — used as hints for the AI
# step, and as the only source of truth if --no-ai is passed)
# ---------------------------------------------------------------------------

HOURS_MINUTES_PATTERN = re.compile(r"(\d+)\s*hours?\s+(\d+)\s*minutes?\b", re.IGNORECASE)

HOURS_PATTERNS = [
    r"(\d+(?:\.\d+)?)\s*(?:hour|hr)s?\b",
    r"(\d+(?:\.\d+)?)\s*(?:CEU|PDU|CPE)s?\b",
]

MONTHS = (
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)

DATE_PATTERNS = [
    rf"\b{MONTHS}\s+\d{{1,2}},?\s+\d{{4}}\b",
    rf"\b\d{{1,2}}\s+{MONTHS},?\s+\d{{4}}\b",
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
]

SKILLS_LINE_PATTERN = re.compile(r"Skills?\s*(?:covered|gained|learned)?\s*[:\-]\s*(.+)", re.IGNORECASE)
SKILLS_HEADER_PATTERN = re.compile(r"^\s*(?:top\s+)?skills?\s*(?:covered|gained|learned)?\s*$", re.IGNORECASE)

KNOWN_ORGS = [
    "Coursera", "LinkedIn Learning", "Udemy", "CompTIA", "Cisco", "Microsoft",
    "Google", "Amazon Web Services", "AWS", "edX", "Pluralsight", "PMI",
    "ISC2", "(ISC)2", "SANS", "HubSpot", "IBM", "Meta", "Western Governors University",
    "WGU", "Udacity", "Codecademy", "DataCamp", "freeCodeCamp", "Skillsoft",
    "O'Reilly", "Khan Academy", "Salesforce Trailhead", "Adobe", "Oracle",
    "Red Hat", "VMware", "NVIDIA", "Databricks", "Snowflake",
]


def guess_hours(text: str) -> str:
    hm_match = HOURS_MINUTES_PATTERN.search(text)
    if hm_match:
        hours = int(hm_match.group(1)) + int(hm_match.group(2)) / 60
        return f"{hours:.2f}".rstrip("0").rstrip(".")
    for pattern in HOURS_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def guess_date(text: str) -> str:
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return ""


def guess_org(text: str) -> str:
    for org in KNOWN_ORGS:
        if org.lower() in text.lower():
            return org
    return ""


def guess_skills(text: str) -> str:
    match = SKILLS_LINE_PATTERN.search(text)
    if match:
        raw = match.group(1).split("\n")[0]
        parts = re.split(r"[,•|]\s*", raw)
        parts = [p.strip() for p in parts if p.strip()]
        if parts:
            return "; ".join(parts)

    lines = text.splitlines()
    for i, line in enumerate(lines):
        if SKILLS_HEADER_PATTERN.match(line):
            for next_line in lines[i + 1:]:
                next_line = next_line.strip()
                if next_line:
                    return next_line
    return ""


TITLE_BOILERPLATE_STARTS = (
    "certificate of", "certificate number", "this is to certify", "this is to recognize",
    "successfully completed", "recipients of", "awarded to", "presented to", "journey",
)

EXACT_BOILERPLATE_LINES = {
    "completion", "certificate", "certification", "recipient", "recipients",
    "award", "awarded", "date", "presented", "certify",
}


def is_boilerplate_line(line: str) -> bool:
    low = line.lower().strip().strip(".:,")
    return low.startswith(TITLE_BOILERPLATE_STARTS) or low in EXACT_BOILERPLATE_LINES


TITLE_MARKER_PATTERN = re.compile(
    r"(?:for\s+successfully\s+completing|for\s+completing|has\s+(?:successfully\s+)?completed"
    r"(?:\s+the\s+following)?|successfully\s+completed\s+the\s+following\s+course)\s*:?\s*$",
    re.IGNORECASE,
)

TITLE_END_MARKER_PATTERN = re.compile(r"course\s+completed\s+by|completed\s+by\b", re.IGNORECASE)


def guess_cert_name(text: str, filename: str) -> str:
    lines = [l.strip() for l in text.splitlines()]

    def extend_title(start_index: int) -> list:
        collected = []
        for line in lines[start_index:]:
            if not line:
                break
            if len(collected) > 0 and len(line) > 40:
                break
            if is_boilerplate_line(line):
                break
            if re.search(r"\d{4}", line) or re.search(r"\bhrs?\b|\bhours?\b", line, re.IGNORECASE):
                break
            if SKILLS_HEADER_PATTERN.match(line):
                break
            collected.append(line)
        return collected

    for i, line in enumerate(lines):
        if TITLE_MARKER_PATTERN.search(line):
            title_lines = extend_title(i + 1)
            if title_lines:
                return " ".join(title_lines)

    for i, line in enumerate(lines):
        if TITLE_END_MARKER_PATTERN.search(line):
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            collected = []
            while j >= 0:
                candidate = lines[j].strip()
                if not candidate:
                    break
                if is_boilerplate_line(candidate):
                    break
                if collected and len(candidate) > 40:
                    break
                collected.insert(0, candidate)
                j -= 1
            if collected:
                return " ".join(collected)
            break

    for i, line in enumerate(lines):
        if len(line) < 8 or is_boilerplate_line(line):
            continue
        title_lines = extend_title(i)
        if title_lines:
            return " ".join(title_lines)

    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


# ---------------------------------------------------------------------------
# Local AI extraction via Ollama (vision models read the cert image
# directly — no cloud, no API key, nothing leaves this machine)
# ---------------------------------------------------------------------------

def load_taxonomy() -> dict:
    with open(CATEGORIES_PATH, "r") as f:
        return json.load(f)["categories"]


def build_prompt(heuristic_guesses: dict, ocr_text: str, taxonomy: dict) -> str:
    taxonomy_str = json.dumps(taxonomy, indent=2)
    guesses_str = json.dumps(heuristic_guesses, indent=2)
    return f"""You are reading a professional certificate image for a personal skills tracker.

Look at the certificate image directly. A simple text-extraction pass already made first guesses for some fields (may be wrong or incomplete, especially if the cert uses a graphic/badge layout) — use these as hints, but trust the image over them if they disagree:
{guesses_str}

Extracted text, if any (may be empty or garbled):
---
{ocr_text[:1500]}
---

Allowed taxonomy for category/subcategories (pick from this list only):
{taxonomy_str}

Respond with ONLY a JSON object, no other text, in this exact shape:
{{
  "cert_name": "<the actual course/certificate title, not boilerplate like 'Certificate of Completion'>",
  "issuing_org": "<the issuing organization/platform, e.g. 'Coursera', 'Western Governors University'>",
  "date_completed": "<date in a clean 'Month D, YYYY' format>",
  "hours": "<numeric hours as a plain number if determinable, else empty string>",
  "category": "<one category from the taxonomy>",
  "subcategories": ["<one or more subcategories from that category's list>"],
  "skills": ["<specific skills shown or implied on the cert, freeform, 2-6 items>"],
  "low_confidence": true or false
}}"""


def extract_with_ollama(model: str, ollama_host: str, image_b64: str, heuristic_guesses: dict,
                         ocr_text: str, taxonomy: dict) -> dict:
    """
    Call a local Ollama vision model. Returns {} on any failure (connection
    refused, model not pulled, bad JSON back, etc.) so the caller can fall
    back to the heuristic guesses and flag the row for review.
    """
    prompt = build_prompt(heuristic_guesses, ocr_text, taxonomy)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
        "format": "json",
        "stream": False,
    }
    req = urllib.request.Request(
        f"{ollama_host.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body.get("message", {}).get("content", "")
        content = re.sub(r"^```json\s*|\s*```$", "", content.strip())
        return json.loads(content)
    except urllib.error.URLError as e:
        print(f"  [!] Can't reach Ollama at {ollama_host} ({e}). Is 'ollama serve' running?", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"  [!] Ollama ({model}) extraction failed: {e}", file=sys.stderr)
        return {}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_page(pdf_name: str, page_num: int, page_data: dict, model: str, ollama_host: str,
                  taxonomy: dict, use_ai: bool) -> dict:
    source_label = f"{pdf_name} (page {page_num})"
    text = page_data["text"]
    method = page_data["method"]

    heuristic = {
        "cert_name": guess_cert_name(text, pdf_name) if text.strip() else "",
        "issuing_org": guess_org(text),
        "date_completed": guess_date(text),
        "hours": guess_hours(text),
        "skills": guess_skills(text),
    }

    cert_name = heuristic["cert_name"] or f"{Path(pdf_name).stem} - page {page_num}"
    org = heuristic["issuing_org"]
    date = heuristic["date_completed"]
    hours = heuristic["hours"]
    skills = heuristic["skills"]
    category = ""
    subcategories = ""
    needs_review = "yes"

    if use_ai:
        result = extract_with_ollama(model, ollama_host, page_data["image_b64"], heuristic, text, taxonomy)
        if result:
            cert_name = result.get("cert_name") or cert_name
            org = result.get("issuing_org") or org
            date = result.get("date_completed") or date
            hours = str(result.get("hours") or hours)
            skills = "; ".join(result.get("skills", [])) or skills
            category = result.get("category", "")
            subcategories = "; ".join(result.get("subcategories", []))
            needs_review = "yes (AI flagged low confidence — verify)" if result.get("low_confidence") else ("no" if category else "yes")
        else:
            needs_review = "yes (AI call failed — see console output; heuristic values only)"

    if not text.strip():
        needs_review = f"{needs_review}; yes (no text layer found on this page)"

    return {
        "cert_name": cert_name,
        "issuing_org": org,
        "date_completed": date,
        "hours": hours,
        "category": category,
        "subcategories": subcategories,
        "skills": skills,
        "source_file": source_label,
        "needs_review": needs_review,
    }


def write_csv(rows: list, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    arg_parser = argparse.ArgumentParser(description="Parse certs into skills CSV(s) using local Ollama vision models.")
    arg_parser.add_argument("--certs-dir", default=str(DEFAULT_CERTS_DIR), help="Folder of cert PDFs")
    arg_parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to write CSV(s) into")
    arg_parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                             help="Comma-separated Ollama vision model names, one CSV per model")
    arg_parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST, help="Ollama server URL")
    arg_parser.add_argument("--no-ai", action="store_true",
                             help="Skip AI entirely; write a single heuristics-only output/skills.csv")
    args = arg_parser.parse_args()

    certs_dir = Path(args.certs_dir)
    out_dir = Path(args.out_dir)
    use_ai = not args.no_ai
    models = [m.strip() for m in args.models.split(",") if m.strip()] if use_ai else ["heuristic"]

    if not certs_dir.exists():
        print(f"Certs folder not found: {certs_dir}")
        sys.exit(1)

    pdf_files = sorted(certs_dir.glob("*.pdf"))
    taxonomy = load_taxonomy()

    if not pdf_files:
        print(f"No PDFs found in {certs_dir} — writing empty CSV(s) with headers only.")
        for model in models:
            name = "skills.csv" if model == "heuristic" else f"skills_{model.replace(':', '-')}.csv"
            write_csv([], out_dir / name)
        return

    # Extract each page once (text + image), reused across every model so
    # we don't re-render/re-OCR the PDF once per model.
    pages_by_pdf = {}
    for pdf_path in pdf_files:
        print(f"Reading {pdf_path.name}...")
        pages_by_pdf[pdf_path.name] = extract_pages(pdf_path)

    for model in models:
        print(f"\n=== Model: {model} ===")
        rows = []
        for pdf_path in pdf_files:
            for i, page_data in enumerate(pages_by_pdf[pdf_path.name], start=1):
                row = process_page(pdf_path.name, i, page_data, model, args.ollama_host, taxonomy, use_ai)
                rows.append(row)
                tag = f" [{page_data['method']}]" if page_data["method"] != "text" else ""
                print(f"  {pdf_path.name} page {i}{tag}: {row['cert_name']}")

        out_name = "skills.csv" if model == "heuristic" else f"skills_{model.replace(':', '-')}.csv"
        out_path = out_dir / out_name
        write_csv(rows, out_path)

        review_count = sum(1 for r in rows if r["needs_review"].startswith("yes"))
        print(f"Wrote {len(rows)} row(s) to {out_path}")
        if review_count:
            print(f"{review_count} row(s) need manual review — check the needs_review column.")


if __name__ == "__main__":
    main()
