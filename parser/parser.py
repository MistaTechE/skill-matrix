#parser.py

#!/usr/bin/env python3
"""
parser.py — Phase 1 of skill-matrix

Scans the local certs/ folder for PDF certificates, extracts useful fields
(name, issuing org, date, hours, skills), infers a category/subcategory for
each one, and writes everything out to output/skills.csv.

Nothing in certs/, data/, or output/ is ever committed to git — see .gitignore.

Usage:
    python parser/parser.py
    python parser/parser.py --certs-dir /path/to/certs --out output/skills.csv
    python parser/parser.py --no-llm     # skip AI categorization (fast, free, leaves blanks)
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import pdfplumber
from dotenv import load_dotenv

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

# Load ANTHROPIC_API_KEY (and anything else) from a .env file in the repo root
load_dotenv(REPO_ROOT / ".env")
DEFAULT_CERTS_DIR = REPO_ROOT / "certs"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "output" / "skills.csv"
CATEGORIES_PATH = Path(__file__).resolve().parent / "categories.json"

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
# PDF text extraction
# ---------------------------------------------------------------------------

GARBLED_MARKER = "(cid:"  # pdfplumber's literal placeholder for unmappable font glyphs


def extract_pages(pdf_path: Path) -> list:
    """
    Return (text, method) for each page. One page = one cert: this handles
    both single-cert-per-file PDFs (1 page) and multi-cert PDFs where
    several certs were merged into one file (e.g. via Canva).

    Tries normal text extraction first. Falls back to OCR when a page has
    no text at all (scanned/flattened image) or garbled text (a broken
    font encoding — some LinkedIn Learning exports do this; pdfplumber
    literally emits "(cid:14)"-style placeholders when it can't map a
    glyph, which is how we detect it).
    """
    results = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip() and GARBLED_MARKER not in text:
                results.append((text, "text"))
                continue

            if not OCR_AVAILABLE:
                results.append(("", "none"))
                continue

            try:
                image = page.to_image(resolution=300).original
                ocr_text = pytesseract.image_to_string(image)
                if ocr_text.strip():
                    results.append((ocr_text, "ocr"))
                else:
                    results.append(("", "none"))
            except Exception as e:
                print(f"  [!] OCR failed: {e}", file=sys.stderr)
                results.append(("", "none"))
    return results


# ---------------------------------------------------------------------------
# Heuristic field extraction
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
    rf"\b{MONTHS}\s+\d{{1,2}},?\s+\d{{4}}\b",       # "October 27, 2023" / "Nov 01, 2023"
    rf"\b\d{{1,2}}\s+{MONTHS},?\s+\d{{4}}\b",        # "27 October, 2023"
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
]

# Matches an explicit "Skills:" line with content on the same line, OR a
# "Skills covered" header (no colon, no content on that line — the actual
# skill chips are on the next line, common in Coursera/LinkedIn exports).
SKILLS_LINE_PATTERN = re.compile(r"Skills?\s*(?:covered|gained|learned)?\s*[:\-]\s*(.+)", re.IGNORECASE)
SKILLS_HEADER_PATTERN = re.compile(r"^\s*(?:top\s+)?skills?\s*(?:covered|gained|learned)?\s*$", re.IGNORECASE)

# Known issuing orgs to look for directly in text (extend as you go)
KNOWN_ORGS = [
    "Coursera", "LinkedIn Learning", "Udemy", "CompTIA", "Cisco", "Microsoft",
    "Google", "Amazon Web Services", "AWS", "edX", "Pluralsight", "PMI",
    "ISC2", "(ISC)2", "SANS", "HubSpot", "IBM", "Meta",
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
    # Case 1: "Skills: Python, SQL, ..." — content on the same line.
    match = SKILLS_LINE_PATTERN.search(text)
    if match:
        raw = match.group(1).split("\n")[0]
        parts = re.split(r"[,•|]\s*", raw)
        parts = [p.strip() for p in parts if p.strip()]
        if parts:
            return "; ".join(parts)

    # Case 2: a "Skills covered" header line with no colon — the chip list
    # is the next non-empty line (common on Coursera/LinkedIn Learning
    # exports). Chips usually have no reliable delimiter once flattened to
    # plain text, so this returns the raw blob rather than guessing splits —
    # the AI categorization step (or you, by eye) can parse it accurately.
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

# Phrases that typically appear right before the real course/cert title
# (e.g. "for successfully completing" -> next line is the title).
TITLE_MARKER_PATTERN = re.compile(
    r"(?:for\s+successfully\s+completing|for\s+completing|has\s+(?:successfully\s+)?completed"
    r"(?:\s+the\s+following)?|successfully\s+completed\s+the\s+following\s+course)\s*:?\s*$",
    re.IGNORECASE,
)

# Phrases that appear AFTER the title on some layouts (e.g. LinkedIn Learning's
# "Course completed by ___") — walk backward from this to find the title block.
TITLE_END_MARKER_PATTERN = re.compile(r"course\s+completed\s+by|completed\s+by\b", re.IGNORECASE)


def guess_cert_name(text: str, filename: str) -> str:
    """
    Certs rarely label 'name' explicitly. Strategy:
    1. Look for a marker phrase like "for successfully completing" and use
       the line(s) right after it as the title — this is the most reliable
       signal when present.
    2. Otherwise, fall back to the first substantial, non-boilerplate line,
       extending it with short follow-up lines (titles often wrap, e.g.
       "Programming Foundations:" + "Databases").
    3. Falls back to the filename if nothing usable is found.
    """
    lines = [l.strip() for l in text.splitlines()]

    def extend_title(start_index: int) -> list:
        collected = []
        for line in lines[start_index:]:
            if not line:
                break
            if len(collected) > 0 and len(line) > 40:
                break
            if line.lower().startswith(TITLE_BOILERPLATE_STARTS):
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

    # Title-before-marker layouts (e.g. "Generative AI Imaging..." / blank /
    # "Course completed by ___"): walk backward from the marker, collecting
    # the nearest contiguous non-blank block. The blank line naturally
    # separates the title from the logo/header above it, so this works
    # without needing to know every provider's name.
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
                if candidate.lower().startswith(TITLE_BOILERPLATE_STARTS):
                    break
                if collected and len(candidate) > 40:
                    break
                collected.insert(0, candidate)
                j -= 1
            if collected:
                return " ".join(collected)
            break

    for i, line in enumerate(lines):
        if len(line) < 8 or line.lower().startswith(TITLE_BOILERPLATE_STARTS):
            continue
        title_lines = extend_title(i)
        if title_lines:
            return " ".join(title_lines)

    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


# ---------------------------------------------------------------------------
# LLM categorization (Phase 1 fallback for certs without explicit skill tags)
# ---------------------------------------------------------------------------

def load_taxonomy() -> dict:
    with open(CATEGORIES_PATH, "r") as f:
        return json.load(f)["categories"]


def categorize_with_llm(cert_name: str, raw_text: str, taxonomy: dict) -> dict:
    """
    Ask Claude to pick a category + subcategories + skill list from the
    fixed taxonomy, so results stay consistent across all your certs.
    Requires ANTHROPIC_API_KEY to be set in the environment.
    Returns {} on any failure (caller should mark the row needs_review).
    """
    try:
        import anthropic
    except ImportError:
        return {}

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {}

    client = anthropic.Anthropic(api_key=api_key)

    taxonomy_str = json.dumps(taxonomy, indent=2)
    prompt = f"""You are categorizing a professional certificate for a personal skills tracker.

Allowed taxonomy (you MUST pick the category and subcategories from this list only):
{taxonomy_str}

Certificate name: {cert_name}

Certificate text (may be messy OCR/PDF extraction):
---
{raw_text[:3000]}
---

Respond with ONLY a JSON object, no other text, in this exact shape:
{{
  "category": "<one category from the taxonomy>",
  "subcategories": ["<one or more subcategories from that category's list>"],
  "skills": ["<specific skills inferred from the cert, freeform, 2-6 items>"]
}}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r"^```json\s*|\s*```$", "", raw.strip())
        parsed = json.loads(raw)
        return parsed
    except Exception as e:
        print(f"  [!] LLM categorization failed: {e}", file=sys.stderr)
        return {}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_page(pdf_name: str, page_num: int, text: str, extraction_method: str, taxonomy: dict, use_llm: bool) -> dict:
    source_label = f"{pdf_name} (page {page_num})"

    # No extractable text at all, even after an OCR attempt.
    if not text.strip():
        return {
            "cert_name": f"{Path(pdf_name).stem} - page {page_num}",
            "issuing_org": "",
            "date_completed": "",
            "hours": "",
            "category": "",
            "subcategories": "",
            "skills": "",
            "source_file": source_label,
            "needs_review": "yes (no extractable text — install/check tesseract-ocr for OCR fallback)",
        }

    cert_name = guess_cert_name(text, pdf_name)
    org = guess_org(text)
    date = guess_date(text)
    hours = guess_hours(text)
    skills = guess_skills(text)

    category = ""
    subcategories = ""
    needs_review = "yes"

    if use_llm:
        result = categorize_with_llm(cert_name, text, taxonomy)
        if result:
            category = result.get("category", "")
            subcategories = "; ".join(result.get("subcategories", []))
            if not skills:
                skills = "; ".join(result.get("skills", []))
            needs_review = "no" if category else "yes"

    if extraction_method == "ocr":
        # OCR is usually accurate but not perfect — flag so you spot-check
        # dates/hours especially, since a misread digit fails silently.
        ocr_note = "yes (OCR used — verify fields, esp. dates/numbers)"
        needs_review = ocr_note if needs_review == "no" else f"{needs_review}; {ocr_note}"

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


def main():
    parser = argparse.ArgumentParser(description="Parse certs into a skills CSV.")
    parser.add_argument("--certs-dir", default=str(DEFAULT_CERTS_DIR), help="Folder of cert PDFs")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_CSV), help="Output CSV path")
    parser.add_argument("--no-llm", action="store_true", help="Skip AI categorization")
    args = parser.parse_args()

    certs_dir = Path(args.certs_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not certs_dir.exists():
        print(f"Certs folder not found: {certs_dir}")
        sys.exit(1)

    pdf_files = sorted(certs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {certs_dir}")
        sys.exit(0)

    taxonomy = load_taxonomy()
    use_llm = not args.no_llm

    rows = []
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}...")
        pages = extract_pages(pdf_path)
        for i, (page_text, method) in enumerate(pages, start=1):
            row = process_page(pdf_path.name, i, page_text, method, taxonomy, use_llm)
            rows.append(row)
            tag = f" [{method}]" if method != "text" else ""
            print(f"  page {i}{tag}: {row['cert_name']}")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    review_count = sum(1 for r in rows if r["needs_review"].startswith("yes"))
    print(f"\nDone. Wrote {len(rows)} cert row(s) to {out_path}")
    if review_count:
        print(f"{review_count} row(s) need manual review — check the needs_review column.")


if __name__ == "__main__":
    main()
