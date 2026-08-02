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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
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

def extract_text(pdf_path: Path) -> str:
    """Pull all text out of a PDF, page by page."""
    text_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_chunks.append(page_text)
    return "\n".join(text_chunks)


# ---------------------------------------------------------------------------
# Heuristic field extraction
# ---------------------------------------------------------------------------

HOURS_PATTERNS = [
    r"(\d+(?:\.\d+)?)\s*(?:hour|hr)s?\b",
    r"(\d+(?:\.\d+)?)\s*(?:CEU|PDU|CPE)s?\b",
]

DATE_PATTERNS = [
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
]

SKILLS_LINE_PATTERN = re.compile(r"Skills?\s*(?:covered|gained|learned)?\s*[:\-]\s*(.+)", re.IGNORECASE)

# Known issuing orgs to look for directly in text (extend as you go)
KNOWN_ORGS = [
    "Coursera", "LinkedIn Learning", "Udemy", "CompTIA", "Cisco", "Microsoft",
    "Google", "Amazon Web Services", "AWS", "edX", "Pluralsight", "PMI",
    "ISC2", "(ISC)2", "SANS", "HubSpot", "IBM", "Meta",
]


def guess_hours(text: str) -> str:
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
        raw = match.group(1)
        # Cut off at the next likely section break / newline
        raw = raw.split("\n")[0]
        # Normalize separators (commas, bullets, pipes) to semicolons
        parts = re.split(r"[,•|]\s*", raw)
        parts = [p.strip() for p in parts if p.strip()]
        return "; ".join(parts)
    return ""


def guess_cert_name(text: str, filename: str) -> str:
    """
    Certs rarely label 'name' explicitly. Best-effort: use the first
    substantial line of text (skip very short lines like logos/headers),
    falling back to the filename.
    """
    for line in text.splitlines():
        line = line.strip()
        if len(line) >= 8 and not line.lower().startswith(("certificate of", "this is to certify")):
            return line
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

Allowed taxonomy (pick the category and subcategories from this list only):
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

def process_cert(pdf_path: Path, taxonomy: dict, use_llm: bool) -> dict:
    text = extract_text(pdf_path)

    cert_name = guess_cert_name(text, pdf_path.name)
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

    return {
        "cert_name": cert_name,
        "issuing_org": org,
        "date_completed": date,
        "hours": hours,
        "category": category,
        "subcategories": subcategories,
        "skills": skills,
        "source_file": pdf_path.name,
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
        row = process_cert(pdf_path, taxonomy, use_llm)
        rows.append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    review_count = sum(1 for r in rows if r["needs_review"] == "yes")
    print(f"\nDone. Wrote {len(rows)} certs to {out_path}")
    if review_count:
        print(f"{review_count} cert(s) need manual review (no category assigned).")


if __name__ == "__main__":
    main()
