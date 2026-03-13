"""
Mobitel HR Assistant - Holiday Parser
Automatically extracts holiday dates from the Holiday Planner PDF
using the LLM, and saves them to holidays.json.

Usage:
    python holiday_parser.py
    
This script is also called automatically during document ingestion.
HR staff only need to replace the Holiday Planner PDF in data/Holiday/
and re-run ingest — no code changes needed.
"""

import os
import sys
import json
import re
import glob
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pypdf import PdfReader
from config import setup_logging, DATA_PATH, BASE_DIR

logger = setup_logging("hr_assistant.holiday_parser")

# Output path for the structured holiday data
HOLIDAYS_JSON_PATH = BASE_DIR / "holidays.json"


def find_holiday_pdf() -> str:
    """
    Auto-detect the Holiday Planner PDF in the data directory.
    Looks for files matching common holiday planner naming patterns.
    
    Returns:
        Path to the holiday PDF, or None if not found.
    """
    holiday_dir = os.path.join(str(DATA_PATH), "Holiday")
    
    if not os.path.exists(holiday_dir):
        logger.warning(f"Holiday directory not found: {holiday_dir}")
        return None
    
    # Search for PDFs in the Holiday folder
    patterns = ["*.pdf", "*.PDF"]
    for pattern in patterns:
        matches = glob.glob(os.path.join(holiday_dir, pattern))
        if matches:
            # If multiple PDFs, pick the most recent
            matches.sort(key=os.path.getmtime, reverse=True)
            logger.info(f"Found holiday PDF: {matches[0]}")
            return matches[0]
    
    logger.warning(f"No PDF files found in {holiday_dir}")
    return None


def extract_year_from_pdf(pdf_path: str, text: str) -> int:
    """
    Extract the year from the PDF filename or content.
    
    Args:
        pdf_path: Path to the PDF file
        text: Extracted text from the PDF
        
    Returns:
        Year as integer
    """
    filename = os.path.basename(pdf_path)
    
    # Try filename first (e.g., "HOLIDAY PLANNER - 2026.pdf")
    year_match = re.search(r'20\d{2}', filename)
    if year_match:
        return int(year_match.group())
    
    # Try content
    year_match = re.search(r'(?:YEAR|year)\s*(?:of\s*)?(\d{4})', text)
    if year_match:
        return int(year_match.group(1))
    
    # Fallback: look for any 4-digit year in content
    year_match = re.search(r'20\d{2}', text)
    if year_match:
        return int(year_match.group())
    
    # Default to current year
    return datetime.now().year


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract raw text from the Holiday Planner PDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text content
    """
    reader = PdfReader(pdf_path)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    
    full_text = "\n".join(text_parts)
    logger.info(f"Extracted {len(full_text)} characters from PDF ({len(reader.pages)} pages)")
    return full_text


def extract_holidays_with_llm(pdf_text: str, year: int) -> list:
    """
    Use the Ollama LLM to extract structured holiday data from the PDF text.
    
    Args:
        pdf_text: Raw text extracted from the PDF
        year: The year for the holidays
        
    Returns:
        List of {"date": "YYYY-MM-DD", "name": "Holiday Name"} dicts
    """
    from ollama import chat
    from config import LLM_MODEL
    
    prompt = f"""Extract ALL public holidays from the following text. This is a company holiday planner for the year {year}.

RULES:
1. Extract EVERY holiday mentioned, including Poya days, religious holidays, and national holidays.
2. Return ONLY a JSON array — no other text, no explanation, no markdown.
3. Each entry must have "date" (format: YYYY-MM-DD) and "name" (the holiday name).
4. Use the year {year} for all dates.
5. If a date says something like "15th January", convert it to "{year}-01-15".
6. Do NOT invent or guess any holidays. Only extract what is explicitly in the text.
7. Make sure the month is correct — read carefully.

TEXT FROM PDF:
---
{pdf_text}
---

Return ONLY a valid JSON array like:
[{{"date": "{year}-01-15", "name": "Tamil Thai Pongal Day"}}, ...]
"""

    logger.info(f"Sending extraction request to LLM ({LLM_MODEL})...")
    
    response = chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0}  # Zero temperature for factual extraction
    )
    
    response_text = response.message.content.strip()
    logger.debug(f"LLM response: {response_text[:500]}...")
    
    # Parse the JSON from the response
    # Handle cases where LLM wraps in markdown code blocks
    json_text = response_text
    if "```" in json_text:
        # Extract JSON from code block
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', json_text, re.DOTALL)
        if match:
            json_text = match.group(1).strip()
    
    try:
        holidays = json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"Response was: {response_text}")
        return []
    
    if not isinstance(holidays, list):
        logger.error(f"Expected a list, got {type(holidays)}")
        return []
    
    logger.info(f"LLM extracted {len(holidays)} holidays")
    return holidays


def validate_holidays(holidays: list, year: int) -> list:
    """
    Validate and clean the extracted holiday data.
    
    Args:
        holidays: Raw list of holiday dicts from LLM
        year: Expected year
        
    Returns:
        Cleaned and validated list
    """
    validated = []
    seen_dates = set()
    
    for entry in holidays:
        if not isinstance(entry, dict):
            logger.warning(f"Skipping non-dict entry: {entry}")
            continue
            
        date_str = entry.get("date", "").strip()
        name = entry.get("name", "").strip()
        
        if not date_str or not name:
            logger.warning(f"Skipping entry with missing date or name: {entry}")
            continue
        
        # Validate date format
        try:
            parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
            if parsed_date.year != year:
                logger.warning(f"Wrong year for {name}: {date_str}, fixing to {year}")
                date_str = f"{year}-{parsed_date.month:02d}-{parsed_date.day:02d}"
        except ValueError:
            logger.warning(f"Invalid date format: {date_str} for {name}")
            continue
        
        # Skip duplicates
        if date_str in seen_dates:
            logger.warning(f"Duplicate date {date_str}, skipping: {name}")
            continue
        
        seen_dates.add(date_str)
        validated.append({"date": date_str, "name": name})
    
    # Sort by date
    validated.sort(key=lambda x: x["date"])
    
    logger.info(f"Validated {len(validated)} holidays (from {len(holidays)} raw entries)")
    return validated


def save_holidays_json(holidays: list, year: int, pdf_path: str) -> str:
    """
    Save the holiday data to holidays.json.
    
    Args:
        holidays: Validated list of holiday dicts
        year: The year
        pdf_path: Source PDF path
        
    Returns:
        Path to the saved JSON file
    """
    data = {
        "year": year,
        "source_pdf": pdf_path,
        "extracted_at": datetime.now().isoformat(),
        "holiday_count": len(holidays),
        "holidays": holidays
    }
    
    output_path = str(HOLIDAYS_JSON_PATH)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Saved {len(holidays)} holidays to {output_path}")
    return output_path


def parse_holiday_pdf(pdf_path: str = None) -> bool:
    """
    Main function: Parse the Holiday Planner PDF and generate holidays.json.
    
    Args:
        pdf_path: Optional explicit path. If None, auto-detects from data/Holiday/.
        
    Returns:
        True if successful, False otherwise.
    """
    logger.info("=" * 60)
    logger.info("HOLIDAY PARSER - Starting")
    logger.info("=" * 60)
    
    # Step 1: Find the PDF
    if pdf_path is None:
        pdf_path = find_holiday_pdf()
    
    if not pdf_path or not os.path.exists(pdf_path):
        logger.error("No Holiday Planner PDF found!")
        return False
    
    logger.info(f"Processing: {pdf_path}")
    
    # Step 2: Extract text
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text or len(pdf_text) < 50:
        logger.error("PDF text extraction failed or too short")
        return False
    
    # Step 3: Detect year
    year = extract_year_from_pdf(pdf_path, pdf_text)
    logger.info(f"Detected year: {year}")
    
    # Step 4: Extract holidays using LLM
    raw_holidays = extract_holidays_with_llm(pdf_text, year)
    if not raw_holidays:
        logger.error("LLM extraction returned no holidays")
        return False
    
    # Step 5: Validate
    validated = validate_holidays(raw_holidays, year)
    if not validated:
        logger.error("No valid holidays after validation")
        return False
    
    # Step 6: Save
    output_path = save_holidays_json(validated, year, pdf_path)
    
    logger.info("=" * 60)
    logger.info(f"HOLIDAY PARSER - Complete! {len(validated)} holidays saved")
    logger.info("=" * 60)
    
    # Print summary
    print(f"\n{'=' * 50}")
    print(f"  Holiday Planner Parsed Successfully!")
    print(f"  Year: {year}")
    print(f"  Holidays found: {len(validated)}")
    print(f"  Saved to: {output_path}")
    print(f"{'=' * 50}")
    print(f"\nHolidays:")
    for h in validated:
        d = datetime.strptime(h['date'], '%Y-%m-%d')
        print(f"  {d.strftime('%b %d')} ({d.strftime('%A')}) — {h['name']}")
    
    return True


if __name__ == "__main__":
    success = parse_holiday_pdf()
    if not success:
        print("\n❌ Holiday parsing failed! Check logs for details.")
        sys.exit(1)
