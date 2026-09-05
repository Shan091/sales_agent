"""
Edge-case tests for the pre-ingestion cleaner (Phase 3).

Regression focus: the phone/email regexes must NOT delete whole lines of technical
content or corrupt markdown tables/spec bullets.
"""
from src.scripts.cleaner import clean_markdown


def test_email_in_prose_redacted_but_spec_text_kept():
    dirty = "Contact support@otohom.com for the 16A 250V Grande 4SW load rating"
    cleaned = clean_markdown(dirty)
    assert "support@otohom.com" not in cleaned
    assert "16A 250V Grande 4SW load rating" in cleaned


def test_table_row_with_numeric_sequence_preserved():
    dirty = "| Voltage | 100-240-50-60 |"
    cleaned = clean_markdown(dirty)
    assert "100-240-50-60" in cleaned
    assert "| Voltage |" in cleaned


def test_spec_bullet_with_hyphenated_numbers_preserved():
    dirty = "- Rated: 100-240V AC, 50-60Hz, 3000W"
    cleaned = clean_markdown(dirty)
    assert "100-240V AC, 50-60Hz, 3000W" in cleaned


def test_marketing_line_still_removed():
    dirty = "Digitalize your physical world\n- Max Load: 800W"
    cleaned = clean_markdown(dirty)
    assert "Digitalize your physical world" not in cleaned
    assert "Max Load: 800W" in cleaned


def test_prose_phone_number_redacted():
    dirty = "Call us at +91-9876543210 for support."
    cleaned = clean_markdown(dirty)
    assert "9876543210" not in cleaned
