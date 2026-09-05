# file: src/scripts/cleaner.py
"""
Phase 3: Pre-Ingestion Document Cleaner.

Strips non-technical marketing boilerplate from Otohom PDF-parsed Markdown
before it enters the chunking and embedding pipeline.

Run this BEFORE Docling or after Docling's Markdown output to filter noise.
"""
import re
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
#  Marketing Fluff Patterns
# ═══════════════════════════════════════════════
# These regex patterns match common Otohom brochure boilerplate that pollutes
# the vector space with non-technical semantic noise.

# Whole-line boilerplate — these lines are pure marketing and are dropped entirely.
LINE_FLUFF_PATTERNS = [
    # Brand taglines and mission statements
    r"(?i)digitali[sz]e your physical world",
    r"(?i)our story is worthy of reading",
    r"(?i)elevated\s+living",
    r"(?i)experience\s+the\s+future",
    r"(?i)smart\s+living\s+redefined",
    r"(?i)transform\s+your\s+home\s+into",

    # Contact / footer boilerplate
    r"(?i)for\s+more\s+information\s*,?\s*visit",
    r"(?i)follow\s+us\s+on\s+(social\s+media|instagram|facebook|twitter|linkedin)",
    r"(?i)©\s*\d{4}\s*otohom",
    r"(?i)all\s+rights\s+reserved",
    r"(?i)terms\s+and\s+conditions\s+apply",

    # Marketing fluff phrases
    r"(?i)crafted\s+with\s+(love|passion|precision)",
    r"(?i)award[- ]winning\s+design",
    r"(?i)trusted\s+by\s+thousands",
    r"(?i)join\s+the\s+revolution",

    # Generic promotional lines
    r"(?i)limited\s+time\s+offer",
    r"(?i)call\s+us\s+(today|now)\s+at",
    r"(?i)book\s+a\s+free\s+(consultation|demo|trial)",
]

# Inline PII/contact noise — REDACTED in place (never whole-line deleted) so any
# technical content that shares the line survives.
INLINE_NOISE_PATTERNS = [
    r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}",  # Phone numbers
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # Emails
]

# Compile patterns for performance
COMPILED_LINE_FLUFF = [re.compile(p) for p in LINE_FLUFF_PATTERNS]
COMPILED_INLINE_NOISE = [re.compile(p) for p in INLINE_NOISE_PATTERNS]


def _is_structural(stripped: str) -> bool:
    """
    Markdown lines that carry technical/structural payload and must NEVER be dropped or
    redacted: headers (`#...`), table rows (contain a pipe), and list items (spec
    bullets). The phone/number regex routinely matches legitimate spec sequences
    (e.g. '100-240-50-60'), so exempting these lines prevents the cleaner from
    destroying tables and specification rows.
    """
    return (
        stripped.startswith("#")
        or "|" in stripped
        or bool(re.match(r"[-*+]\s", stripped))
    )


def clean_markdown(raw_markdown: str) -> str:
    """
    Strips marketing boilerplate from Markdown text.

    Args:
        raw_markdown: The raw Markdown output from Docling or manual PDF-to-MD conversion.

    Returns:
        Cleaned Markdown with marketing fluff lines removed.
    """
    lines = raw_markdown.split("\n")
    cleaned_lines = []
    removed_count = 0

    for line in lines:
        stripped = line.strip()

        # Preserve empty lines (Markdown structure)
        if not stripped:
            cleaned_lines.append(line)
            continue

        # Structural/technical lines (headers, tables, spec bullets) are kept verbatim —
        # never dropped or redacted, so tables and specification rows survive intact.
        if _is_structural(stripped):
            cleaned_lines.append(line)
            continue

        # Prose line: first redact inline phone/email noise (keeping the rest of the
        # line)...
        redacted = line
        for pattern in COMPILED_INLINE_NOISE:
            redacted = pattern.sub("", redacted)

        # ...then drop the whole line only if it is pure marketing boilerplate.
        if any(pattern.search(redacted) for pattern in COMPILED_LINE_FLUFF):
            removed_count += 1
            continue

        cleaned_lines.append(redacted)

    # Remove excessive blank lines (3+ consecutive -> 2)
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines))

    logger.info(f"DocumentCleaner: Removed {removed_count} marketing/boilerplate lines.")
    return result.strip()
