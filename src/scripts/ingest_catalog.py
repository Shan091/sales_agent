# file: src/scripts/ingest_catalog.py
"""
Offline batch ingestion CLI for the RAG corpus.

Handles both kinds of input in one pass over the directory:
1. `.md` files are read directly — this is how the hand-written catalogue in `docs/catalog/` is
   ingested, and it is the path the setup guide uses.
2. `.pdf` files are parsed into layout-aware Markdown with IBM Docling first (an optional
   dependency; the script says so and skips rather than crashing if it is absent).

Both are then cleaned by the DocumentCleaner and chunked into PostgreSQL by the ingestion pipeline.

Usage:
    python -m src.scripts.ingest_catalog --input-dir ./docs/catalog/switches \
        --doc-type TECHNICAL_SPEC --category switches

This script is NOT part of the real-time API path. It runs offline as a batch job.
"""
import asyncio
import argparse
import logging
import sys
from pathlib import Path

from src.scripts.cleaner import clean_markdown
from src.rag.ingestion import ingest_markdown_document
from src.core.database import async_session_maker, close_db_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


async def ingest_pdf(pdf_path: Path, doc_type: str, category: str | None = None):
    """
    Processes a single PDF file through the Docling -> Cleaner -> Ingestion pipeline.

    Args:
        pdf_path: Path to the PDF file.
        doc_type: Document type metadata (TECHNICAL_SPEC, INSTALLATION_GUIDE, etc.).
        category: Optional product category metadata (switches, security, etc.).
    """
    logger.info(f"Processing: {pdf_path.name}")

    # Step 1: Parse PDF to Markdown using Docling
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        raw_markdown = result.document.export_to_markdown()
    except ImportError:
        logger.error(
            "IBM Docling is not installed. Install it with: pip install docling\n"
            "Alternatively, place pre-converted .md files in the input directory."
        )
        return
    except Exception as e:
        logger.error(f"Docling failed to parse {pdf_path.name}: {e}")
        return

    # Step 2: Clean the Markdown
    cleaned_markdown = clean_markdown(raw_markdown)

    if not cleaned_markdown.strip():
        logger.warning(f"Cleaned Markdown is empty for {pdf_path.name}. Skipping.")
        return

    # Step 3: Build metadata
    metadata = {"doc_type": doc_type, "source_file": pdf_path.name}
    if category:
        metadata["category"] = category

    # Step 4: Ingest into PostgreSQL
    async with async_session_maker() as session:
        total = await ingest_markdown_document(session, cleaned_markdown, metadata)
        logger.info(f"✅ Ingested {total} chunks from {pdf_path.name}")


async def ingest_markdown_file(md_path: Path, doc_type: str, category: str | None = None):
    """
    Processes a pre-converted Markdown file (skip Docling).
    Useful for manually curated documents or when Docling has already been run.
    """
    logger.info(f"Processing Markdown: {md_path.name}")

    raw_markdown = md_path.read_text(encoding="utf-8")
    cleaned_markdown = clean_markdown(raw_markdown)

    if not cleaned_markdown.strip():
        logger.warning(f"Cleaned Markdown is empty for {md_path.name}. Skipping.")
        return

    metadata = {"doc_type": doc_type, "source_file": md_path.name}
    if category:
        metadata["category"] = category

    async with async_session_maker() as session:
        total = await ingest_markdown_document(session, cleaned_markdown, metadata)
        logger.info(f"✅ Ingested {total} chunks from {md_path.name}")


async def main(input_dir: str, doc_type: str, category: str | None):
    """Main entry point: scans a directory for PDFs and Markdown files."""
    input_path = Path(input_dir)

    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Collect all PDF and MD files
    pdf_files = list(input_path.glob("*.pdf"))
    md_files = list(input_path.glob("*.md"))

    if not pdf_files and not md_files:
        logger.warning(f"No PDF or Markdown files found in {input_dir}")
        sys.exit(0)

    logger.info(f"Found {len(pdf_files)} PDFs and {len(md_files)} Markdown files.")

    # Process PDFs (requires Docling)
    for pdf in pdf_files:
        await ingest_pdf(pdf, doc_type, category)

    # Process Markdown files (pre-converted, skip Docling)
    for md in md_files:
        await ingest_markdown_file(md, doc_type, category)

    # Cleanup
    await close_db_engine()
    logger.info("🏁 Ingestion pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Otohom RAG Document Ingestion CLI")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing PDF or Markdown files to ingest."
    )
    parser.add_argument(
        "--doc-type",
        type=str,
        default="TECHNICAL_SPEC",
        choices=["TECHNICAL_SPEC", "INSTALLATION_GUIDE", "TROUBLESHOOTING", "PRODUCT_CATALOG"],
        help="Document type metadata for all files in this batch."
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Optional product category (e.g., switches, security, lighting)."
    )

    args = parser.parse_args()
    asyncio.run(main(args.input_dir, args.doc_type, args.category))
