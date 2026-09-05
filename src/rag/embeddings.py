# file: src/rag/embeddings.py
"""
Phase 3: RAG Chunking & Embedding Pipeline.

Architecture:
1. MarkdownHeaderTextSplitter: Primary splitter that preserves product hierarchy
   (e.g., all specs under "Grande Series" stay grouped).
2. RecursiveCharacterTextSplitter: Safety-net fallback for oversized sections
   with strict token limits and forced overlap.
3. Parent-Child Chunking: Parent chunks (1024 tokens) store full context;
   child chunks (128-256 tokens) are embedded for precise vector matching.
"""
import hashlib
import logging
from typing import List, Tuple
from dataclasses import dataclass, field

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ProcessedChunk:
    """Represents a chunk ready for database insertion."""
    content: str
    is_parent: bool
    metadata: dict = field(default_factory=dict)
    source_hash: str = ""
    # For child chunks, this will be set after parent insertion
    parent_content_ref: str = ""  # Temporary: the parent's content to link after DB insert


# ═══════════════════════════════════════════════
#  Markdown-Aware Splitting
# ═══════════════════════════════════════════════

# Headers to split on — preserves H1/H2/H3 structural hierarchy
HEADERS_TO_SPLIT_ON = [
    ("#", "product_line"),
    ("##", "product_series"),
    ("###", "specification_section"),
]

# RAG_*_CHUNK_SIZE / _OVERLAP in settings are expressed in TOKENS, but
# RecursiveCharacterTextSplitter (length_function=len) and the parent slice below
# count CHARACTERS. bge-m3 averages ~4 chars/token on Latin-script spec text, so we
# convert token budgets to character budgets here. (Exact token counting would need
# the bge-m3 tokenizer, which we avoid at split-time to keep offline ingestion and
# the chunking unit tests dependency-free.)
CHARS_PER_TOKEN = 4

# A section this short is not worth being a parent on its own. Measured on the live corpus:
# docs/catalog/switches/smart_switches.md has ten product sections between 53 and 105 characters
# ("## 4 SW (4-gang smart switch)" plus a single bullet), which produced ten near-identical tiny
# parents. Two things went wrong with that. Dense retrieval cannot tell them apart — cosine over
# "## 2 SW (2-gang smart switch) / Power: 5A (2 gangs)" and its neighbour is close to a coin toss —
# so "what is the max load on the 4SW?" returned the 1 SW and 2 SW parents and the answer was in
# neither. And a 65-character parent carries no context even when it IS the right one: every shared
# spec (voltage, wiring, protocol, temperature range) lives in the document intro, so the model got
# a bare number with nothing around it.
#
# Merging consecutive short sections into one parent fixes both without touching the embeddings:
# whichever child wins, the parent now holds its neighbours too. Children are still split PER
# SECTION (see below), so the precise per-product embedding target is unchanged — only the context
# handed to the model grows.
RAG_MIN_PARENT_CHARS = 400


def _group_small_sections(sections: list, parent_char_budget: int) -> List[List]:
    """
    Group consecutive under-floor sections into single parents; leave substantial ones alone.

    Two boundaries are respected. A group never crosses an H1 (`product_line`), because that is a
    topic boundary — the switches intro must not end up in the same parent as a lock. And a group
    never exceeds `parent_char_budget`, so merging cannot recreate the oversized-section problem
    the parent splitter exists to solve.
    """
    groups: List[List] = []
    current: List = []
    current_len = 0

    def line_of(section) -> str:
        return (getattr(section, "metadata", None) or {}).get("product_line", "")

    for section in sections:
        text = section.page_content
        if len(text) >= RAG_MIN_PARENT_CHARS:
            if current:
                groups.append(current)
                current, current_len = [], 0
            groups.append([section])
            continue
        crosses_h1 = bool(current) and line_of(current[0]) != line_of(section)
        would_overflow = current_len + len(text) + 2 > parent_char_budget
        if current and (crosses_h1 or would_overflow):
            groups.append(current)
            current, current_len = [], 0
        current.append(section)
        current_len += len(text) + 2

    if current:
        groups.append(current)
    return groups


def _shared_metadata(sections: list, base: dict) -> dict:
    """
    Metadata for a merged parent: the base document keys plus only those header keys the grouped
    sections agree on.

    A parent covering `## 1 SW` through `## 8 SW` has no single `product_series`, so the key is
    dropped rather than being made up from the first section. The CHILDREN keep their own — which
    is what any per-product filtering would need anyway.
    """
    merged = {**base}
    if not sections:
        return merged
    first = {**base, **((getattr(sections[0], "metadata", None) or {}))}
    for key, value in first.items():
        if all(
            ({**base, **((getattr(s, "metadata", None) or {}))}).get(key) == value
            for s in sections
        ):
            merged[key] = value
        else:
            merged.pop(key, None)
    return merged


def split_markdown_to_parent_child(
    markdown_text: str,
    doc_metadata: dict | None = None,
) -> List[ProcessedChunk]:
    """
    Splits a Markdown document into Parent-Child chunk pairs.

    Flow:
    1. MarkdownHeaderTextSplitter splits by H1/H2/H3 headers.
    2. Each header-split section becomes a PARENT chunk (up to 1024 tokens).
    3. Each parent is further split into CHILD chunks (128-256 tokens)
       using RecursiveCharacterTextSplitter with 100-token overlap.
    4. MD5 hashes are computed on each section for idempotent upserts.

    Args:
        markdown_text: The cleaned Markdown string from Docling.
        doc_metadata: Base metadata dict (e.g., {"doc_type": "TECHNICAL_SPEC"}).

    Returns:
        A flat list of ProcessedChunk objects (parents + children interleaved).
    """
    if doc_metadata is None:
        doc_metadata = {}

    all_chunks: List[ProcessedChunk] = []

    # Step 1: Split by Markdown headers
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False  # Keep headers in content for context
    )
    header_sections = md_splitter.split_text(markdown_text)

    # Character budget for a parent (token setting → chars).
    parent_char_budget = settings.RAG_PARENT_CHUNK_SIZE * CHARS_PER_TOKEN

    # Step 2: Safety-net splitters.
    #  - child_splitter produces the dense embedding targets.
    #  - parent_splitter recursively splits a section that EXCEEDS the parent budget
    #    into multiple parent-sized blocks, so no content is dropped (previously an
    #    oversized section was truncated to a single parent and the overflow lost).
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.RAG_CHILD_CHUNK_SIZE * CHARS_PER_TOKEN,
        chunk_overlap=settings.RAG_CHUNK_OVERLAP * CHARS_PER_TOKEN,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_char_budget,
        chunk_overlap=settings.RAG_CHUNK_OVERLAP * CHARS_PER_TOKEN,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )

    for group in _group_small_sections(header_sections, parent_char_budget):
        # A group is either one substantial section or several short consecutive ones merged into
        # a single parent (see RAG_MIN_PARENT_CHARS).
        section_text = "\n\n".join(s.page_content for s in group)
        section_metadata = _shared_metadata(group, doc_metadata)
        if len(group) > 1:
            logger.info(
                "Merged %d short sections (%d chars) into one parent: %s",
                len(group), len(section_text),
                ", ".join(
                    ((getattr(s, "metadata", None) or {}).get("product_series") or "<unnamed>")
                    for s in group
                )[:160],
            )

        # Oversized sections are recursively split into MULTIPLE parent-sized blocks so
        # no content is dropped; small sections pass through unchanged as one parent.
        if len(section_text) > parent_char_budget:
            parent_texts = parent_splitter.split_text(section_text)
            logger.info(
                "Section '%s' (%d chars) exceeds parent budget %d; split into %d parents.",
                section_metadata.get("product_series")
                or section_metadata.get("product_line")
                or "<unnamed>",
                len(section_text), parent_char_budget, len(parent_texts),
            )
            child_sources = None
        else:
            parent_texts = [section_text]
            # Children come from each ORIGINAL section, not from the merged block: a merged parent
            # would otherwise be re-split on size alone and one embedding would cover four
            # products. Each section's text is a verbatim substring of the merged parent, so the
            # child-is-a-substring-of-its-parent guarantee still holds.
            child_sources = [(s.page_content, {**doc_metadata, **((getattr(s, "metadata", None) or {}))}) for s in group]

        for parent_text in parent_texts:
            # ─── Create PARENT chunk ───
            # Namespace the MD5 by role ("parent:") so a parent never collides with its
            # own sole child when a small block yields a single child equal to it.
            parent_hash = hashlib.md5(("parent:" + parent_text).encode("utf-8")).hexdigest()
            all_chunks.append(ProcessedChunk(
                content=parent_text,
                is_parent=True,
                metadata=section_metadata,
                source_hash=parent_hash,
            ))

            # ─── Create CHILD chunks ───
            # Split children from parent_text (not the full section) so every child is a
            # substring of its parent — otherwise a child could match on text absent
            # from the parent context injected into the LLM.
            sources = child_sources if child_sources is not None else [(parent_text, section_metadata)]
            for source_text, source_metadata in sources:
                child_docs = child_splitter.create_documents(
                    [source_text],
                    metadatas=[source_metadata]
                )
                for child_doc in child_docs:
                    child_hash = hashlib.md5(("child:" + child_doc.page_content).encode("utf-8")).hexdigest()
                    all_chunks.append(ProcessedChunk(
                        content=child_doc.page_content,
                        is_parent=False,
                        metadata=child_doc.metadata,
                        source_hash=child_hash,
                        parent_content_ref=parent_text,  # Used to link after parent DB insert
                    ))

    logger.info(
        f"Split document into {sum(1 for c in all_chunks if c.is_parent)} parents "
        f"and {sum(1 for c in all_chunks if not c.is_parent)} children."
    )
    return all_chunks
