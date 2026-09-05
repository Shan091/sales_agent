"""
Edge-case tests for parent-child chunking (Phase 3). No DB / embedding model required.

Regression focus: oversized sections must be recursively split into MULTIPLE parents
(no content dropped), every child must be a substring of its parent, and parent/child
source_hashes must never collide.
"""
from src.rag.embeddings import split_markdown_to_parent_child, CHARS_PER_TOKEN
from config.settings import settings


def test_child_is_substring_of_some_parent():
    md = "# Series\n\n## Model A\n" + ("spec line alpha. " * 50)
    chunks = split_markdown_to_parent_child(md, {"doc_type": "TECHNICAL_SPEC"})
    parents = [c.content for c in chunks if c.is_parent]
    for child in [c for c in chunks if not c.is_parent]:
        assert any(child.content in p for p in parents), "child is not a substring of any parent"


def test_oversized_section_splits_into_multiple_parents_without_content_loss():
    budget = settings.RAG_PARENT_CHUNK_SIZE * CHARS_PER_TOKEN
    big = "# Line\n\n## Big Model\n" + "\n".join(
        f"Spec row number {i} with detailed technical data." for i in range(400)
    )
    assert len(big) > budget * 2  # genuinely oversized

    chunks = split_markdown_to_parent_child(big, {"doc_type": "TECHNICAL_SPEC"})
    parents = [c for c in chunks if c.is_parent]

    assert len(parents) >= 2, "oversized section must yield multiple parents, not a truncated one"
    assert all(len(p.content) <= budget + 50 for p in parents), "a parent exceeded the budget"
    # Content near the END of the section must survive (previously truncated away).
    joined = "\n".join(p.content for p in parents)
    assert "Spec row number 399" in joined


def test_parent_child_source_hashes_are_unique():
    md = "# S\n\n## Tiny\n- one line only"
    chunks = split_markdown_to_parent_child(md)
    hashes = [c.source_hash for c in chunks]
    assert len(hashes) == len(set(hashes)), "parent/child source_hash collision"


def test_category_metadata_propagates_to_all_chunks():
    chunks = split_markdown_to_parent_child(
        "# S\n\n## M\n- spec", {"doc_type": "X", "category": "switches"}
    )
    assert chunks and all(c.metadata.get("category") == "switches" for c in chunks)
