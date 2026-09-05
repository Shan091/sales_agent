# file: src/rag/search.py
"""
Phase 3: Async Hybrid Search with Asymmetric RRF.

Architecture:
1. Dense Search: pgvector cosine distance (<=>) on child chunk embeddings.
2. Lexical Search: PostgreSQL tsvector with GIN index and `simple` dictionary.
3. Reciprocal Rank Fusion (RRF): Merges dense + lexical scores with asymmetric
   weighting that heavily favors exact keyword matches (the "Trump Card").
4. Parent Dedup: Deduplicates parent_ids via Python set() and retrieves parent
   text via a single IN clause to prevent N+1 queries.
5. Fail-Closed: If top similarity score < RAG_SIMILARITY_THRESHOLD, returns
   empty list to trigger the RAG-to-Search Fallback upstream.

All queries use SQLAlchemy Core primitives (NOT ORM) to prevent serialization errors.
"""
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy import Integer, literal, select, text, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.storage.models import DocumentChunk
from config.settings import settings

logger = logging.getLogger(__name__)


# Minimum number of DISTINCT query terms a chunk must match for the lexical hit to rescue an
# otherwise below-threshold retrieval. 1 would let a single coincidental word through (see the
# fail-closed gate below); 2 keeps real multi-word product references working.
MIN_LEXICAL_TERMS_FOR_RESCUE = 2


# Terms carrying no retrieval signal. The tsvector uses the `simple` dictionary (which
# deliberately keeps stopwords so exact codes like "4SW" survive), so question words would
# otherwise be treated as real search terms.
_LEXICAL_STOPWORDS = frozenset({
    "what", "whats", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "can", "could", "will", "would", "shall", "should", "may", "might",
    "have", "has", "had", "the", "a", "an", "and", "or", "but", "if", "then", "than",
    "of", "on", "in", "at", "to", "for", "with", "from", "by", "about", "as", "into",
    "it", "its", "this", "that", "these", "those", "there", "here",
    "i", "you", "we", "they", "he", "she", "me", "my", "your", "our", "their",
    "tell", "give", "need", "want", "please", "any", "some", "get", "got", "know",
})


# A SKU is written two ways and only one of them ever matched. The catalogue writes "4 SW"; the
# customer — and `QueryExpansion`, which rewrites their words before we ever see them — writes
# "4SW" just as often. The tsvector uses the `simple` dictionary, which does no stemming and
# splits on whitespace, so "4 SW" indexes as the lexemes (4, sw) while the token "4sw" is a
# single lexeme: the two never meet. Measured against the live chunk holding "less than 500W per
# gang":
#     to_tsquery('simple','4sw')    @@ vector  ->  FALSE
#     to_tsquery('simple','4<->sw') @@ vector  ->  TRUE
# So the exact-SKU hit that the asymmetric k=30 boost exists to surface never fired for the
# spelling the model most often produces, and a 65-character product section lost every time to
# the 1.7kB document intro. `<->` is tsquery's "immediately followed by", which is exactly the
# relation between "4" and "SW" in the corpus — and far tighter than ORing the two lexemes, where
# a bare "sw" would match every switch section in the file.
#
# The same normalisation in a different dialect already exists in `rag.py::_mentions`, which
# strips non-alphanumerics so `6SW` == `6 SW` for the grounding check. A tsquery cannot do that,
# hence the phrase alias.
_SKU_SPLIT_RE = re.compile(r"^(\d{1,3})([a-z]{1,4})$")

# Longest trailing unit accepted when joining a digit to the word after it ("4" + "sw"). Also the
# length at or below which that word is NOT kept as a term of its own: "sw", "a", "m" and "v" are
# noise on their own, while a real word ("way", "fan") still earns its own concept.
_SKU_UNIT_MAX_LEN = 4
_BARE_UNIT_MAX_LEN = 2


def _lexical_term_groups(query_text: str) -> List[List[str]]:
    """
    The meaningful search terms of a query, grouped one list per CONCEPT.

    `plainto_tsquery` ANDs every token, so a full question ("what is the warranty on the smart
    switches") only matched a chunk containing *every* word — including stopwords the `simple`
    dictionary never strips. In practice that matched nothing, which silently disabled the lexical
    half of the hybrid search AND the fail-closed rescue path. Callers OR the groups together, so
    any keyword/SKU hit still surfaces and RRF ranking decides how much it matters.

    Grouped rather than flat because one concept can have two spellings (see `_SKU_SPLIT_RE`) and
    `matched_terms` counts how many distinct query CONCEPTS a chunk matched:
    `MIN_LEXICAL_TERMS_FOR_RESCUE` is a documented safety property, so two spellings of the same
    product name must not be able to rescue an off-domain query between them.

    Returns an empty list when nothing useful remains.
    """
    # Keep alphanumerics only; tsquery operators in raw user text must never reach SQL.
    tokens = re.findall(r"[A-Za-z0-9]+", query_text.lower())
    groups: List[List[str]] = []
    seen: set[str] = set()

    def add(variants: List[str]) -> None:
        if variants[0] in seen:
            return
        seen.add(variants[0])
        groups.append(variants)

    i = 0
    while i < len(tokens):
        token = tokens[i]
        following = tokens[i + 1] if i + 1 < len(tokens) else None
        if (
            token.isdigit() and len(token) <= 3
            and following is not None
            and following.isalpha() and len(following) <= _SKU_UNIT_MAX_LEN
            and following not in _LEXICAL_STOPWORDS
        ):
            # "4 SW" arrived as two tokens: one concept, both spellings.
            add([f"{token}{following}", f"{token}<->{following}"])
            if len(following) > _BARE_UNIT_MAX_LEN:
                add([following])
            i += 2
            continue
        i += 1
        if len(token) <= 1 or token in _LEXICAL_STOPWORDS:
            continue
        variants = [token]
        sku = _SKU_SPLIT_RE.match(token)
        if sku:
            variants.append(f"{sku.group(1)}<->{sku.group(2)}")
        add(variants)
    return groups


def _group_tsquery(group: List[str]) -> str:
    """
    One concept as a tsquery fragment — its spellings OR'd and parenthesised.

    Explicit brackets so that mixing `|` with `<->` never rests on operator precedence, either
    here or in the joined query the caller builds.
    """
    return f"({' | '.join(group)})" if len(group) > 1 else group[0]


# ═══════════════════════════════════════════════
#  Result Dataclass (Decoupled from ORM)
# ═══════════════════════════════════════════════

@dataclass
class RetrievedContext:
    """A clean, ORM-decoupled container for a retrieved parent chunk."""
    parent_id: int
    content: str
    metadata: dict
    rrf_score: float


# ═══════════════════════════════════════════════
#  Core Hybrid Search
# ═══════════════════════════════════════════════

async def hybrid_search(
    session: AsyncSession,
    query_embeddings: List[List[float]],
    query_texts: List[str],
    category_filter: Optional[str] = None,
    doc_type_filter: Optional[str] = None,
    top_k_children: int = None,
    top_k_final: int = None,
) -> List[RetrievedContext]:
    """
    Executes an Asymmetric RRF hybrid search across multiple queries.

    Args:
        session: The async database session.
        query_embeddings: List of embedding vectors (one per expanded query).
        query_texts: List of raw text queries for lexical matching.
        category_filter: Optional hard SQL WHERE on metadata->>'category'.
        doc_type_filter: Optional hard SQL WHERE on metadata->>'doc_type'.
        top_k_children: Number of child chunks to retrieve per query.
        top_k_final: Number of deduplicated parent chunks to return.

    Returns:
        List of RetrievedContext objects (parent chunks), sorted by RRF score.
        Returns empty list if top score < RAG_SIMILARITY_THRESHOLD (fail-closed).
    """
    if top_k_children is None:
        top_k_children = settings.RAG_TOP_K_CHILDREN
    if top_k_final is None:
        top_k_final = settings.RAG_TOP_K_FINAL

    threshold = settings.RAG_SIMILARITY_THRESHOLD

    # ─── Collect child chunk IDs + scores across all queries ───
    # We accumulate (child_id, parent_id, rrf_score) tuples from all 3 queries
    child_scores: dict[int, dict] = {}  # child_id -> {parent_id, dense_rank, lexical_rank}

    for i, (embedding, query_text) in enumerate(zip(query_embeddings, query_texts)):
        # ─── Dense Vector Search (cosine distance) ───
        dense_stmt = (
            select(
                DocumentChunk.id,
                DocumentChunk.parent_id,
                DocumentChunk.embedding.cosine_distance(embedding).label("distance")
            )
            .where(DocumentChunk.is_parent == False)  # Only search child chunks
            .where(DocumentChunk.embedding.isnot(None))
        )

        # Apply hard metadata filters if provided
        if category_filter:
            dense_stmt = dense_stmt.where(
                DocumentChunk.chunk_metadata["category"].astext == category_filter
            )
        if doc_type_filter:
            dense_stmt = dense_stmt.where(
                DocumentChunk.chunk_metadata["doc_type"].astext == doc_type_filter
            )

        dense_stmt = dense_stmt.order_by("distance").limit(top_k_children)

        dense_results = (await session.execute(dense_stmt)).all()

        # ─── Lexical Search (tsvector with simple dictionary) ───
        # OR-joined tsquery over concept groups (see _lexical_term_groups): ANDing every token
        # made natural questions match nothing. Skipped entirely when the query has no
        # meaningful terms.
        lexical_groups = _lexical_term_groups(query_text)
        lexical_query = " | ".join(_group_tsquery(g) for g in lexical_groups) if lexical_groups else None
        lexical_results = []
        if lexical_query:
            # Count how many DISTINCT query concepts each chunk matched. A single incidental
            # word ("today") must not count as keyword evidence, while a real SKU hit must —
            # and the two spellings of one SKU count once, not twice.
            matched_terms_expr = sum(
                (
                    func.cast(
                        DocumentChunk.text_search_vector.op("@@")(
                            func.to_tsquery("simple", _group_tsquery(group))
                        ),
                        Integer,
                    )
                    for group in lexical_groups
                ),
                literal(0),
            )
            lexical_stmt = (
                select(
                    DocumentChunk.id,
                    DocumentChunk.parent_id,
                    func.ts_rank(
                        DocumentChunk.text_search_vector,
                        func.to_tsquery("simple", lexical_query)
                    ).label("ts_rank"),
                    matched_terms_expr.label("matched_terms"),
                )
                .where(DocumentChunk.is_parent == False)
                .where(DocumentChunk.text_search_vector.isnot(None))
                .where(
                    DocumentChunk.text_search_vector.op("@@")(
                        func.to_tsquery("simple", lexical_query)
                    )
                )
            )

            if category_filter:
                lexical_stmt = lexical_stmt.where(
                    DocumentChunk.chunk_metadata["category"].astext == category_filter
                )
            if doc_type_filter:
                lexical_stmt = lexical_stmt.where(
                    DocumentChunk.chunk_metadata["doc_type"].astext == doc_type_filter
                )

            lexical_stmt = lexical_stmt.order_by(text("ts_rank DESC")).limit(top_k_children)

            lexical_results = (await session.execute(lexical_stmt)).all()

        # ─── Merge into child_scores (Asymmetric RRF) ───
        # Dense: rank by distance (lower = better)
        for rank, row in enumerate(dense_results):
            child_id = row.id
            if child_id not in child_scores:
                child_scores[child_id] = {
                    "parent_id": row.parent_id,
                    "dense_rank_sum": 0.0,
                    "lexical_rank_sum": 0.0,
                    "max_matched_terms": 0,
                    "min_distance": row.distance,
                }
            # RRF: 1 / (k + rank), with k=60 (standard RRF constant)
            child_scores[child_id]["dense_rank_sum"] += 1.0 / (60 + rank)
            child_scores[child_id]["min_distance"] = min(
                child_scores[child_id]["min_distance"], row.distance
            )

        # Lexical: rank by ts_rank (higher = better)
        # ASYMMETRIC BOOST: We use k=30 (instead of 60) for lexical scores.
        # This mathematically doubles the weight of exact keyword matches,
        # forcing exact product codes (4SW, Grande) to dominate the final rank.
        for rank, row in enumerate(lexical_results):
            child_id = row.id
            if child_id not in child_scores:
                child_scores[child_id] = {
                    "parent_id": row.parent_id,
                    "dense_rank_sum": 0.0,
                    "lexical_rank_sum": 0.0,
                    "max_matched_terms": 0,
                    "min_distance": 1.0,  # Default high distance for lexical-only matches
                }
            # ASYMMETRIC: k=30 for lexical (vs k=60 for dense) = 2x weight boost
            child_scores[child_id]["lexical_rank_sum"] += 1.0 / (30 + rank)
            child_scores[child_id]["max_matched_terms"] = max(
                child_scores[child_id]["max_matched_terms"], row.matched_terms or 0
            )

    if not child_scores:
        logger.warning("Hybrid search returned zero results.")
        return []

    # ─── Calculate combined RRF scores ───
    scored_children = []
    for child_id, scores in child_scores.items():
        rrf_score = scores["dense_rank_sum"] + scores["lexical_rank_sum"]
        similarity = 1.0 - scores["min_distance"]  # Convert distance to similarity
        scored_children.append({
            "child_id": child_id,
            "parent_id": scores["parent_id"],
            "rrf_score": rrf_score,
            "similarity": similarity,
            "lexical_rank_sum": scores["lexical_rank_sum"],
            "matched_terms": scores["max_matched_terms"],
        })

    # Sort by RRF score descending
    scored_children.sort(key=lambda x: x["rrf_score"], reverse=True)

    # ─── Fail-Closed Gate ───
    # Retrieval "passes" if EITHER some chunk is semantically close enough
    # (dense cosine similarity >= threshold) OR a chunk is a STRONG lexical
    # (keyword/SKU) hit. Lexical-only matches are seeded with min_distance=1.0
    # (similarity 0.0), so gating on the top-RRF chunk's similarity alone would
    # wrongly discard exact product-code matches — the very thing the asymmetric
    # RRF boost exists to surface (the "Trump Card"). We therefore gate on the
    # best similarity across ALL children plus a strong lexical hit.
    #
    # "Strong" = the chunk matched at least MIN_LEXICAL_TERMS_FOR_RESCUE distinct query
    # concepts. The tsquery ORs its groups (see _lexical_term_groups), so a single incidental word
    # ("today" in "what is the weather in Paris today") matches something in almost any
    # corpus; rescuing on that would let every off-domain query through. Requiring 2+
    # distinct terms keeps genuine multi-word product references (e.g. "8 SW panel")
    # working while blocking coincidental single-word overlap.
    best_similarity = max((c["similarity"] for c in scored_children), default=0.0)
    has_strong_lexical_match = any(
        c["matched_terms"] >= MIN_LEXICAL_TERMS_FOR_RESCUE for c in scored_children
    )
    if best_similarity < threshold and not has_strong_lexical_match:
        logger.warning(
            f"RAG Fail-Closed: best similarity {best_similarity:.4f} < threshold {threshold} "
            "and no lexical matches. Returning empty results to trigger fallback."
        )
        return []

    # ─── Deduplicate parent IDs ───
    # Multiple child chunks may point to the same parent table.
    # We use a set to prevent injecting the same parent text into the LLM context multiple times.
    seen_parent_ids: set[int] = set()
    unique_parents: List[dict] = []

    for child in scored_children:
        pid = child["parent_id"]
        if pid is None:
            continue
        if pid not in seen_parent_ids:
            seen_parent_ids.add(pid)
            unique_parents.append(child)
        if len(unique_parents) >= top_k_final:
            break

    if not unique_parents:
        logger.warning("No valid parent chunks found after deduplication.")
        return []

    # ─── Array-Aware Parent Retrieval (Single IN clause) ───
    parent_ids = list(seen_parent_ids)
    parent_stmt = (
        select(
            DocumentChunk.id,
            DocumentChunk.content,
            DocumentChunk.chunk_metadata.label("chunk_metadata"),
        )
        .where(DocumentChunk.id.in_(parent_ids))
    )
    parent_rows = (await session.execute(parent_stmt)).all()

    # Build a lookup map
    parent_map = {row.id: row for row in parent_rows}

    # ─── Assemble final results ───
    results: List[RetrievedContext] = []
    for parent_info in unique_parents:
        pid = parent_info["parent_id"]
        parent_row = parent_map.get(pid)
        if parent_row:
            results.append(RetrievedContext(
                parent_id=pid,
                content=parent_row.content,
                metadata=parent_row.chunk_metadata if parent_row.chunk_metadata else {},
                rrf_score=parent_info["rrf_score"],
            ))

    if not results:
        # unique_parents was non-empty, but none of the parent_ids resolved to a
        # row (e.g. orphaned child parent_ids). Bail out instead of indexing
        # results[0] below, which would raise IndexError.
        logger.warning(
            "Hybrid search resolved zero parent rows (orphaned child parent_ids?). "
            "Returning empty results to trigger fallback."
        )
        return []

    logger.info(
        f"Hybrid search complete. Retrieved {len(results)} parent chunks "
        f"(from {len(child_scores)} child matches). Top RRF: {results[0].rrf_score:.4f}"
    )
    return results
