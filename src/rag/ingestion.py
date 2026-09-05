# file: src/rag/ingestion.py
"""
Phase 3: RAG Document Ingestion Pipeline.

Architecture:
1. Accepts pre-cleaned Markdown text (from src/scripts/cleaner.py).
2. Splits into Parent-Child chunks via src/rag/embeddings.py.
3. Generates bge-m3 embeddings for child chunks only (parents are retrieved via FK).
4. Performs idempotent upserts using MD5 source_hash to allow patch updates
   without dropping the entire vector index.
5. Batch-embeds all child chunks in a single call to prevent sequential latency.
"""
import asyncio
import logging
from typing import List, Optional

from sqlalchemy import select, delete, or_
from sqlalchemy.ext.asyncio import AsyncSession

from src.storage.models import DocumentChunk
from src.rag.embeddings import split_markdown_to_parent_child, ProcessedChunk
from config.settings import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
#  Embedding Client Abstraction
# ═══════════════════════════════════════════════

class EmbeddingClient:
    """
    Abstraction layer for embedding generation.
    Supports local models (via sentence-transformers) or remote APIs.
    All calls MUST use batch embedding to prevent sequential latency.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None

    def _load_model(self):
        """Lazy-load the embedding model to prevent import-time VRAM allocation."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        return self._model

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Batch-embed multiple texts in a single call.
        CRITICAL: This MUST be a single round-trip, not sequential calls.
        For local models, we use SentenceTransformer.encode() which batches internally.

        encode() (and the first-call model load) are CPU/GPU-blocking, so we run them
        in a worker thread via asyncio.to_thread — otherwise they stall the event loop
        and freeze every other concurrent conversation sharing this worker.
        """
        def _embed_sync() -> List[List[float]]:
            model = self._load_model()
            # encode() handles batching internally on GPU/CPU
            return model.encode(texts, normalize_embeddings=True).tolist()

        return await asyncio.to_thread(_embed_sync)

    async def aembed_query(self, text: str) -> List[float]:
        """Embed a single query string. Used for retrieval (not ingestion)."""
        result = await self.aembed_documents([text])
        return result[0]


# Global singleton — lazy-loaded on first use
_embedding_client: Optional[EmbeddingClient] = None


def get_embedding_client() -> EmbeddingClient:
    """Returns the global EmbeddingClient singleton."""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client


# ═══════════════════════════════════════════════
#  Idempotent Ingestion Pipeline
# ═══════════════════════════════════════════════

async def ingest_markdown_document(
    session: AsyncSession,
    markdown_text: str,
    doc_metadata: dict | None = None,
) -> int:
    """
    Ingests a cleaned Markdown document into the DocumentChunk table.

    Flow:
    1. Split into parent-child chunks via embeddings.py.
    2. Check existing source_hashes for idempotent upserts.
    3. Delete stale chunks whose hashes no longer match.
    4. Insert new parent chunks first (to get IDs for FK linking).
    5. Batch-embed all child chunks in a single call.
    6. Insert child chunks with parent_id FK and embedding vectors.

    Args:
        session: The async database session.
        markdown_text: Pre-cleaned Markdown string.
        doc_metadata: Base metadata (e.g., {"doc_type": "TECHNICAL_SPEC"}).

    Returns:
        Total number of chunks inserted.
    """
    # Step 1: Split into parent-child chunks
    chunks = split_markdown_to_parent_child(markdown_text, doc_metadata)

    if not chunks:
        logger.warning("No chunks generated from document. Skipping ingestion.")
        return 0

    new_hashes = {c.source_hash for c in chunks if c.source_hash}

    # Step 2: Delete stale chunks — rows for THIS source document whose hash is no longer produced
    # by the current version (edited or removed sections). Scoped by metadata->>'source_file' so
    # other documents are never touched. Without a source_file we cannot safely scope the delete,
    # so we skip cleanup rather than risk wiping unrelated documents.
    #
    # An orphaned child is not merely untidy, it is a foreign-key violation: a child whose own text
    # did not change keeps a hash that IS still produced, so the hash filter spares it — while the
    # PARENT it points at can still be stale and deleted underneath it. That is not hypothetical;
    # it is what merging short sections into one parent does to every unchanged child, and it
    # aborted the whole re-ingest with `document_chunks_parent_id_fkey`. So the children of a
    # doomed parent go too, whatever their own hash says, and the "does this already exist?" check
    # runs AFTER the deletes — otherwise a child removed here would be filtered out of the insert
    # as already-present and silently lost.
    source_file = doc_metadata.get("source_file") if doc_metadata else None
    deleted_stale = 0
    if source_file:
        of_this_file = DocumentChunk.chunk_metadata["source_file"].astext == source_file
        is_stale = ~DocumentChunk.source_hash.in_(list(new_hashes))
        doomed_parents = select(DocumentChunk.id).where(
            of_this_file, is_stale, DocumentChunk.is_parent == True
        )
        del_children = await session.execute(
            delete(DocumentChunk).where(
                DocumentChunk.is_parent == False,
                of_this_file,
                or_(is_stale, DocumentChunk.parent_id.in_(doomed_parents)),
            )
        )
        del_parents = await session.execute(
            delete(DocumentChunk).where(DocumentChunk.is_parent == True, of_this_file, is_stale)
        )
        deleted_stale = (del_children.rowcount or 0) + (del_parents.rowcount or 0)
        if deleted_stale:
            logger.info(f"Deleted {deleted_stale} stale chunk(s) for source_file='{source_file}'.")
    else:
        logger.debug("No 'source_file' in metadata; skipping stale-chunk cleanup.")

    # Step 3: What survives (idempotent upserts). Read after the deletes, never before.
    existing_rows = (await session.execute(
        select(DocumentChunk.source_hash).where(DocumentChunk.source_hash.in_(list(new_hashes)))
    )).all()
    existing_hashes = {row.source_hash for row in existing_rows}

    # Filter out chunks that already exist (hash match = no change)
    chunks_to_insert = [c for c in chunks if c.source_hash not in existing_hashes]

    if not chunks_to_insert:
        if deleted_stale:
            await session.commit()  # persist stale deletions even when there is nothing new to insert
        logger.info("All current chunks already exist (hashes match). No new inserts needed.")
        return 0

    # Step 3: Separate parents and children
    parent_chunks = [c for c in chunks_to_insert if c.is_parent]
    child_chunks = [c for c in chunks_to_insert if not c.is_parent]

    # Step 4: Insert parent chunks first to get their IDs
    parent_content_to_id: dict[str, int] = {}

    for pc in parent_chunks:
        db_parent = DocumentChunk(
            content=pc.content,
            is_parent=True,
            parent_id=None,
            chunk_metadata=pc.metadata,
            source_hash=pc.source_hash,
            embedding=None,  # Parents are NOT embedded; they are retrieved via FK
        )
        session.add(db_parent)
        await session.flush()  # Flush to get the auto-generated ID
        parent_content_to_id[pc.content] = db_parent.id

    # Also map existing parents (for children whose parents already existed)
    if child_chunks:
        # Find parent IDs for children whose parents were already in the DB
        parent_refs_needed = {
            c.parent_content_ref for c in child_chunks
            if c.parent_content_ref not in parent_content_to_id
        }
        if parent_refs_needed:
            existing_parent_stmt = select(DocumentChunk.id, DocumentChunk.content).where(
                DocumentChunk.is_parent == True,
                DocumentChunk.content.in_(list(parent_refs_needed))
            )
            existing_parent_rows = (await session.execute(existing_parent_stmt)).all()
            for row in existing_parent_rows:
                parent_content_to_id[row.content] = row.id

    # Step 5: Batch-embed all child chunk texts in a single call
    inserted_children = 0
    if child_chunks:
        embed_client = get_embedding_client()
        child_texts = [c.content for c in child_chunks]
        logger.info(f"Batch-embedding {len(child_texts)} child chunks...")
        child_embeddings = await embed_client.aembed_documents(child_texts)

        # Step 6: Insert child chunks with parent_id FK and embeddings
        for child, embedding in zip(child_chunks, child_embeddings):
            parent_id = parent_content_to_id.get(child.parent_content_ref)
            if parent_id is None:
                logger.warning(
                    f"Orphan child chunk detected (no parent match). "
                    f"Hash: {child.source_hash}. Skipping."
                )
                continue

            db_child = DocumentChunk(
                content=child.content,
                is_parent=False,
                parent_id=parent_id,
                chunk_metadata=child.metadata,
                source_hash=child.source_hash,
                embedding=embedding,
            )
            session.add(db_child)
            inserted_children += 1

    await session.commit()

    # Count actual inserts — orphaned children are skipped above, so len(child_chunks)
    # would over-report.
    total_inserted = len(parent_chunks) + inserted_children
    logger.info(
        f"Ingestion complete. Inserted {len(parent_chunks)} parents + "
        f"{inserted_children} children = {total_inserted} total chunks"
        f"{f'; deleted {deleted_stale} stale' if deleted_stale else ''}."
    )
    return total_inserted
