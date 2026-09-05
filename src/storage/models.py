# file: src/storage/models.py
from typing import List, Optional, Any
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field, Column, JSON
from sqlalchemy import DateTime, Index, ForeignKey, func, text, Computed
from sqlalchemy.ext.mutable import MutableList, MutableDict
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from pgvector.sqlalchemy import Vector

# H3 FIX: Import the canonical enum from core instead of defining a stale duplicate
from src.core.enums import UserArchetype


class CRMLead(SQLModel, table=True):
    __tablename__ = "crm_leads"

    id: Optional[int] = Field(default=None, primary_key=True)
    whatsapp_id: str = Field(index=True, unique=True)

    archetype: str = Field(default="UNKNOWN")  # Stored as string to match graph output directly

    budget_bracket: Optional[str] = Field(default=None)
    property_scope: Optional[str] = Field(default=None)

    # FIX: Wrapped JSON with MutableList so .append() triggers DB updates
    pain_points: List[str] = Field(
        default=[],
        sa_column=Column(MutableList.as_mutable(JSON))
    )
    product_interests: List[str] = Field(
        default=[],
        sa_column=Column(MutableList.as_mutable(JSON))
    )

    # Phase 3: Append RAG context titles the user interacted with for CRM handoff
    rag_contexts_viewed: List[str] = Field(
        default=[],
        sa_column=Column(MutableList.as_mutable(JSON))
    )

    timeline: Optional[str] = Field(default=None)

    # timezone=True is load-bearing, not cosmetic. The default_factory produces an AWARE
    # datetime; a bare `datetime` annotation maps to TIMESTAMP WITHOUT TIME ZONE, and asyncpg
    # then rejects the insert outright ("can't subtract offset-naive and offset-aware
    # datetimes"). Because the write is fail-soft, that surfaced as rows silently never
    # appearing rather than as an error anyone would notice.
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, onupdate=func.now()),
    )


class ProductPricing(SQLModel, table=True):
    __tablename__ = "products_pricing"

    __table_args__ = (
        Index("ix_pricing_product_region_active", "product_name", "region_code", "is_active"),
    )

    sku_id: Optional[int] = Field(default=None, primary_key=True)
    product_name: str = Field(index=True)
    region_code: str = Field(index=True)

    base_price: float = Field(...)
    installation_fee: float = Field(default=0.0)
    currency: str = Field(default="INR")

    is_active: bool = Field(default=True)


# ═══════════════════════════════════════════════
#  Agentic commerce: the money audit trail
# ═══════════════════════════════════════════════

class PaymentOrder(SQLModel, table=True):
    """
    One row per money action, written by the worker (never by a graph node) so a
    LangGraph checkpoint replay can't duplicate it.

    This IS the audit trail Track 01 asks for: every proposal, every link mint, every
    paid/failed transition is recorded together with the *inputs that produced the
    amount* — the resolved unit prices, the offer the agent chose, the discount that
    was actually allowed after clamping, and the grand total. Given a row you can
    re-derive the amount by hand and see exactly why it is what it is.

    Status lifecycle: proposed -> link_created -> paid | failed | expired.
    """
    __tablename__ = "payment_orders"

    __table_args__ = (
        Index("ix_payment_orders_thread_status", "thread_id", "status"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)

    # WhatsApp number == LangGraph thread_id, so an order is always traceable to a chat.
    thread_id: str = Field(index=True)

    # Exactly what was charged for: [{sku, qty, unit_price, installation_fee, line_total}].
    # unit_price is the value read from products_pricing at mint time, NOT anything the
    # LLM produced — this is the field that makes the amount re-derivable.
    line_items: List[dict] = Field(
        default_factory=list,
        sa_column=Column(MutableList.as_mutable(JSONB))
    )
    product_summary: str = Field(default="")

    # Bounded-discount audit: which predefined offer the agent selected, the pct that
    # survived clamping to MAX_DISCOUNT_PCT, and the rupee value of that discount.
    applied_offer: Optional[str] = Field(default=None)
    discount_pct: float = Field(default=0.0)
    discount_amount: float = Field(default=0.0)

    subtotal: float = Field(default=0.0)
    amount: float = Field(default=0.0)  # grand total actually sent to Razorpay
    currency: str = Field(default="INR")

    status: str = Field(default="proposed", index=True)

    razorpay_link_id: Optional[str] = Field(default=None, index=True)
    razorpay_payment_id: Optional[str] = Field(default=None)
    payment_link_url: Optional[str] = Field(default=None)

    # Why this order ended up where it did: clamp notes, guardrail reasons, decline
    # reasons. Human-readable so the trail explains itself without the code at hand.
    audit_notes: List[str] = Field(
        default_factory=list,
        sa_column=Column(MutableList.as_mutable(JSONB))
    )

    # Verbatim Razorpay webhook event that closed the loop (paid/failed).
    raw_event: dict = Field(
        default_factory=dict,
        sa_column=Column("raw_event", MutableDict.as_mutable(JSONB))
    )

    # timezone=True is load-bearing, not cosmetic. The default_factory produces an AWARE
    # datetime; a bare `datetime` annotation maps to TIMESTAMP WITHOUT TIME ZONE, and asyncpg
    # then rejects the insert outright ("can't subtract offset-naive and offset-aware
    # datetimes"). Because the write is fail-soft, that surfaced as rows silently never
    # appearing rather than as an error anyone would notice.
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, onupdate=func.now()),
    )


# ═══════════════════════════════════════════════
#  Phase 3: RAG Document Storage (Parent-Child)
# ═══════════════════════════════════════════════

class DocumentChunk(SQLModel, table=True):
    """
    Stores RAG document chunks with Parent-Child hierarchy.

    Architecture:
    - Parent Chunks (1024 tokens): The full structural context (e.g., a complete product table).
      Injected into the LLM context window for grounding.
    - Child Chunks (128-256 tokens): Dense, highly specific text blocks.
      Embedded as vectors for precise pgvector distance matching.
    - A child's `parent_id` foreign key links back to the parent row.
      Parent rows have `parent_id = NULL`.

    Indexes:
    - HNSW (m=16, ef_construction=64) on the embedding column for sub-ms dense lookups.
    - GIN on the pre-computed tsvector column for instant lexical search.
    """
    __tablename__ = "document_chunks"

    __table_args__ = (
        # HNSW index for sub-millisecond dense vector lookups.
        # m=16: max connections per layer (higher = better recall, slower build).
        # ef_construction=64: search width during index build (higher = better recall).
        Index(
            "idx_doc_chunk_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"}
        ),
        # GIN index on the pre-computed tsvector for instant keyword lookups.
        Index(
            "idx_doc_chunk_text_search",
            "text_search_vector",
            postgresql_using="gin"
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)

    # The raw text content of this chunk.
    content: str

    # Self-referencing FK for Parent-Child hierarchy.
    # Parent rows: parent_id is NULL.
    # Child rows: parent_id points to the parent row's id.
    parent_id: Optional[int] = Field(
        default=None,
        foreign_key="document_chunks.id",
    )

    # Whether this chunk is a parent (context) or child (vector target).
    is_parent: bool = Field(default=False)

    # Structured metadata for hard filtering during retrieval.
    # Example: {"doc_type": "TECHNICAL_SPEC", "category": "security", "product_series": "Grande"}
    # NOTE: the Python attribute is `chunk_metadata` because SQLAlchemy reserves the
    # `metadata` attribute name on declarative classes (naming it `metadata` raises
    # InvalidRequestError at import). The DB column is still named "metadata", so JSONB
    # filtering (metadata->>'category') is unchanged.
    # MutableDict.as_mutable ensures in-place dict-key mutations trigger DB writes
    # (MutableList would reject a dict value with a ValueError).
    chunk_metadata: dict = Field(
        default_factory=dict,
        sa_column=Column("metadata", MutableDict.as_mutable(JSONB))
    )

    # MD5 hash of the source section for idempotent upserts.
    # Allows patch updates without full re-indexing.
    source_hash: Optional[str] = Field(default=None, index=True)

    # Vector embedding — 1024 dimensions for BAAI/bge-m3.
    # Only populated for child chunks (parents are retrieved via FK join, not vector search).
    embedding: Any = Field(
        default=None,
        sa_column=Column(Vector(1024), nullable=True)
    )

    # Pre-computed tsvector column using PostgreSQL's `simple` dictionary.
    # `simple` preserves exact brand codes (Grande, Eco, 4SW) without stemming.
    # Computed at write-time to prevent CPU spikes during runtime read loops.
    text_search_vector: Any = Field(
        default=None,
        sa_column=Column(
            TSVECTOR,
            Computed("to_tsvector('simple', content)", persisted=True),
            nullable=True
        )
    )