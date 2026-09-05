# file: tests/test_rag_pipeline.py
"""
Phase 3: RAG Pipeline Evaluation Suite (LLM-as-a-Judge).

This test script validates the entire RAG pipeline end-to-end:
1. Ingests sample Markdown documents into the test database.
2. Runs hybrid search queries against the ingested data.
3. Evaluates retrieval quality using the RAG Triad:
   - Context Relevance: Did we retrieve the right document chunks?
   - Answer Groundedness: Is the answer grounded in the retrieved context?
4. Reports pass/fail for each "Golden Question" against known ground-truth answers.

Usage:
    pytest tests/test_rag_pipeline.py -v

Prerequisites:
    - A running PostgreSQL instance with pgvector extension enabled.
    - The BAAI/bge-m3 model downloaded (auto-downloads on first run).
    - Environment variables set in .env
"""
import pytest
import asyncio
from typing import List

from sqlalchemy import select, func, delete

from src.rag.embeddings import split_markdown_to_parent_child
from src.rag.search import hybrid_search, RetrievedContext
from src.rag.ingestion import ingest_markdown_document, get_embedding_client
from src.scripts.cleaner import clean_markdown
from src.core.database import async_session_maker, close_db_engine
from src.storage.models import DocumentChunk


# ═══════════════════════════════════════════════
#  Sample Test Document (Otohom-Like Spec Sheet)
# ═══════════════════════════════════════════════

SAMPLE_TECHNICAL_SPEC = """
# Grande Series Smart Switches

## Grande 1SW
- Type: Smart Switch (1 Gang)
- Max Load: 800W per gang
- Protocol: Wi-Fi 2.4GHz, Zigbee 3.0
- Voltage: 100-240V AC, 50/60Hz
- Frame: Machine-cut aluminum, Golden/Black
- Glass: Tempered, Black/White
- Neutral Wire: Required

## Grande 4SW
- Type: Smart Switch (4 Gang)
- Max Load: 500W per gang
- Protocol: Wi-Fi 2.4GHz, Zigbee 3.0
- Voltage: 100-240V AC, 50/60Hz
- Frame: Machine-cut aluminum, Golden/Black
- Glass: Tempered, Black/White
- Neutral Wire: Required

## Grande 6SW Fan
- Type: Smart Fan Regulator + 6 Gang Switch
- Max Load: 400W per gang (fan: 120W)
- Protocol: Wi-Fi 2.4GHz
- Voltage: 100-240V AC, 50/60Hz
- Fan Speed Levels: 1-5

# Security Solutions

## Smart Door Lock Premium
- Type: Digital Door Lock
- Unlock Methods: Fingerprint, PIN, RFID Card, Key, App
- Battery: 4x AA (8-12 months)
- Protocol: Zigbee 3.0, BLE
- Fire Alarm Auto-Unlock: Yes
- Anti-Tamper Alarm: Yes

## Smart Door Lock Base
- Type: Digital Door Lock
- Unlock Methods: PIN, RFID Card, Key
- Battery: 4x AA (6-8 months)
- Protocol: Wi-Fi 2.4GHz
- Fire Alarm Auto-Unlock: Yes
"""


# ═══════════════════════════════════════════════
#  Golden Questions (Ground Truth)
# ═══════════════════════════════════════════════

GOLDEN_QUESTIONS = [
    {
        "query": "What is the max load for the Grande 4SW switch?",
        "expected_substring": "500W per gang",
        "category": "switches",
    },
    {
        "query": "What protocol does the Grande 1SW use?",
        "expected_substring": "Zigbee 3.0",
        "category": "switches",
    },
    {
        "query": "How many fan speed levels does the Grande 6SW support?",
        "expected_substring": "1-5",
        "category": "switches",
    },
    {
        "query": "Does the Smart Door Lock Premium support fingerprint unlock?",
        "expected_substring": "Fingerprint",
        "category": "security",
    },
    {
        "query": "What is the battery life of the Smart Door Lock Base?",
        "expected_substring": "6-8 months",
        "category": "security",
    },
    {
        "query": "What voltage range do the Grande switches support?",
        "expected_substring": "100-240V",
        "category": "switches",
    },
    {
        "query": "Does the Grande 4SW require a neutral wire?",
        "expected_substring": "Neutral Wire: Required",
        "category": "switches",
    },
    {
        "query": "What unlock methods does the Smart Door Lock Base have?",
        "expected_substring": "PIN, RFID Card, Key",
        "category": "security",
    },
    {
        "query": "What is the max fan wattage on the Grande 6SW?",
        "expected_substring": "120W",
        "category": "switches",
    },
    {
        "query": "Does the Smart Door Lock Premium have anti-tamper protection?",
        "expected_substring": "Anti-Tamper Alarm: Yes",
        "category": "security",
    },
]


# ═══════════════════════════════════════════════
#  Test Fixtures
# ═══════════════════════════════════════════════

@pytest.fixture
async def ingested_db():
    """
    Ingest the sample document into the test database once for all tests.

    The sample mixes two product lines under separate H1 headers, so we ingest each
    block with its correct `category` metadata. The Golden Questions filter hard on
    metadata->>'category', so without per-section category tags every query would
    match zero rows and the eval would always score 0%.
    """
    # Split the sample on its second H1 so each block carries the right category.
    switches_md, security_md = SAMPLE_TECHNICAL_SPEC.split("# Security Solutions", 1)
    security_md = "# Security Solutions" + security_md

    batches = [
        (switches_md, {"doc_type": "TECHNICAL_SPEC", "category": "switches", "source_file": "sample_switches.md"}),
        (security_md, {"doc_type": "TECHNICAL_SPEC", "category": "security", "source_file": "sample_security.md"}),
    ]

    # Ingest (idempotent — duplicate source_hashes are skipped, so a re-run inserts 0).
    async with async_session_maker() as session:
        for raw_md, metadata in batches:
            await ingest_markdown_document(session, clean_markdown(raw_md), metadata)

    # Verify the rows are present regardless of whether they were freshly inserted.
    sample_sources = ["sample_switches.md", "sample_security.md"]
    async with async_session_maker() as session:
        count = (await session.execute(
            select(func.count()).select_from(DocumentChunk).where(
                DocumentChunk.chunk_metadata["source_file"].astext.in_(sample_sources)
            )
        )).scalar()
    assert count > 0, "Ingestion fixture: no test chunks found in DB after ingest."

    yield  # Run tests

    # Teardown: delete the test sample so test sample data doesn't pollute the live catalog
    # and so the next pytest run re-seeds cleanly.
    async with async_session_maker() as session:
        await session.execute(
            delete(DocumentChunk).where(
                DocumentChunk.chunk_metadata["source_file"].astext.in_(sample_sources)
            )
        )
        await session.commit()

    await close_db_engine()


# ═══════════════════════════════════════════════
#  Chunking Tests
# ═══════════════════════════════════════════════

class TestChunking:
    """Validates the MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter pipeline."""

    def test_parent_child_split_produces_chunks(self):
        """The splitter must produce at least 1 parent and 1 child chunk."""
        chunks = split_markdown_to_parent_child(SAMPLE_TECHNICAL_SPEC, {"doc_type": "TECHNICAL_SPEC"})
        parents = [c for c in chunks if c.is_parent]
        children = [c for c in chunks if not c.is_parent]

        assert len(parents) > 0, "No parent chunks were generated."
        assert len(children) > 0, "No child chunks were generated."

    def test_child_chunks_have_parent_reference(self):
        """Every child chunk must have a non-empty parent_content_ref."""
        chunks = split_markdown_to_parent_child(SAMPLE_TECHNICAL_SPEC)
        children = [c for c in chunks if not c.is_parent]

        for child in children:
            assert child.parent_content_ref, f"Child chunk has empty parent_content_ref: {child.content[:50]}"

    def test_source_hashes_are_unique(self):
        """MD5 hashes must be unique across chunks (no collisions)."""
        chunks = split_markdown_to_parent_child(SAMPLE_TECHNICAL_SPEC)
        hashes = [c.source_hash for c in chunks]
        assert len(hashes) == len(set(hashes)), "Duplicate source_hash detected! MD5 collision."

    def test_metadata_propagation(self):
        """Metadata from the base dict must be present on every chunk."""
        chunks = split_markdown_to_parent_child(SAMPLE_TECHNICAL_SPEC, {"doc_type": "TECHNICAL_SPEC"})
        for chunk in chunks:
            assert chunk.metadata.get("doc_type") == "TECHNICAL_SPEC", \
                f"Metadata not propagated to chunk: {chunk.content[:50]}"


# ═══════════════════════════════════════════════
#  Document Cleaner Tests
# ═══════════════════════════════════════════════

class TestCleaner:
    """Validates the pre-ingestion marketing fluff stripper."""

    def test_strips_marketing_fluff(self):
        """Known marketing phrases must be removed."""
        dirty = "# Product Specs\nDigitalize your physical world\n- Voltage: 240V\nOur story is worthy of reading!"
        cleaned = clean_markdown(dirty)
        assert "Digitalize your physical world" not in cleaned
        assert "Our story is worthy of reading" not in cleaned
        assert "Voltage: 240V" in cleaned

    def test_preserves_technical_content(self):
        """Technical specifications must survive the cleaner."""
        cleaned = clean_markdown(SAMPLE_TECHNICAL_SPEC)
        assert "Grande 4SW" in cleaned
        assert "500W per gang" in cleaned
        assert "Zigbee 3.0" in cleaned

    def test_strips_phone_numbers(self):
        """Phone numbers must be removed."""
        dirty = "Call us at +91-9876543210 for support.\n- Max Load: 800W"
        cleaned = clean_markdown(dirty)
        assert "+91-9876543210" not in cleaned
        assert "Max Load: 800W" in cleaned


# ═══════════════════════════════════════════════
#  Hybrid Search Tests (Requires DB)
# ═══════════════════════════════════════════════

@pytest.mark.asyncio
class TestHybridSearch:
    """
    End-to-end retrieval tests using Golden Questions.
    These tests require a running PostgreSQL instance with pgvector.
    """

    @pytest.mark.usefixtures("ingested_db")
    async def test_golden_questions_retrieve_correct_context(self):
        """Each Golden Question must retrieve a parent chunk containing the expected substring."""
        embed_client = get_embedding_client()
        passed = 0
        failed_questions = []

        for gq in GOLDEN_QUESTIONS:
            query_embedding = await embed_client.aembed_documents([gq["query"]])

            async with async_session_maker() as session:
                results: List[RetrievedContext] = await hybrid_search(
                    session=session,
                    query_embeddings=query_embedding,
                    query_texts=[gq["query"]],
                    category_filter=gq.get("category"),
                )

            # Check if the expected substring exists in any retrieved parent chunk
            found = any(
                gq["expected_substring"].lower() in r.content.lower()
                for r in results
            )

            if found:
                passed += 1
            else:
                retrieved_preview = [r.content[:100] for r in results] if results else ["NO RESULTS"]
                failed_questions.append({
                    "query": gq["query"],
                    "expected": gq["expected_substring"],
                    "retrieved": retrieved_preview,
                })

        # Report
        total = len(GOLDEN_QUESTIONS)
        pass_rate = (passed / total) * 100

        if failed_questions:
            fail_report = "\n".join(
                f"  ❌ Q: {fq['query']}\n     Expected: {fq['expected']}\n     Got: {fq['retrieved']}"
                for fq in failed_questions
            )
            print(f"\n{'='*60}")
            print(f"RAG Evaluation: {passed}/{total} passed ({pass_rate:.1f}%)")
            print(f"{'='*60}")
            print(fail_report)

        assert pass_rate >= 80.0, (
            f"RAG pipeline failed: {pass_rate:.1f}% pass rate (minimum 80%). "
            f"Failed questions: {[fq['query'] for fq in failed_questions]}"
        )


# ═══════════════════════════════════════════════
#  The gate that measures what actually ships
# ═══════════════════════════════════════════════

# GOLDEN_QUESTIONS above assert substrings from SAMPLE_TECHNICAL_SPEC, so they measure the
# PIPELINE — chunk, embed, fuse, fetch the parent, fail closed — on a corpus written to be
# measured. They cannot measure the corpus a customer actually reaches: run them against the
# live catalogue and they score 2/10, because four of them ask for facts `docs/catalog/**`
# does not contain at all (a battery life in months, RFID, an anti-tamper alarm, a fan
# wattage) and three ask for wording it does not use ("Neutral Wire: Required" vs the
# catalogue's "Neutral + Line required").
#
# These questions are the other half: every `expected_substring` below was verified present in
# `docs/catalog/**` before being written down, so a failure here means RETRIEVAL missed
# something the corpus really holds — which is the only thing this gate should ever be able to
# say. Categories match the folder each fact lives in, because hybrid_search filters hard on
# metadata->>'category'.
LIVE_CATALOGUE_QUESTIONS = [
    # switches/smart_switches.md — the shared-spec intro
    {"query": "Do the smart switches need a neutral wire?",
     "expected_substring": "Neutral + Line required", "category": "switches"},
    {"query": "What voltage do the Otohom smart switches run on?",
     "expected_substring": "100-240V AC", "category": "switches"},
    {"query": "What are the panel dimensions of the switch?",
     "expected_substring": "108L x 101W x 35H mm", "category": "switches"},
    {"query": "Which series is the retrofit module that avoids rewiring?",
     "expected_substring": "Hider", "category": "switches"},
    # switches/smart_switches.md — the per-product sections, 53-105 characters each. These are
    # the ones that need the SKU phrase alias in search.py::_lexical_term_groups: the product
    # heading is "## 4 SW", the model writes "4SW", and before the alias a 65-character section
    # lost every time to the 1.7kB intro chunk.
    {"query": "What is the max load for the Grande 4SW switch?",
     "expected_substring": "500W per gang", "category": "switches"},
    {"query": "What is the output on the 1SW switch?",
     "expected_substring": "20A (1 gang)", "category": "switches"},
    # security/security_solutions.md
    {"query": "How does the Smart Door Lock Premium open?",
     "expected_substring": "fingerprint, password, card swiping", "category": "security"},
    {"query": "What battery does the Smart Door Lock Base use?",
     "expected_substring": "AA x 4", "category": "security"},
    {"query": "What battery is in the Smart Door Lock Premium?",
     "expected_substring": "3200mAh", "category": "security"},
    {"query": "How big is the video door phone indoor screen?",
     "expected_substring": "7-inch HD touchscreen", "category": "security"},
    # sensors/sensors_smart_controls.md
    {"query": "How far can the motion sensor detect?",
     "expected_substring": "8m (max)", "category": "sensors"},
    {"query": "What is the current range on the single phase energy meter?",
     "expected_substring": "63A", "category": "sensors"},
]


class TestTheLiveCatalogueAnswersRealQuestions:
    """
    Retrieval against the catalogue that is actually ingested — no fixture, no sample document,
    nothing written or deleted.

    Requires the real corpus in pgvector (`python -m src.scripts.ingest_catalog --input-dir
    ./docs/catalog/<cat> …` per folder). Run it explicitly:

        pytest tests/test_rag_pipeline.py::TestTheLiveCatalogueAnswersRealQuestions -v -s

    Why it exists: "retrieval returning rows is not the same as retrieval finding the thing".
    Asymmetric RRF always hands back its top-k, so every one of these questions came back with
    three chunks even when the answer was in none of them — the pass/fail here is the substring,
    never the row count.
    """

    @pytest.fixture(autouse=True)
    async def _dispose_pool_between_tests(self):
        """
        pytest-asyncio gives each test its own event loop, but `src.core.database.engine` is a
        module-level singleton, so the second test in this class would inherit a pool of asyncpg
        connections bound to the first test's closed loop ("RuntimeError: Event loop is closed").
        Disposing after each test drops the pooled connections and the next loop opens its own.
        """
        yield
        await close_db_engine()

    async def test_every_fact_the_catalogue_states_is_retrievable(self):
        embed_client = get_embedding_client()
        passed, failures = 0, []

        for gq in LIVE_CATALOGUE_QUESTIONS:
            query_embedding = await embed_client.aembed_documents([gq["query"]])
            async with async_session_maker() as session:
                results: List[RetrievedContext] = await hybrid_search(
                    session=session,
                    query_embeddings=query_embedding,
                    query_texts=[gq["query"]],
                    category_filter=gq.get("category"),
                )
            if any(gq["expected_substring"].lower() in r.content.lower() for r in results):
                passed += 1
            else:
                failures.append({
                    "query": gq["query"],
                    "expected": gq["expected_substring"],
                    "retrieved": [r.content[:90] for r in results] or ["NO RESULTS"],
                })

        total = len(LIVE_CATALOGUE_QUESTIONS)
        rate = passed / total * 100
        if failures:
            print(f"\n{'=' * 60}\nLive catalogue: {passed}/{total} ({rate:.1f}%)\n{'=' * 60}")
            for f in failures:
                print(f"  MISS  {f['query']}\n        wanted: {f['expected']}\n        got:    {f['retrieved']}")

        # Measured at 12/12 with the SKU phrase alias in place. The bar is the same 80% the
        # pipeline gate uses, so a single regression is visible without being fatal to CI.
        assert rate >= 80.0, (
            f"Live catalogue retrieval: {rate:.1f}% (minimum 80%). "
            f"Missed: {[f['query'] for f in failures]}"
        )

    async def test_a_spaced_sku_heading_is_reachable_by_its_unspaced_name(self):
        """
        The narrow regression the phrase alias bought, asserted on its own.

        "## 4 SW" indexes as the lexemes (4, sw) under the `simple` dictionary; the token "4sw"
        is a single lexeme and matched neither. This question failed before the alias and passes
        after it, so it is worth one test that names the cause — a later chunking or tokenisation
        change would otherwise silently take it away again.
        """
        embed_client = get_embedding_client()
        query = "What is the max load for the Grande 4SW switch?"
        embedding = await embed_client.aembed_documents([query])
        async with async_session_maker() as session:
            results = await hybrid_search(
                session=session, query_embeddings=embedding,
                query_texts=[query], category_filter="switches",
            )
        assert any("500W per gang" in r.content for r in results), (
            "the 4 SW product section was not retrieved for '4SW' — check the SKU phrase alias "
            f"in search.py::_lexical_term_groups. Got: {[r.content[:70] for r in results]}"
        )
