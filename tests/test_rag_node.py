"""
The retrieval layer's infra-free logic: node behaviour, query-term building, and chunking. No DB /
real LLM / embedding model required — the fast model, embedding client, hybrid search, and DB
session are all monkeypatched, and the term builder and chunker are pure.

Contract under test (A+B fallback): when query condensation fails, the node MUST NOT escalate
to a human. Instead it embeds the raw last user message and runs an unfiltered hybrid search;
if that returns nothing it proceeds with empty context (the archetype node answers warmly from
OTOHOM_OVERVIEW). Human handoff is never set from inside the RAG node.

The two classes at the end cover `search.py::_lexical_term_groups` and `embeddings.py`'s parent
merging. They live here rather than in test_rag_pipeline.py because that file is excluded from the
default suite for needing live infra, and a pure function whose test never runs is not tested.
"""
import pytest
from langchain_core.messages import HumanMessage, AIMessage
import src.graph.nodes.rag as rag


class _FakeEmbedClient:
    async def aembed_documents(self, texts):
        return [[0.0] * 8 for _ in texts]


class _FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Result:
    def __init__(self, content):
        self.content = content


def _patch_common(monkeypatch):
    """Stub out the fast LLM, embedding client, and DB session (no infra)."""
    monkeypatch.setattr(rag.LLMFactory, "get_llm", staticmethod(lambda **kwargs: object()))
    monkeypatch.setattr(rag, "get_embedding_client", lambda: _FakeEmbedClient())
    monkeypatch.setattr(rag, "async_session_maker", lambda: _FakeSession())


def _patch_condensation(monkeypatch, expansion):
    async def _run(*args, **kwargs):
        return expansion
    monkeypatch.setattr(rag, "execute_vendor_agnostic_node", _run)


# ─── Happy path: condensation succeeds ─────────────────────────────────────

@pytest.mark.asyncio
async def test_condensation_success_uses_expansion_and_filters(monkeypatch):
    _patch_common(monkeypatch)

    class _Expansion:
        semantic_query = "smart switch specifications"
        keyword_query = "Grande 4SW"
        symptom_query = "switch overheating under load"
        product_name = "4SW"
        category_filter = "switches"
        doc_type_filter = None

    _patch_condensation(monkeypatch, _Expansion())

    captured = {}

    async def _search(session, query_embeddings, query_texts, category_filter=None, doc_type_filter=None):
        captured["query_texts"] = query_texts
        captured["category_filter"] = category_filter
        captured["doc_type_filter"] = doc_type_filter
        return [_Result("chunk A"), _Result("chunk B")]

    monkeypatch.setattr(rag, "hybrid_search", _search)

    state = {"messages": [HumanMessage(content="tell me about the 4SW")]}
    result = await rag.node_retrieve_technical_context(state)

    assert not result.get("requires_human_handoff")
    assert result["context_chunks"] == ["chunk A", "chunk B"]
    assert result["rag_query"] == "Grande 4SW"  # keyword_query drives rag_query
    # All 3 expansion queries are searched, with the hard filters applied.
    assert captured["query_texts"] == [
        "smart switch specifications", "Grande 4SW", "switch overheating under load"
    ]
    assert captured["category_filter"] == "switches"
    assert captured["doc_type_filter"] is None


# ─── Fallback path: condensation fails ─────────────────────────────────────

@pytest.mark.asyncio
async def test_condensation_failure_falls_back_to_raw_embed(monkeypatch):
    _patch_common(monkeypatch)
    _patch_condensation(monkeypatch, None)  # condensation returns None

    captured = {}

    async def _search(session, query_embeddings, query_texts, category_filter=None, doc_type_filter=None):
        captured["query_texts"] = query_texts
        captured["category_filter"] = category_filter
        captured["doc_type_filter"] = doc_type_filter
        return [_Result("Grande 4SW wiring: line + load.")]

    monkeypatch.setattr(rag, "hybrid_search", _search)

    state = {"messages": [HumanMessage(content="how do I wire the Grande 4SW?")]}
    result = await rag.node_retrieve_technical_context(state)

    # No escalation — the raw-message fallback handled it.
    assert not result.get("requires_human_handoff")
    assert result["context_chunks"] == ["Grande 4SW wiring: line + load."]
    # Raw message is used verbatim as the single, unfiltered query.
    assert result["rag_query"] == "how do I wire the Grande 4SW?"
    assert captured["query_texts"] == ["how do I wire the Grande 4SW?"]
    assert captured["category_filter"] is None
    assert captured["doc_type_filter"] is None


@pytest.mark.asyncio
async def test_condensation_failure_empty_search_no_handoff(monkeypatch):
    _patch_common(monkeypatch)
    _patch_condensation(monkeypatch, None)

    async def _search(**kwargs):
        return []

    monkeypatch.setattr(rag, "hybrid_search", _search)

    state = {"messages": [HumanMessage(content="just tell me about smart homes")]}
    result = await rag.node_retrieve_technical_context(state)

    # Even with zero results after fallback, never escalate — proceed with empty context.
    assert not result.get("requires_human_handoff")
    assert result["context_chunks"] == []
    assert result["rag_query"] == "just tell me about smart homes"


@pytest.mark.asyncio
async def test_condensation_failure_no_user_message(monkeypatch):
    _patch_common(monkeypatch)
    _patch_condensation(monkeypatch, None)

    async def _search(**kwargs):
        raise AssertionError("hybrid_search must not run when there is no user message")

    monkeypatch.setattr(rag, "hybrid_search", _search)

    # Latest turn is the AI's -> nothing inbound to embed -> empty context, still no handoff.
    state = {"messages": [AIMessage(content="Hi! How can I help?")]}
    result = await rag.node_retrieve_technical_context(state)

    assert not result.get("requires_human_handoff")
    assert result["context_chunks"] == []
    assert result.get("rag_query") is None


@pytest.mark.asyncio
async def test_a_named_product_with_no_results_flags_specs_unavailable(monkeypatch):
    """
    Zero results for a BROAD question is fine — the archetype answers from the overview. Zero
    results for a product the customer NAMED is different: an LLM turn there answers from
    pretraining, so it is flagged for code to handle instead.
    """
    _patch_common(monkeypatch)

    class _Expansion:
        semantic_query = "premium door lock features"
        keyword_query = "Smart Door Lock Premium"
        symptom_query = "door lock unlock methods"
        product_name = "Smart Door Lock Premium"
        category_filter = None
        doc_type_filter = None

    _patch_condensation(monkeypatch, _Expansion())

    async def _search(**kwargs):
        return []

    monkeypatch.setattr(rag, "hybrid_search", _search)

    result = await rag.node_retrieve_technical_context(
        {"messages": [HumanMessage(content="what are the features of the premium door lock?")]}
    )
    assert result["specs_unavailable"] is True
    assert result["context_chunks"] == []
    # Still never escalates from inside the RAG node.
    assert not result.get("requires_human_handoff")


@pytest.mark.asyncio
async def test_a_broad_question_with_no_results_does_not_flag_it(monkeypatch):
    _patch_common(monkeypatch)

    class _Expansion:
        semantic_query = "smart home ideas"
        keyword_query = "smart home"
        symptom_query = "home automation"
        product_name = None          # nothing specific was named
        category_filter = None
        doc_type_filter = None

    _patch_condensation(monkeypatch, _Expansion())

    async def _search(**kwargs):
        return []

    monkeypatch.setattr(rag, "hybrid_search", _search)

    result = await rag.node_retrieve_technical_context(
        {"messages": [HumanMessage(content="what kind of things can you do?")]}
    )
    assert result["specs_unavailable"] is False


@pytest.mark.asyncio
async def test_successful_retrieval_clears_the_flag(monkeypatch):
    _patch_common(monkeypatch)

    class _Expansion:
        semantic_query = "a"
        keyword_query = "b"
        symptom_query = "c"
        product_name = "Smart Door Lock Premium"
        category_filter = None
        doc_type_filter = None

    _patch_condensation(monkeypatch, _Expansion())

    async def _search(**kwargs):
        return [_Result("Premium lock: five unlock methods")]

    monkeypatch.setattr(rag, "hybrid_search", _search)

    result = await rag.node_retrieve_technical_context({"messages": [HumanMessage(content="specs?")]})
    assert result["specs_unavailable"] is False
    assert result["context_chunks"]


@pytest.mark.asyncio
async def test_the_archetype_refuses_to_invent_when_the_flag_is_set():
    """The whole point: no LLM call, and no product detail in the reply."""
    from src.graph.nodes.sales import node_high_intent

    out = await node_high_intent({
        "messages": [("user", "tell me the features of the premium door lock")],
        "specs_unavailable": True,
        "context_chunks": [],
    })
    text = out["messages"][0].content.lower()
    assert "confirmed" in text
    for invented in ("fingerprint", "aluminium", "aluminum", "rose gold", "anti-pry", "doorbell"):
        assert invented not in text
    # And it still moves the conversation on rather than dead-ending.
    assert out["messages"][0].response_metadata["options"]
    assert out["specs_unavailable"] is False


@pytest.mark.asyncio
async def test_raw_fallback_accepts_an_uncoerced_tuple_message(monkeypatch):
    """
    The worker seeds a turn as ("user", text). The fallback must read that shape, not just a
    HumanMessage, and must preserve the original casing it embeds.
    """
    _patch_common(monkeypatch)
    _patch_condensation(monkeypatch, None)

    captured = {}

    async def _search(session, query_embeddings, query_texts, category_filter=None, doc_type_filter=None):
        captured["query_texts"] = query_texts
        return [_Result("chunk")]

    monkeypatch.setattr(rag, "hybrid_search", _search)

    state = {"messages": [("user", "How do I wire the Grande 4SW?")]}
    result = await rag.node_retrieve_technical_context(state)

    assert captured["query_texts"] == ["How do I wire the Grande 4SW?"]
    assert result["rag_query"] == "How do I wire the Grande 4SW?"


class TestASkuIsReachableByEitherSpelling:
    """
    `search.py::_lexical_term_groups` — pure, so it belongs in the infra-free suite even though
    the retrieval it feeds does not.

    The corpus writes "4 SW"; the customer and `QueryExpansion` write "4SW". The tsvector uses the
    `simple` dictionary, which neither stems nor strips whitespace, so "4 SW" indexes as the
    lexemes (4, sw) while "4sw" is one lexeme — measured against the live chunk holding
    "less than 500W per gang", `to_tsquery('simple','4sw')` was FALSE and
    `to_tsquery('simple','4<->sw')` was TRUE. The exact-SKU hit the asymmetric k=30 boost exists to
    surface never fired for the spelling the model most often produces, and a 65-character product
    section lost every time to a 1.7kB document intro.
    """

    def test_an_unspaced_sku_also_searches_for_the_spaced_form(self):
        from src.rag.search import _lexical_term_groups
        groups = _lexical_term_groups("What is the max load for the Grande 4SW switch?")
        sku = [g for g in groups if g[0] == "4sw"]
        assert sku == [["4sw", "4<->sw"]], groups

    def test_a_spaced_sku_also_searches_for_the_unspaced_form(self):
        from src.rag.search import _lexical_term_groups
        groups = _lexical_term_groups("tell me about the 4 SW panel")
        assert ["4sw", "4<->sw"] in groups, groups

    def test_a_bare_unit_never_becomes_a_term_of_its_own(self):
        # "sw" alone matches every switch section in the file, which is the opposite of the
        # precision the SKU alias exists to buy.
        from src.rag.search import _lexical_term_groups
        flat = [v for group in _lexical_term_groups("the 4 SW panel") for v in group]
        assert "sw" not in flat
        assert "4" not in flat

    def test_a_real_word_after_a_digit_keeps_its_own_concept(self):
        # "2 Way 2 SW" is a real product heading: the pair is worth a concept, and so is "way".
        from src.rag.search import _lexical_term_groups
        groups = _lexical_term_groups("2 Way 2 SW for the staircase")
        assert ["2way", "2<->way"] in groups
        assert ["way"] in groups
        assert ["2sw", "2<->sw"] in groups

    def test_two_spellings_of_one_sku_count_as_one_concept(self):
        """
        The safety property behind MIN_LEXICAL_TERMS_FOR_RESCUE.

        `matched_terms` counts GROUPS, so a chunk that happens to contain both spellings of one
        product name still scores 1 — otherwise a single product mention could rescue an
        off-domain query that the similarity threshold had already refused.
        """
        from src.rag.search import _lexical_term_groups
        assert len(_lexical_term_groups("4SW")) == 1
        assert len(_lexical_term_groups("4 SW")) == 1

    def test_question_words_and_single_characters_are_dropped(self):
        from src.rag.search import _lexical_term_groups
        assert _lexical_term_groups("what is it") == []
        assert _lexical_term_groups("a x") == []

    def test_repeated_terms_appear_once(self):
        from src.rag.search import _lexical_term_groups
        groups = _lexical_term_groups("switch switch SWITCH")
        assert groups == [["switch"]]

    def test_no_tsquery_operator_can_arrive_from_user_text(self):
        # Raw user text reaches to_tsquery, so anything but alphanumerics must be stripped before
        # it gets there — the `<->` in a group is code's, never the customer's.
        from src.rag.search import _lexical_term_groups, _group_tsquery
        groups = _lexical_term_groups("switch & (bogus | !injected) <-> 4SW")
        rendered = " | ".join(_group_tsquery(g) for g in groups)
        for forbidden in ("&", "!", "(bogus", "injected)"):
            assert forbidden not in rendered, rendered
        assert "4<->sw" in rendered

    def test_a_single_variant_group_renders_without_brackets(self):
        from src.rag.search import _group_tsquery
        assert _group_tsquery(["switch"]) == "switch"
        assert _group_tsquery(["4sw", "4<->sw"]) == "(4sw | 4<->sw)"


class TestTinySectionsShareAParent:
    """
    `embeddings.py` — the chunker is pure, so it belongs in the infra-free suite too.

    The live switches document has ten product sections between 53 and 105 characters ("## 4 SW
    (4-gang smart switch)" plus one bullet). Each used to become its own parent, and both halves of
    that went wrong: dense retrieval cannot separate ten near-identical tiny chunks, so a question
    about the 4 SW returned the 1 SW and 2 SW parents with the answer in neither; and a 65-character
    parent carries no context even when it is the right one, because every shared spec lives in the
    document intro. Merging consecutive short sections fixes both — whichever child wins, the parent
    holds its neighbours — while children stay per-section so the embedding targets do not change.
    """

    # Shaped like the live document: a substantial shared-spec intro (the real one is 1761
    # characters, well over RAG_MIN_PARENT_CHARS) followed by product sections of 40-70.
    SWITCHES = (
        "# Smart Switches\n\n"
        "All Otohom smart switch panels share these characteristics: premium glass touch panel, "
        "machine-cut aluminum finish, retrofittable with existing wiring so no rewiring is needed, "
        "and they work with Alexa and Google Home voice assistants.\n\n"
        "- Connectivity: Wi-Fi 2.4 GHz / 5 GHz, Zigbee\n"
        "- Voltage: 100-240V AC, 50/60 Hz\n"
        "- Panel dimensions: 108L x 101W x 35H mm\n"
        "- Device working temperature: -10 to 60 C\n"
        "- Wiring: Neutral + Line required\n"
        "- Available frame colors: Golden, Black\n\n"
        "## 1 SW (1-gang)\n\n- Output: 20A (1 gang)\n\n"
        "## 2 SW (2-gang)\n\n- Power: 5A (2 gangs)\n\n"
        "## 4 SW (4-gang)\n\n- Output: less than 500W per gang\n\n"
        "## 6 SW (6-gang)\n\n- Power: 5A x 4 + 16A x 1\n"
    )

    def _split(self, text=None):
        from src.rag.embeddings import split_markdown_to_parent_child
        return split_markdown_to_parent_child(
            text if text is not None else self.SWITCHES,
            {"doc_type": "TECHNICAL_SPEC", "category": "switches", "source_file": "s.md"},
        )

    def test_short_product_sections_end_up_in_one_parent(self):
        chunks = self._split()
        parents = [c for c in chunks if c.is_parent]
        holding = [p for p in parents if "500W per gang" in p.content]
        assert len(holding) == 1
        # The neighbours the dense retriever confuses it with are in the same parent, so picking
        # the wrong one of them still returns the right answer.
        for sibling in ("20A (1 gang)", "5A (2 gangs)", "5A x 4 + 16A x 1"):
            assert sibling in holding[0].content, sibling

    def test_children_stay_per_section_so_the_embedding_targets_do_not_change(self):
        chunks = self._split()
        children = [c for c in chunks if not c.is_parent]
        # One child per product section, each naming its own product — not one child covering four.
        precise = [c for c in children if "500W per gang" in c.content]
        assert len(precise) == 1
        assert "4 SW" in precise[0].content
        assert "20A (1 gang)" not in precise[0].content

    def test_every_child_is_still_a_substring_of_its_parent(self):
        # The documented guarantee: a child must never match on text absent from the parent that
        # gets injected into the prompt. Merging parents is exactly the change that could break it.
        for chunk in self._split():
            if not chunk.is_parent:
                assert chunk.content in chunk.parent_content_ref, chunk.content[:60]

    def test_a_substantial_section_is_left_alone(self):
        chunks = self._split()
        parents = [c for c in chunks if c.is_parent]
        intro = [p for p in parents if "Panel dimensions" in p.content]
        assert len(intro) == 1
        assert "4 SW" not in intro[0].content, "the intro must not absorb the product table"

    def test_a_merged_parent_claims_no_single_product_series(self):
        # Children keep their own product_series; a parent covering 1 SW through 6 SW has none, and
        # inventing one from the first section would be a lie any metadata filter would act on.
        chunks = self._split()
        merged = next(c for c in chunks if c.is_parent and "500W per gang" in c.content)
        assert merged.metadata.get("product_series") is None
        assert merged.metadata["category"] == "switches"
        child = next(c for c in chunks if not c.is_parent and "500W per gang" in c.content)
        assert "4 SW" in (child.metadata.get("product_series") or "")

    def test_a_group_never_crosses_a_top_level_heading(self):
        chunks = self._split(
            "# Switches\n\n## 1 SW\n\n- Output: 20A\n\n"
            "# Security\n\n## Door Lock\n\n- Battery: AA x 4\n"
        )
        parents = [c for c in chunks if c.is_parent]
        assert not any(
            "20A" in p.content and "AA x 4" in p.content for p in parents
        ), "a switch and a lock must never share a parent"

    def test_hashes_stay_unique_after_merging(self):
        chunks = self._split()
        hashes = [c.source_hash for c in chunks]
        assert len(hashes) == len(set(hashes))
