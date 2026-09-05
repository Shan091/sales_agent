"""
Observability + semantic-memory tests. Infra-free: no Postgres, no mem0 server, no Langfuse.

These two subsystems share one dangerous property — they are both *optional and fail-soft*, so a
wiring mistake doesn't crash anything, it just silently produces nothing. A wrong mem0 call
signature means recall returns [] forever and the agent quietly forgets everyone; an unconfigured
Langfuse client means traces vanish while the code still looks instrumented. Both look identical
to "disabled" from the outside, which is exactly why they need tests rather than eyeballing.
"""
import pytest

from src.memory import semantic as sem
from src.core import tracing


# ═══════════════════════════════════════════════
#  Fakes standing in for the two mem0 major lines
# ═══════════════════════════════════════════════

class Mem0V2:
    """mem0 2.x: user_id must arrive inside filters=, and the cap is named top_k. A top-level
    user_id raises ValueError (mem0's own _reject_top_level_entity_params)."""

    def __init__(self, results=None):
        self.results = results if results is not None else [{"memory": "Lives in Kochi"}]
        self.calls = []

    def search(self, query, *, top_k=20, filters=None, **kwargs):
        if "user_id" in kwargs or "limit" in kwargs:
            raise ValueError("Top-level entity parameters {'user_id'} are not supported in search().")
        if not filters or "user_id" not in filters:
            raise ValueError("filters must contain at least one of: user_id, agent_id, run_id.")
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return {"results": self.results}


class Mem0V1:
    """mem0 1.x: the legacy shape — user_id and limit as top-level kwargs, bare list back."""

    def __init__(self, results=None):
        self.results = results if results is not None else ["Owns 6 SW panels"]
        self.calls = []

    def search(self, query, user_id=None, limit=None, **kwargs):
        if "filters" in kwargs or "top_k" in kwargs:
            raise TypeError("search() got an unexpected keyword argument")
        self.calls.append({"query": query, "user_id": user_id, "limit": limit})
        return self.results


def _memory(client) -> sem.SemanticMemory:
    """A SemanticMemory wired to a fake client, bypassing the real constructor (which would need
    MEM0_ENABLED plus a live pgvector). `enabled` is derived from self.memory, so this is enough."""
    m = sem.SemanticMemory.__new__(sem.SemanticMemory)
    m.memory = client
    return m


# ═══════════════════════════════════════════════
#  Recall across both mem0 signatures
# ═══════════════════════════════════════════════

class TestSemanticSearch:

    async def test_modern_signature_is_tried_first(self):
        client = Mem0V2()
        facts = await _memory(client).search("smart switches", user_id="919812345678")
        assert facts == ["Lives in Kochi"]
        # user_id went in via filters, and the cap went in as top_k — not the legacy names.
        assert client.calls[0]["filters"] == {"user_id": "919812345678"}
        assert client.calls[0]["top_k"] == 5

    async def test_falls_back_to_legacy_signature(self):
        """A 1.x client rejects filters=/top_k=; recall must still work rather than silently
        returning nothing (the failure mode this whole test file exists for)."""
        client = Mem0V1()
        facts = await _memory(client).search("door lock", user_id="919812345678")
        assert facts == ["Owns 6 SW panels"]
        assert client.calls[0]["user_id"] == "919812345678"
        assert client.calls[0]["limit"] == 5

    async def test_explicit_limit_overrides_the_setting(self):
        client = Mem0V2()
        await _memory(client).search("q", user_id="919", limit=2)
        assert client.calls[0]["top_k"] == 2

    async def test_thread_id_is_coerced_to_str(self):
        client = Mem0V2()
        await _memory(client).search("q", user_id=919812345678)
        assert client.calls[0]["filters"] == {"user_id": "919812345678"}


class TestSemanticResultShapes:
    """mem0's return shape has moved between releases; every one must flatten to plain strings."""

    async def test_wrapped_results_dict(self):
        client = Mem0V2(results=[{"memory": "3BHK in Kochi"}, {"memory": "Wants curtain motors"}])
        assert await _memory(client).search("q", user_id="919") == ["3BHK in Kochi", "Wants curtain motors"]

    async def test_bare_list_of_dicts(self):
        client = Mem0V1(results=[{"memory": "Prefers black panels"}])
        assert await _memory(client).search("q", user_id="919") == ["Prefers black panels"]

    async def test_alternate_dict_keys(self):
        client = Mem0V2(results=[{"text": "from text key"}, {"data": "from data key"}])
        assert await _memory(client).search("q", user_id="919") == ["from text key", "from data key"]

    async def test_plain_strings(self):
        client = Mem0V1(results=["already a string"])
        assert await _memory(client).search("q", user_id="919") == ["already a string"]

    async def test_blank_and_null_facts_are_dropped(self):
        client = Mem0V2(results=[{"memory": ""}, {"memory": "   "}, {"memory": None}, {"memory": "real"}])
        assert await _memory(client).search("q", user_id="919") == ["real"]


class TestSemanticFailSoft:
    """Every failure path must degrade to "no memory" — never raise into the turn."""

    async def test_disabled_returns_empty(self):
        assert await _memory(None).search("q", user_id="919") == []

    async def test_blank_query_short_circuits(self):
        client = Mem0V2()
        assert await _memory(client).search("   ", user_id="919") == []
        assert client.calls == []

    async def test_client_explosion_is_swallowed(self):
        class Boom:
            def search(self, *a, **kw):
                raise RuntimeError("mem0 is down")

        assert await _memory(Boom()).search("q", user_id="919") == []

    async def test_store_failure_is_swallowed(self):
        class Boom:
            def add(self, *a, **kw):
                raise RuntimeError("mem0 is down")

        await _memory(Boom()).extract_and_store("customer: hi", user_id="919")  # must not raise

    async def test_reset_failure_is_swallowed(self):
        class Boom:
            def delete_all(self, *a, **kw):
                raise RuntimeError("mem0 is down")

        await _memory(Boom()).reset(user_id="919")  # must not raise

    async def test_store_skips_blank_text(self):
        class Recorder:
            def __init__(self):
                self.calls = []

            def add(self, text, **kw):
                self.calls.append(text)

        rec = Recorder()
        await _memory(rec).extract_and_store("   ", user_id="919")
        assert rec.calls == []

    async def test_delete_all_alias_delegates_to_reset(self):
        class Recorder:
            def __init__(self):
                self.deleted = []

            def delete_all(self, user_id=None):
                self.deleted.append(user_id)

        rec = Recorder()
        await _memory(rec).delete_all(user_id=919)
        assert rec.deleted == ["919"]


# ═══════════════════════════════════════════════
#  {memory_block} prompt fragment
# ═══════════════════════════════════════════════

class TestMemoryBlock:

    def test_no_facts_is_empty_string(self):
        """Must be "" not a header with nothing under it — the prompt concatenates it blindly,
        exactly like {rag_context_block}."""
        assert sem.format_memory_block([]) == ""

    def test_facts_are_rendered_as_a_bullet_list(self):
        block = sem.format_memory_block(["Lives in Kochi", "Owns 6 SW panels"])
        assert "- Lives in Kochi" in block
        assert "- Owns 6 SW panels" in block

    def test_block_forbids_reciting_and_defers_to_today(self):
        """The two behavioural rules that keep recall from feeling like being read your own file."""
        block = sem.format_memory_block(["Lives in Kochi"])
        assert "Do NOT recite this list back" in block
        assert "what they tell you today wins" in block

    def test_truncates_to_the_prompt_budget(self):
        block = sem.format_memory_block([f"fact {i}" for i in range(20)])
        assert block.count("\n- ") == 5


# ═══════════════════════════════════════════════
#  Langfuse config + request ids
# ═══════════════════════════════════════════════

class TestRequestId:

    def test_is_short_and_log_friendly(self):
        rid = tracing.new_request_id()
        assert len(rid) == 12 and rid.isalnum()

    def test_is_unique_per_turn(self):
        assert len({tracing.new_request_id() for _ in range(200)}) == 200


class TestLangfuseConfig:
    """With no keys set (the default), config must still be a valid, complete LangChain config —
    callers splat it unconditionally and must never have to branch on whether tracing is on."""

    def test_callbacks_empty_when_disabled(self, monkeypatch):
        monkeypatch.setattr(tracing.settings, "LANGFUSE_PUBLIC_KEY", "", raising=False)
        monkeypatch.setattr(tracing.settings, "LANGFUSE_SECRET_KEY", "", raising=False)
        monkeypatch.setattr(tracing, "_handler_built", False, raising=False)
        monkeypatch.setattr(tracing, "_handler", None, raising=False)
        assert tracing.langfuse_config("919", "abc123")["callbacks"] == []

    def test_session_and_user_are_the_whatsapp_number(self):
        """These two reserved keys are what group a customer's turns into ONE session in the UI;
        without them every turn is an orphan trace."""
        meta = tracing.langfuse_config("919812345678", "abc123")["metadata"]
        assert meta["langfuse_session_id"] == "919812345678"
        assert meta["langfuse_user_id"] == "919812345678"

    def test_metadata_carries_the_audit_fields(self):
        meta = tracing.langfuse_config("919", "req42")["metadata"]
        assert meta["thread_id"] == "919"
        assert meta["request_id"] == "req42"
        assert "agent_full_autonomy" in meta and "environment" in meta

    def test_extra_metadata_is_merged(self):
        meta = tracing.langfuse_config("919", "r", extra_metadata={"msg_type": "interactive"})["metadata"]
        assert meta["msg_type"] == "interactive"
        assert meta["thread_id"] == "919"  # base fields survive the merge

    def test_custom_tags_reach_both_places(self):
        cfg = tracing.langfuse_config("919", "r", tags=["payment"])
        assert cfg["metadata"]["langfuse_tags"] == ["payment"]
        assert cfg["tags"] == ["payment"]

    def test_run_name_identifies_the_turn(self):
        assert "919" in tracing.langfuse_config("919", "r")["run_name"]

    def test_does_not_clobber_the_graph_thread_config(self):
        """The graph needs its own configurable.thread_id to find the checkpoint, so this fragment
        must be merge-safe rather than a whole config."""
        cfg = {"configurable": {"thread_id": "919"}, **tracing.langfuse_config("919", "r")}
        assert cfg["configurable"]["thread_id"] == "919"
        assert "callbacks" in cfg

    def test_flush_is_safe_when_nothing_is_configured(self):
        tracing.flush_langfuse()  # must not raise


class TestPgvectorConfig:
    """The old prototype hardcoded otohom/user/pass/localhost, which silently pointed at nothing
    inside Docker. Connection details must come from the one DATABASE_URL the app already uses.

    _pgvector_configs returns CANDIDATE shapes, most-current first: a `connection_string` form
    (preferred — it carries the database name unambiguously) then the legacy split-field form.
    Both must resolve to the same database; the split-field one must never omit `dbname`, or
    mem0 would connect to the driver's default DB and write memories somewhere nobody looks.
    """

    def _split_field(self, monkeypatch, url):
        monkeypatch.setattr(sem.settings, "DATABASE_URL", url, raising=False)
        candidates = sem._pgvector_configs()
        assert "connection_string" in candidates[0], "connection_string shape must be tried first"
        return candidates[0], candidates[1]

    def test_derives_from_database_url(self, monkeypatch):
        first, split = self._split_field(
            monkeypatch, "postgresql+asyncpg://otohom:s3cret@postgres:5432/otohom"
        )
        # The preferred shape hands mem0 a psycopg-speaking DSN (the '+asyncpg' driver suffix
        # stripped), so the database name travels with it.
        assert first["connection_string"] == "postgresql://otohom:s3cret@postgres:5432/otohom"
        assert split["host"] == "postgres"
        assert split["user"] == "otohom"
        assert split["password"] == "s3cret"
        assert split["dbname"] == "otohom"
        assert split["port"] == 5432

    def test_percent_encoded_password_is_decoded(self, monkeypatch):
        _, split = self._split_field(
            monkeypatch, "postgresql+asyncpg://otohom:p%40ss%2Fword@postgres:5432/otohom"
        )
        assert split["password"] == "p@ss/word"

    def test_own_collection_and_matching_dims(self, monkeypatch):
        """
        Sharing a table with the RAG pipeline, or disagreeing with the embedder on dimensions,
        would corrupt retrieval — mem0's OpenAI default is 1536-dim, its small local embedder is
        384, and the RAG stack is 1024. Every candidate shape must name mem0's own collection and
        mem0's own width, never the RAG one.
        """
        for cfg in sem._pgvector_configs():
            assert cfg["collection_name"] == "otohom_semantic_memory"
            assert cfg["embedding_model_dims"] == sem.settings.MEM0_EMBEDDING_DIMS

    def test_mem0_does_not_borrow_the_rag_embedding_model(self):
        """mem0 builds its OWN embedder instance, so pointing it at bge-m3 loaded a second ~2.2GB
        copy into the same worker process to match one-line facts."""
        assert sem.settings.MEM0_EMBEDDING_MODEL != sem.settings.EMBEDDING_MODEL
        assert sem.settings.MEM0_EMBEDDING_DIMS != sem.settings.EMBEDDING_DIMENSIONS


class TestSingleton:

    def test_returns_none_when_disabled(self, monkeypatch):
        monkeypatch.setattr(sem, "_instance", None, raising=False)
        monkeypatch.setattr(sem, "_instance_built", False, raising=False)
        monkeypatch.setattr(sem.settings, "MEM0_ENABLED", False, raising=False)
        assert sem.get_semantic_memory() is None

    def test_construction_is_attempted_once(self, monkeypatch):
        """mem0's constructor builds an LLM + embedder client, so a per-turn rebuild would be a
        real latency bug — and a per-turn *failed* rebuild would spam the logs."""
        monkeypatch.setattr(sem, "_instance", None, raising=False)
        monkeypatch.setattr(sem, "_instance_built", False, raising=False)
        calls = []

        class Counting(sem.SemanticMemory):
            def __init__(self):
                calls.append(1)
                self.memory = None

        monkeypatch.setattr(sem, "SemanticMemory", Counting)
        sem.get_semantic_memory()
        sem.get_semantic_memory()
        assert len(calls) == 1


class TestTheTypingDotsShowWhileTheCustomerWaits:
    """
    Two failure modes, opposite directions, both reported as "the typing indicator doesn't work".

    An INSTANT turn used to post dots anyway: the six no-LLM paths hand their reply to Meta ~80ms in,
    the dots and the reply then race each other there, and when the reply wins the dots are drawn
    AFTER the answer and hold for ~25s with nothing coming. A LONG turn had the opposite problem —
    Meta dismisses the indicator ~25s after the POST and offers no way to extend it, so the dots went
    out once and the rest of the turn was silent.

    Neither can be tested against Meta from here, so what is asserted is what this code controls:
    when it posts, when it does not, and that it never costs the turn.
    """

    @staticmethod
    def _harness(monkeypatch):
        import asyncio
        from src.tasks import processing

        posts = []

        class _WA:
            async def send_typing_indicator(self, thread_id, wamid, received_at=None, repost=False):
                posts.append("repost" if repost else "first")

        monkeypatch.setattr(processing, "_whatsapp", _WA())
        monkeypatch.setattr(processing, "_TYPING_GRACE_SECONDS", 0.05, raising=False)
        monkeypatch.setattr(processing, "_TYPING_REPOST_EVERY_SECONDS", 0.05, raising=False)
        return processing, posts, asyncio

    async def test_an_instant_answer_gets_no_dots_at_all(self, monkeypatch):
        processing, posts, asyncio = self._harness(monkeypatch)
        stop, acquired, replied = asyncio.Event(), asyncio.Event(), asyncio.Event()
        acquired.set()
        replied.set()          # the turn was already sending before the grace window was up
        await processing.typing_heartbeat("91", "wamid.X", stop, acquired, None, replied)
        assert posts == [], "dots drawn after the answer are worse than no dots"

    async def test_a_turn_that_ends_inside_the_grace_window_posts_nothing_either(self, monkeypatch):
        processing, posts, asyncio = self._harness(monkeypatch)
        stop, acquired = asyncio.Event(), asyncio.Event()
        stop.set()
        await processing.typing_heartbeat("91", "wamid.X", stop, acquired, None, asyncio.Event())
        assert posts == []

    async def test_a_turn_the_customer_waits_on_gets_the_dots(self, monkeypatch):
        processing, posts, asyncio = self._harness(monkeypatch)
        stop, acquired, replied = asyncio.Event(), asyncio.Event(), asyncio.Event()
        acquired.set()
        task = asyncio.create_task(
            processing.typing_heartbeat("91", "wamid.X", stop, acquired, None, replied)
        )
        await asyncio.sleep(0.12)
        replied.set()
        stop.set()
        await task
        assert posts and posts[0] == "first"

    async def test_a_long_turn_keeps_them_alive_up_to_the_cap(self, monkeypatch):
        processing, posts, asyncio = self._harness(monkeypatch)
        stop, acquired = asyncio.Event(), asyncio.Event()
        acquired.set()
        # Never sets `stop`, so the loop runs to its own ceiling rather than forever.
        await processing.typing_heartbeat("91", "wamid.X", stop, acquired, None, asyncio.Event())
        assert posts[0] == "first"
        assert posts.count("repost") == processing._TYPING_REPOST_MAX

    async def test_the_dots_never_cost_the_turn(self, monkeypatch):
        processing, posts, asyncio = self._harness(monkeypatch)

        class _Broken:
            async def send_typing_indicator(self, *_a, **_k):
                raise RuntimeError("Meta said no")

        monkeypatch.setattr(processing, "_whatsapp", _Broken())
        stop, acquired = asyncio.Event(), asyncio.Event()
        acquired.set()
        stop.set()
        await processing.typing_heartbeat("91", "wamid.X", stop, acquired, None, None)

    async def test_no_wamid_means_no_request_rather_than_a_400(self, monkeypatch):
        # Internally-triggered jobs (payment confirmations) have no inbound message to key it to.
        processing, posts, asyncio = self._harness(monkeypatch)
        stop, acquired = asyncio.Event(), asyncio.Event()
        acquired.set()
        await processing.typing_heartbeat("91", None, stop, acquired, None, None)
        assert posts.count("repost") == 0
