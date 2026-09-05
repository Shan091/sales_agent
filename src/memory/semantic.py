"""
Semantic (long-term) memory via mem0 — the agent's THIRD memory.

The three are deliberately separate:
  * EPISODIC — LangGraph checkpoints (Postgres, keyed by thread_id). The verbatim turns of a
    conversation. Great for "what did we just say", useless for "who is this person".
  * SEMANTIC — this module. Durable, distilled FACTS about a person ("lives in Kochi",
    "3BHK", "worried about the front door", "bought 6 SW panels") that survive across
    sessions and outlive the transcript.
  * RECORDS — PaymentOrder / CRMLead rows. The audit trail.

Everything here is best-effort and NON-FATAL by design. mem0 runs its own LLM extraction pass,
so it is the slowest and least reliable thing in a turn; a mem0 outage must never cost a sale.
Every method swallows its exceptions and degrades to "no memory" rather than raising, and the
whole module is gated behind settings.MEM0_ENABLED (default False) so a bare checkout has no
extra moving parts. `from mem0 import Memory` is imported LAZILY inside the constructor for the
same reason — an uninstalled mem0ai must not break `import src.memory.semantic`.

The mem0 client is synchronous, so every call is pushed to a worker thread with
asyncio.to_thread to keep the event loop free.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, unquote

from config.settings import settings

logger = logging.getLogger(__name__)

# How many distilled facts to inject into a prompt. Kept small on purpose: memory is there to
# personalise, not to refill the context window.
_DEFAULT_LIMIT = 5


def _pgvector_configs() -> List[Dict[str, Any]]:
    """
    Candidate mem0 pgvector configs, most-current shape first.

    Both are derived from the single DATABASE_URL the rest of the app uses rather than a second
    hardcoded copy of the credentials (the old prototype hardcoded otohom/user/pass/localhost,
    which silently pointed at nothing under docker-compose).

    Why two: mem0's Python pgvector store resolves a connection as pool -> connection_string ->
    individual host/port/user/password fields, and `dbname` is documented as a TypeScript-only
    key. Passing split fields WITHOUT a database name is the dangerous case — it would connect
    to the driver's default database and quietly write memories somewhere nobody looks. So we
    prefer `connection_string` (which carries the database name unambiguously) and only fall
    back to the older split-field + `dbname` shape if this mem0 build rejects it.
    """
    parsed = urlparse(settings.psycopg_dsn)
    common = {
        # Own table — must never share document_chunks with the RAG pipeline.
        "collection_name": "otohom_semantic_memory",
        # mem0's own embedder width, independent of the RAG model — see MEM0_EMBEDDING_MODEL.
        "embedding_model_dims": settings.MEM0_EMBEDDING_DIMS,
    }
    return [
        {"connection_string": settings.psycopg_dsn, **common},
        {
            "dbname": (parsed.path or "/otohom").lstrip("/") or "otohom",
            "user": unquote(parsed.username or "user"),
            "password": unquote(parsed.password or ""),
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 5432,
            **common,
        },
    ]


class SemanticMemory:
    """
    Thin async facade over mem0. Construct once per worker (see get_semantic_memory) — the
    constructor may build an LLM/embedder client, so it is not free.
    """

    def __init__(self):
        self.memory = None
        if not settings.MEM0_ENABLED:
            logger.info("MEM0_ENABLED is false; semantic memory is a no-op.")
            return
        try:
            from mem0 import Memory  # lazy: an uninstalled mem0ai must not break imports

            # Everything except the vector store is fixed; only the pgvector connection shape
            # varies between mem0 builds (see _pgvector_configs).
            base_config = {
                # Point mem0's own extraction LLM at the SAME OpenRouter endpoint the graph
                # uses, so there is one API key and one vendor to reason about.
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": settings.FAST_MODEL,
                        "api_key": settings.OPENAI_API_KEY,
                        "openai_base_url": settings.OPENAI_API_BASE,
                        "temperature": 0.0,
                    },
                },
                # Embedder MUST be set explicitly. mem0's default is OpenAI
                # text-embedding-3-small (1536 dims), which would break twice here: our
                # OPENAI_API_BASE is OpenRouter, which serves chat completions but NOT an
                # embeddings endpoint, and 1536 dims contradicts the embedding_model_dims we
                # hand pgvector — so the table and the vectors would disagree.
                #
                # A SMALL local model on purpose, not the RAG stack's bge-m3. mem0 builds its own
                # embedder instance, so sharing bge-m3 meant a second ~2.2GB copy in the same worker
                # process and ~12s of extra boot, to match one-line facts like "lives in Kochi" —
                # work a 90MB model does perfectly well.
                "embedder": {
                    "provider": "huggingface",
                    "config": {
                        "model": settings.MEM0_EMBEDDING_MODEL,
                        "embedding_dims": settings.MEM0_EMBEDDING_DIMS,
                    },
                },
            }

            last_err: Optional[Exception] = None
            for vs_config in _pgvector_configs():
                try:
                    self.memory = Memory.from_config({
                        "vector_store": {"provider": "pgvector", "config": vs_config},
                        **base_config,
                    })
                    logger.info(
                        "Semantic memory (mem0 + pgvector) initialized "
                        f"({'connection_string' if 'connection_string' in vs_config else 'split-field'} config)."
                    )
                    break
                except Exception as attempt_err:
                    last_err = attempt_err
                    logger.debug(f"mem0 pgvector config rejected, trying next shape: {attempt_err}")
            if self.memory is None and last_err is not None:
                raise last_err
        except Exception as e:
            logger.error(f"Failed to initialize mem0 (semantic memory disabled): {e}")
            self.memory = None

    @property
    def enabled(self) -> bool:
        return self.memory is not None

    async def extract_and_store(self, text: str, user_id: str) -> None:
        """
        Hand a turn to mem0 so it can distil durable facts. mem0 does the extraction itself
        (its own LLM pass), so we pass the raw exchange rather than pre-summarising.
        """
        if not self.enabled or not text or not text.strip():
            return
        try:
            await asyncio.to_thread(self.memory.add, text, user_id=str(user_id))
            logger.info(f"[{user_id}] Semantic facts extracted and stored.")
        except Exception as e:
            logger.warning(f"[{user_id}] mem0 store failed (non-fatal): {e}")

    async def search(self, query: str, user_id: str, limit: Optional[int] = None) -> List[str]:
        """
        Recall the facts most relevant to this turn, as plain strings ready for prompt
        injection. Returns [] on any failure so the caller never has to branch.

        mem0's return shape has moved between versions (a bare list in older releases,
        {"results": [...]} in newer ones), so both are handled.
        """
        if not self.enabled or not query or not query.strip():
            return []

        top_k = limit or settings.MEM0_SEARCH_LIMIT or _DEFAULT_LIMIT

        def _search() -> Any:
            # mem0 2.x REJECTS a top-level user_id (raises ValueError pointing you at filters=)
            # and names the cap `top_k`; mem0 1.x wanted `user_id=` + `limit=`. Try the modern
            # shape first, then the legacy one. Getting this wrong is a silent failure: the
            # ValueError is swallowed below and recall just returns nothing, forever.
            try:
                return self.memory.search(query, filters={"user_id": str(user_id)}, top_k=top_k)
            except (TypeError, ValueError):
                return self.memory.search(query, user_id=str(user_id), limit=top_k)

        try:
            raw = await asyncio.to_thread(_search)
        except Exception as e:
            logger.warning(f"[{user_id}] mem0 search failed (non-fatal): {e}")
            return []

        items = raw.get("results", []) if isinstance(raw, dict) else (raw or [])
        facts: List[str] = []
        for item in items:
            if isinstance(item, dict):
                fact = item.get("memory") or item.get("text") or item.get("data")
            else:
                fact = item
            if fact and str(fact).strip():
                facts.append(str(fact).strip())
        return facts

    async def reset(self, user_id: str) -> None:
        """
        Forget everything about one person. Used by src/scripts/reset_thread.py for a clean demo
        re-run, and it is what a data-deletion request would call.
        """
        if not self.enabled:
            return
        try:
            await asyncio.to_thread(self.memory.delete_all, user_id=str(user_id))
            logger.info(f"[{user_id}] Semantic memory cleared.")
        except Exception as e:
            logger.warning(f"[{user_id}] mem0 reset failed (non-fatal): {e}")

    # Alias so callers written against the raw mem0 API keep working.
    async def delete_all(self, user_id: str) -> None:
        await self.reset(user_id)


# ── Process-wide singleton ────────────────────────────────────────────────────
# Constructing mem0 builds clients, so the worker does it once (lazily, on first use) rather
# than per turn. Returns None when disabled/unavailable so callers can `if mem: ...`.
_instance: Optional[SemanticMemory] = None
_instance_built = False


def get_semantic_memory() -> Optional[SemanticMemory]:
    global _instance, _instance_built
    if not _instance_built:
        _instance_built = True
        candidate = SemanticMemory()
        _instance = candidate if candidate.enabled else None
    return _instance


def format_memory_block(facts: List[str]) -> str:
    """
    Render recalled facts as the conditional {memory_block} prompt fragment (empty string when
    there is nothing to say, exactly like {rag_context_block}).

    Framed as "you remember" rather than data, and explicitly told not to recite it back — a
    customer being read their own file is unsettling; a customer whose advisor simply *knows*
    them feels looked after.
    """
    if not facts:
        return ""
    lines = "\n".join(f"- {f}" for f in facts[:_DEFAULT_LIMIT])
    return (
        "\nWHAT YOU REMEMBER ABOUT THIS PERSON FROM EARLIER CONVERSATIONS:\n"
        f"{lines}\n"
        "Use this naturally to personalise your reply and to avoid re-asking what you already "
        "know. Do NOT recite this list back to them, and don't claim it as fact if they now say "
        "otherwise — what they tell you today wins.\n"
    )
