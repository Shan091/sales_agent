# file: src/graph/nodes/rag.py
"""
Phase 3: Upstream Query Condensation & Technical Context Retrieval Node.

Architecture:
1. Reads chat_history + current message from ConversationState.
2. Routes to FAST_MODEL to generate a QueryExpansion schema: 3 diverse queries + metadata filters.
3. Batch-embeds all 3 queries in a single concurrent network hop.
4. Executes async hybrid search (Asymmetric RRF) against pgvector.
5. If retrieval succeeds: injects parent chunks into state["context_chunks"].
6. If retrieval returns nothing: proceeds with empty context — the archetype node handles it
   gracefully using OTOHOM_OVERVIEW and tells the customer the team will confirm specifics.

Condensation failure fallback (never escalates to human):
  If the fast model fails to produce a QueryExpansion, the raw last user message is embedded
  and used for an unfiltered hybrid search. If that also returns nothing, we still proceed
  with empty context rather than triggering a human handoff for a transient infra issue.
"""
import logging
import re

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.core.llm_factory import LLMFactory, execute_vendor_agnostic_node
from src.core.schemas import QueryExpansion
from src.core.database import async_session_maker
from src.rag.search import hybrid_search
from src.rag.ingestion import get_embedding_client
from src.graph.state import ConversationState, last_user_text

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
#  Query Condensation Prompt
# ═══════════════════════════════════════════════

QUERY_CONDENSATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a search query optimizer for Otohom's smart home product database.
Given the conversation history, generate 3 diverse search queries to retrieve the most relevant 
technical documentation. Also extract hard metadata filters if the user's intent clearly maps 
to a specific product category or document type.

RULES:
- semantic_query: A broad, natural language rephrasing of the user's technical intent.
- keyword_query: Extract exact product names, model numbers, SKUs as the customer stated them (e.g., "4SW touch panel", "digital door lock", "curtain motor").
- symptom_query: Rephrase the user's problem/symptom as a technical search (e.g., "switch overheating under high wattage load").
- product_name: The exact product name/SKU ONLY, no extra words (e.g., "4SW touch panel", "digital door lock"), used for a deterministic fallback price lookup. null if no specific product is named.
- category_filter: One of [switches, security, curtains, sensors, hubs, company] or null.
  Use ONLY these exact values — any other string will match zero rows. Map "lighting" → switches,
  "locks"/"door locks" → security, "smart controls" → sensors, "automation hub" → hubs.
  When uncertain, leave null so ALL categories are searched.
- doc_type_filter: "TECHNICAL_SPEC" or null. Leave null unless the user explicitly asks for an
  installation guide or troubleshooting help. Default to null.

You must generate your analysis structurally inside valid English JSON formatting keys, completely isolated 
from the conversational language text blocks. Under no circumstances may language text affect JSON key names.
"""),
    MessagesPlaceholder(variable_name="chat_history")
])


# ═══════════════════════════════════════════════
#  The Upstream RAG Node
# ═══════════════════════════════════════════════

def _mentions(haystack: str, product: str) -> bool:
    """
    Whether retrieved text actually talks about the product the customer named.

    Compared on alphanumerics only, so catalogue shorthand and customer phrasing line up: "6SW",
    "6 SW" and "6-sw" all collapse to the same key. A multi-word name counts as mentioned if its
    longest distinctive token appears, which keeps "Smart Door Lock Premium" matching a chunk that
    says "Smart Door Lock (Premium)".
    """
    def key(text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", (text or "").casefold())

    hay = key(haystack)
    if not hay:
        return False
    whole = key(product)
    if whole and whole in hay:
        return True
    # Fall back to the longest word in the name — the one carrying the identity.
    tokens = [key(t) for t in re.split(r"\s+", product or "") if len(t) > 2]
    if not tokens:
        return False
    return max(tokens, key=len) in hay


async def node_retrieve_technical_context(state: ConversationState) -> dict:

    """
    Shared upstream node that populates state["context_chunks"].
    Called BEFORE archetype-specific nodes when data_routing_flag == "TECHNICAL_RAG".
    """
    # ─── Step 1: Query Condensation (Fast Model) ───
    # Built through the factory so this call inherits the shared request timeout, retry policy
    # and thinking-disabled setting rather than silently diverging from every other LLM call.
    fast_llm = LLMFactory.get_llm(temperature=0.0, fast=True)

    formatted = QUERY_CONDENSATION_PROMPT.format_messages(
        chat_history=state["messages"]
    )

    expansion = await execute_vendor_agnostic_node(
        fast_llm, formatted, QueryExpansion, "rag_query_condensation"
    )

    if expansion:
        logger.info(
            f"Query Expansion: semantic='{expansion.semantic_query}', "
            f"keyword='{expansion.keyword_query}', symptom='{expansion.symptom_query}', "
            f"product='{expansion.product_name}', "
            f"category={expansion.category_filter}, doc_type={expansion.doc_type_filter}"
        )
        query_texts = [expansion.semantic_query, expansion.keyword_query, expansion.symptom_query]
        category_filter = expansion.category_filter
        doc_type_filter = expansion.doc_type_filter
        rag_query = expansion.keyword_query
    else:
        # Condensation failed (timeout / parse error). Embed the raw last user message
        # directly as a single unfiltered query — no extra LLM call, no escalation.
        raw = last_user_text(state, lower=False)
        if not raw:
            logger.warning("Query condensation failed and no user text found. Proceeding with empty context.")
            return {"context_chunks": [], "rag_query": None}
        logger.warning(f"Query condensation failed. Falling back to raw-message embed: '{raw[:80]}'")
        query_texts = [raw]
        category_filter = None
        doc_type_filter = None
        rag_query = raw

    # ─── Step 2: Batch Embed (Single Network Hop) ───
    embed_client = get_embedding_client()
    query_embeddings = await embed_client.aembed_documents(query_texts)

    # ─── Step 3: Hybrid Search ───
    async with async_session_maker() as session:
        results = await hybrid_search(
            session=session,
            query_embeddings=query_embeddings,
            query_texts=query_texts,
            category_filter=category_filter,
            doc_type_filter=doc_type_filter,
        )

    named_product = getattr(expansion, "product_name", None) if expansion else None

    if results:
        context_texts = [r.content for r in results]
        # Retrieval returning rows is NOT the same as retrieval finding the thing. Asymmetric RRF
        # always hands back its top-k, so a question about a product that does not exist still comes
        # back with three chunks about neighbouring products — measured: a fictional "SmartVault X9"
        # retrieved three switch/lock chunks. The model then answers from those plus its own priors.
        # So when the customer named a product and nothing retrieved even mentions it, treat the
        # specs as unavailable rather than letting the reply be improvised.
        if named_product and not _mentions(" ".join(context_texts), named_product):
            logger.warning(
                f"RAG returned {len(context_texts)} chunk(s) but none mention '{named_product}'. "
                "Flagging specs_unavailable so the reply stays honest."
            )
            return {"context_chunks": context_texts, "rag_query": rag_query, "specs_unavailable": True}

        logger.info(f"RAG retrieval succeeded. Injecting {len(context_texts)} parent chunks.")
        return {"context_chunks": context_texts, "rag_query": rag_query, "specs_unavailable": False}

    # Zero results is normal for a broad browsing question — the archetype answers warmly from the
    # overview. But when the customer NAMED a product and we found nothing about it, an LLM turn here
    # answers from pretraining: observed live, the model credited a lock with a video doorbell it
    # does not have. A prohibition in the prompt did not hold at 24k characters, so this is flagged
    # for code to handle instead of asked of the model.
    if named_product:
        logger.warning(
            f"RAG found nothing for named product '{named_product}'. "
            "Flagging specs_unavailable so the reply stays honest."
        )
    else:
        logger.info("RAG retrieval returned zero results. Proceeding with empty context.")
    return {"context_chunks": [], "rag_query": rag_query, "specs_unavailable": bool(named_product)}
