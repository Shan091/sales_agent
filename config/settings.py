# file: config/settings.py
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM Settings
    DEFAULT_MODEL: str = "google/gemini-2.5-flash"
    FAST_MODEL: str = "google/gemini-2.5-flash-lite"  # triage, query condensation, memory extraction
    # Ceiling for a whole node execution, which may span up to three output strategies.
    LLM_TIMEOUT_SECONDS: float = 40.0
    # Ceiling for ONE HTTP call. Must stay well under LLM_TIMEOUT_SECONDS: without it a single
    # stalled request consumes the entire node budget, so the structured-output fallback chain in
    # execute_vendor_agnostic_node never gets to run — the reply degrades to a canned escalation
    # instead of a retry that would have worked.
    LLM_REQUEST_TIMEOUT_SECONDS: float = 12.0
    # Cap on generated tokens per call. Every node here returns a small JSON object holding a
    # 1-3 sentence WhatsApp reply, so a few thousand is generous. Left unset, clients advertise
    # the model's full output window (65k+ on Gemini), which inflates the provider's cost
    # pre-authorisation and gets requests refused outright on a low balance.
    LLM_MAX_OUTPUT_TOKENS: int = 2048
    MAX_RETRIES: int = 2
    # Comma-separated model ids tried in order if the primary model fails outright (provider
    # outage, rate limit, decommissioned id). Kept as plain ids rather than a provider-specific
    # routing field so it works against any OpenAI-compatible base_url.
    MODEL_FALLBACKS: str = "google/gemini-2.5-flash-lite,openai/gpt-4o-mini"
    OPENAI_API_BASE: str = "https://openrouter.ai/api/v1"
    OPENAI_API_KEY: str = ""

    # Meta / WhatsApp API
    META_APP_SECRET: str = "your_app_secret"
    WHATSAPP_API_TOKEN: str = "your_whatsapp_token"
    WHATSAPP_PHONE_NUMBER_ID: str = "your_phone_id"
    WHATSAPP_VERIFY_TOKEN: str = "your_verify_token"  # M2 FIX: Added for webhook GET verification

    # Infrastructure
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/otohom"
    REDIS_URL: str = "redis://localhost:6379/0"

    # TaskIQ
    TASKIQ_QUEUE_NAME: str = "salesforge_wa_queue"

    # ═══════════════════════════════════════════════
    #  Lead delivery (client wants: WhatsApp Sales Leads Group + sales@otohom.com + CRM later)
    # ═══════════════════════════════════════════════
    # Two independent sinks; each fires only if configured, so partial setups still work.
    #
    # 1) Webhook — the primary/flexible sink. POSTs the lead as JSON to an automation endpoint
    #    (Google Sheets Apps Script, Zapier / Make / n8n). That automation fans the lead out to
    #    the Sheet, the WhatsApp Sales Leads Group and the CRM. NOTE: the WhatsApp *group* MUST
    #    go through this automation — the Meta Cloud API cannot post to groups directly.
    #    See docs/setup.md for the Apps Script and the deployment steps.
    LEADS_WEBHOOK_URL: str = ""

    # 2) Email — optional direct SMTP delivery to the sales inbox, independent of the webhook.
    #    Leave SMTP_HOST empty to disable email and rely on the webhook/automation instead.
    LEADS_EMAIL_TO: str = "sales@otohom.com"
    LEADS_EMAIL_FROM: str = ""
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_USE_TLS: bool = True

    # Colleagues allowed to control a handoff from WhatsApp — comma-separated, in any form the phone
    # offers: "9812345678", "919887654321" and "+91 98876 54321" all work, because matching is on the
    # last 10 digits (see `staff_numbers`). A colleague works the customer on their OWN number, so
    # nothing they say is visible here; texting `#done <customer> <what you did>` to the agent's
    # number is how the hold is released and the outcome reaches the agent's context.
    # This list IS the authorisation: a number that is not on it is treated as an ordinary customer
    # and gets an ordinary sales reply, because anyone who learned the syntax could otherwise
    # release any hold on any thread. Empty (the default) disables the commands entirely.
    STAFF_WHATSAPP_NUMBERS: str = ""

    @property
    def staff_numbers(self) -> set[str]:
        """
        The allowlist as a set of **last-10-digit** keys; empty when unconfigured.

        Meta delivers `from` in full international form (`919812345678`) while a colleague writes
        their number the way they say it out loud (`9812345678`). Comparing the raw strings meant a
        correctly configured allowlist matched nothing at all — silently, since a non-match is
        indistinguishable by design from an ordinary customer message. The last ten digits are the
        subscriber number in every form either side uses.
        """
        out = set()
        for raw in (self.STAFF_WHATSAPP_NUMBERS or "").split(","):
            digits = "".join(ch for ch in raw if ch.isdigit())
            if digits:
                out.add(digits[-10:])
        return out

    # Deployment — set APP_ENV=production to hard-fail on placeholder secrets at startup.
    APP_ENV: str = "development"

    # ═══════════════════════════════════════════════
    #  Agentic commerce — Razorpay checkout + the money bounds (Track 01)
    # ═══════════════════════════════════════════════
    # Test-mode keys from the Razorpay dashboard (Settings -> API Keys). Empty = the
    # checkout path is config-gated OFF and the agent simply never mints a link.
    RAZORPAY_KEY_ID: str = ""
    RAZORPAY_KEY_SECRET: str = ""
    # Secret you set when creating the webhook in the Razorpay dashboard. Used to verify
    # X-Razorpay-Signature. Empty = inbound Razorpay webhooks are rejected (fail-closed).
    RAZORPAY_WEBHOOK_SECRET: str = ""
    RAZORPAY_API_BASE: str = "https://api.razorpay.com/v1"

    # AUTONOMY: when True the agent owns the whole sale — pricing, quoting, bounded
    # discounts, closing and checkout — and pricing/quote/closing intents no longer
    # escalate. Human escalation is NOT removed: the critical safety valve (safety/legal,
    # post-payment disputes, repeated payment failures, persistent human requests,
    # unresolved anger) stays wired in BOTH modes. Set False to restore the original
    # Otohom client behaviour where pricing hands off to a human.
    AGENT_FULL_AUTONOMY: bool = True

    # ── Hard money bounds. Code clamps every amount to these; the LLM can never widen them.
    # Max discount the agent may ever apply, even if it "agrees" to more.
    MAX_DISCOUNT_PCT: float = 12.0
    # Absolute per-order limits (rupees) for a link mint. A computed total outside this
    # range is refused by validate_payment_request and no link is created.
    RAZORPAY_MIN_AMOUNT: float = 1.0
    RAZORPAY_MAX_AMOUNT: float = 500000.0
    # Per-line quantity sanity cap (blocks a "qty: 9999" proposal).
    MAX_LINE_QTY: int = 20
    # Consecutive failed payment attempts tolerated in-agent before the critical
    # human-escalation safety valve trips.
    # Declines before the critical human valve trips. 3, not 2: the first is a silent retry
    # (a card quirk needs no person), the second adds the human hatch, the third escalates.
    MAX_PAYMENT_FAILURES: int = 3

    # Hours a human may hold a thread before the agent resumes on its own. Handback is
    # human-driven by design (src/scripts/resolve_handoff.py), and this is only a safety net so a
    # forgotten release can't strand a customer with an agent that refuses to answer. Set 0 to
    # disable it and require an explicit release, always.
    HANDOFF_MAX_HOLD_HOURS: float = 24.0

    # Publicly fetchable URL of the Otohom brochure / lookbook PDF. Meta's servers download it, so
    # a localhost or tunnel-only address will be rejected. EMPTY = the agent is told not to offer a
    # brochure at all — better than promising one and sending nothing, which is what it used to do.
    BROCHURE_URL: str = ""
    BROCHURE_FILENAME: str = "Otohom-Lookbook.pdf"
    # The PDF this app serves at /api/v1/brochure. A file-sharing "view" link returns HTML, not a
    # PDF, so Meta rejects it — serving the real file ourselves is the reliable route. The merchant's
    # artwork is not in this repository, so supply your own file here (or set BROCHURE_URL instead).
    BROCHURE_FILE_PATH: str = "docs/brochure.pdf"
    # The externally reachable base URL of this app (the same tunnel/domain Meta and Razorpay post
    # their webhooks to), e.g. https://abc123.ngrok-free.app — no trailing slash.
    PUBLIC_BASE_URL: str = ""

    # ═══════════════════════════════════════════════
    #  Observability (Langfuse) — config-gated, no-op when keys are unset
    # ═══════════════════════════════════════════════
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # Semantic memory (mem0). Disabled by default; enable once pgvector + keys are up.
    MEM0_ENABLED: bool = False
    MEM0_SEARCH_LIMIT: int = 5
    # mem0 gets its OWN embedder, deliberately not the RAG model. Its job is matching short personal
    # facts ("lives in Kochi", "has a 3BHK"), which a small model does well — while bge-m3 is ~2.2GB
    # and was being loaded a SECOND time into the same worker process purely for this. English-only
    # is a conscious scope call for now; swap in a multilingual model here (and re-create the
    # collection) when Malayalam/Hindi recall is needed.
    MEM0_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Must match the model's output width, and must match the pgvector collection that already
    # exists — changing it means dropping otohom_semantic_memory so mem0 recreates it.
    MEM0_EMBEDDING_DIMS: int = 384

    # ═══════════════════════════════════════════════
    #  RAG Pipeline Configuration (Phase 3)
    # ═══════════════════════════════════════════════

    # Embedding Model
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIMENSIONS: int = 1024

    # Retrieval Thresholds (tuned empirically via tests/test_rag_pipeline.py)
    # 0.50 calibrated against BAAI/bge-m3 + the hand-curated docs/catalog markdown: correct
    # in-domain answers score 0.56-0.69, off-domain junk 0.36-0.45. The former 0.70 default
    # sat above every real answer, so retrieval fail-closed on 100% of queries.
    RAG_SIMILARITY_THRESHOLD: float = 0.50
    RAG_TOP_K_CHILDREN: int = 20  # Number of child chunks retrieved before parent dedup
    RAG_TOP_K_FINAL: int = 3  # Number of parent chunks injected into LLM context

    # Chunking Parameters
    RAG_CHILD_CHUNK_SIZE: int = 256  # tokens for child chunks (128-256 range)
    RAG_PARENT_CHUNK_SIZE: int = 1024  # tokens for parent chunks
    RAG_CHUNK_OVERLAP: int = 100  # token overlap for RecursiveCharacterTextSplitter

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def brochure_url(self) -> str:
        """
        The URL the agent may hand to Meta for the lookbook, or "" if none is available.

        An explicit BROCHURE_URL wins (use it to point at a CDN). Otherwise it is derived from
        PUBLIC_BASE_URL, so setting the tunnel you already configure for webhooks is enough — there
        is no second value to keep in sync, and no chance of the agent offering a brochure that
        resolves to a dead host.

        The derived form additionally requires the file to be on disk. Deriving a URL from the
        tunnel is only honest if this app can actually answer it, and the merchant's artwork is not
        in the repository — so a fresh checkout has nothing to serve, and the agent must not offer
        what would 404. An explicit BROCHURE_URL is exempt: something else is serving that one.
        """
        if self.BROCHURE_URL.strip():
            return self.BROCHURE_URL.strip()
        base = self.PUBLIC_BASE_URL.strip().rstrip("/")
        if not base or not Path(self.BROCHURE_FILE_PATH).is_file():
            return ""
        return f"{base}/api/v1/brochure"

    @property
    def razorpay_enabled(self) -> bool:
        """True only when a real key pair is present. Every money action is gated on this."""
        return bool(self.RAZORPAY_KEY_ID and self.RAZORPAY_KEY_SECRET)

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.LANGFUSE_PUBLIC_KEY and self.LANGFUSE_SECRET_KEY)

    @property
    def psycopg_dsn(self) -> str:
        """
        DATABASE_URL rewritten for psycopg3, which is what langgraph-checkpoint-postgres
        speaks. The app's own engine uses the asyncpg driver, so the '+asyncpg' (or any
        other '+driver') suffix has to be stripped for the checkpointer's connection pool.
        """
        url = self.DATABASE_URL
        if "+" in url.split("://", 1)[0]:
            scheme, rest = url.split("://", 1)
            url = f"{scheme.split('+', 1)[0]}://{rest}"
        return url

    def assert_production_secrets(self) -> None:
        """
        Fail fast when APP_ENV=production but security-critical secrets are still the
        placeholder defaults from .env.example. Called from the API lifespan and the
        TaskIQ worker startup so a misconfigured prod deploy never boots with a
        forgeable HMAC secret or an empty LLM key.
        """
        if self.APP_ENV.lower() != "production":
            return
        required = (
            "META_APP_SECRET", "WHATSAPP_API_TOKEN", "WHATSAPP_PHONE_NUMBER_ID",
            "WHATSAPP_VERIFY_TOKEN", "OPENAI_API_KEY",
        )
        bad = [
            name for name in required
            if not getattr(self, name, "") or str(getattr(self, name)).startswith("your_")
        ]
        # Razorpay is optional overall (the agent degrades to no-checkout), but a
        # HALF-configured money path in production is a bug, not a degradation: keys
        # without a webhook secret means paid orders would never be confirmed, and a
        # placeholder secret means anyone could forge a "paid" event.
        if self.RAZORPAY_KEY_ID or self.RAZORPAY_KEY_SECRET or self.RAZORPAY_WEBHOOK_SECRET:
            bad += [
                name for name in ("RAZORPAY_KEY_ID", "RAZORPAY_KEY_SECRET", "RAZORPAY_WEBHOOK_SECRET")
                if not getattr(self, name, "") or str(getattr(self, name)).startswith("your_")
            ]
        if bad:
            raise RuntimeError(
                f"APP_ENV=production but placeholder/empty secrets detected: {', '.join(bad)}. "
                "Set real values before starting."
            )


settings = Settings()
