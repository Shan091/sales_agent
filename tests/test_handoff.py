"""
Human handoff: who owns the thread, what the customer is told, and how it comes back.

Handoff is human-driven — a colleague works the conversation on a DIFFERENT WhatsApp number and
releases the thread with src/scripts/resolve_handoff.py when they're done. Three properties matter
and are asserted here:

  1. The thread stays with the human until released. Resuming on an ordinary-looking message would
     put two voices on two numbers answering the same customer.
  2. The customer is told once, in words that match WHY, and gets a short holding line after that —
     not the same announcement repeated, which is what makes an agent look broken.
  3. It cannot dead-end. A forgotten release is caught by HANDOFF_MAX_HOLD_HOURS.
"""
import asyncio
import time

import pytest

from config.settings import settings
from src.graph.nodes import triage
from src.graph.workflow import route_after_triage


def _state(**kw):
    base = {
        "messages": [("user", "hello")],
        "current_archetype": "SALES_HIGH_INTENT",
        "language_preference": "English",
        "requires_human_handoff": False,
        "data_routing_flag": "NONE",
    }
    base.update(kw)
    return base


class TestTheHumanKeepsTheThreadUntilTheyReleaseIt:
    def test_an_ordinary_message_during_a_hold_does_not_resume_the_agent(self):
        s = _state(handoff_active=True, requires_human_handoff=False,
                   current_archetype="SALES_HIGH_INTENT")
        assert route_after_triage(s) == "human_escalation"

    def test_a_technical_question_during_a_hold_does_not_reach_retrieval(self):
        s = _state(handoff_active=True, data_routing_flag="TECHNICAL_RAG")
        assert route_after_triage(s) == "human_escalation"

    def test_a_hostile_turn_during_a_hold_is_still_deflected_locally(self):
        # Precedence is unchanged: trolling a held thread must not become a human's problem.
        s = _state(handoff_active=True, current_archetype="MALICIOUS_ADVERSARIAL")
        assert route_after_triage(s) == "adversarial_block"

    def test_a_released_thread_routes_normally_again(self):
        s = _state(handoff_active=False, requires_human_handoff=False)
        assert route_after_triage(s) == "high_intent"


class TestPromisesTheSystemCanActuallyKeep:
    """
    The agent told a customer "here's our digital lookbook" and sent nothing. An offer the system
    can't honour is worse than no offer — it spends trust on the one thing they explicitly asked for.
    """

    def test_with_no_brochure_configured_the_agent_is_told_not_to_offer_one(self, monkeypatch):
        from src.graph.nodes.sales import _build_brochure_block
        # BOTH knobs, because settings.brochure_url derives from PUBLIC_BASE_URL when
        # BROCHURE_URL is unset — clearing only one leaves a real .env value in play and the
        # assertion would pass or fail depending on whose machine ran it.
        monkeypatch.setattr(settings, "BROCHURE_URL", "")
        monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "")
        block = _build_brochure_block({})
        assert "NO brochure" in block
        assert "never claim you have" in block

    def test_with_a_brochure_configured_the_flag_is_explained(self, monkeypatch):
        from src.graph.nodes.sales import _build_brochure_block
        monkeypatch.setattr(settings, "BROCHURE_URL", "https://example.com/lookbook.pdf")
        block = _build_brochure_block({})
        assert "send_brochure" in block
        # The transcript's other fault: asking WhatsApp-or-email on a WhatsApp thread.
        assert "WhatsApp or email" in block

    def test_the_url_is_derived_from_the_public_base_when_not_set_explicitly(self, monkeypatch, tmp_path):
        """One knob. The tunnel already configured for webhooks is enough, so the agent can never
        offer a brochure that resolves to a dead host."""
        pdf = tmp_path / "look.pdf"
        pdf.write_bytes(b"%PDF-1.4 stub")
        monkeypatch.setattr(settings, "BROCHURE_URL", "")
        monkeypatch.setattr(settings, "BROCHURE_FILE_PATH", str(pdf))
        monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "https://abc123.ngrok-free.app/")
        assert settings.brochure_url == "https://abc123.ngrok-free.app/api/v1/brochure"

    def test_a_tunnel_with_no_file_to_serve_means_no_brochure(self, monkeypatch, tmp_path):
        """The derived URL points at THIS app, so it is only honest if the file exists. The
        merchant's artwork is not in the repository, so a fresh checkout has nothing to serve —
        and promising a PDF that 404s is the exact failure this class is named after."""
        monkeypatch.setattr(settings, "BROCHURE_URL", "")
        monkeypatch.setattr(settings, "BROCHURE_FILE_PATH", str(tmp_path / "absent.pdf"))
        monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "https://abc123.ngrok-free.app")
        assert settings.brochure_url == ""

    def test_an_explicit_url_is_exempt_from_the_file_check(self, monkeypatch, tmp_path):
        """Something else is serving that one, so this app having no copy is irrelevant."""
        monkeypatch.setattr(settings, "BROCHURE_URL", "https://cdn.example.com/look.pdf")
        monkeypatch.setattr(settings, "BROCHURE_FILE_PATH", str(tmp_path / "absent.pdf"))
        monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "")
        assert settings.brochure_url == "https://cdn.example.com/look.pdf"

    def test_an_explicit_url_wins_over_the_derived_one(self, monkeypatch):
        monkeypatch.setattr(settings, "BROCHURE_URL", "https://cdn.example.com/look.pdf")
        monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "https://abc123.ngrok-free.app")
        assert settings.brochure_url == "https://cdn.example.com/look.pdf"

    def test_neither_configured_means_no_brochure(self, monkeypatch):
        monkeypatch.setattr(settings, "BROCHURE_URL", "")
        monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "")
        assert settings.brochure_url == ""


class TestGroundingWhenRetrievalCameBackEmpty:
    """
    Observed live: with no retrieved context the model credited a lock with a video doorbell it does
    not have. The prose "stay grounded" rules were not enough, so the stop is conditional and only
    appears on the turns where there is genuinely nothing to ground against.
    """

    def test_no_block_when_context_exists(self):
        from src.graph.nodes.sales import _build_grounding_block
        assert _build_grounding_block({"context_chunks": ["Premium lock: five unlock methods"]}) == ""

    def test_an_empty_context_forbids_stating_any_specific(self):
        from src.graph.nodes.sales import _build_grounding_block
        block = _build_grounding_block({"context_chunks": []})
        assert "NO PRODUCT DATA WAS RETRIEVED" in block
        for forbidden in ("features", "unlock methods", "materials", "colours", "comparisons"):
            assert forbidden in block

    def test_it_names_the_safe_alternative_rather_than_only_forbidding(self):
        from src.graph.nodes.sales import _build_grounding_block
        block = _build_grounding_block({})
        assert "exact details" in block
        assert "categories and outcomes" in block


class TestANamedProductWithNoDataIsRefusedInCode:
    """
    The prompt-level stop was not enough: measured live, with no retrieved context the model still
    stated unlock methods, materials and colours — twice out of two runs. The block is present but
    sits 93% through a 24k-character system prompt, competing with the archetype's mandate to be
    helpful. So the guarantee moved to code, the same way the money guarantee did.
    """

    class _Expansion:
        semantic_query = "premium smart door lock features"
        keyword_query = "Smart Door Lock Premium"
        symptom_query = "door lock unlock methods"
        product_name = "Smart Door Lock Premium"
        category_filter = "security"
        doc_type_filter = None

    def _patch(self, monkeypatch, expansion, results):
        import src.graph.nodes.rag as rag

        class _Embed:
            async def aembed_documents(self, texts):
                return [[0.0] * 8 for _ in texts]

        class _Session:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *exc):
                return False

        async def _condense(*a, **kw):
            return expansion

        async def _search(**kw):
            return results

        monkeypatch.setattr(rag.LLMFactory, "get_llm", staticmethod(lambda **kw: object()))
        monkeypatch.setattr(rag, "get_embedding_client", lambda: _Embed())
        monkeypatch.setattr(rag, "async_session_maker", lambda: _Session())
        monkeypatch.setattr(rag, "execute_vendor_agnostic_node", _condense)
        monkeypatch.setattr(rag, "hybrid_search", _search)
        return rag

    async def test_a_named_product_with_zero_hits_raises_the_flag(self, monkeypatch):
        rag = self._patch(monkeypatch, self._Expansion(), [])
        out = await rag.node_retrieve_technical_context({"messages": [("user", "features of the premium lock?")]})
        assert out["specs_unavailable"] is True
        assert out["context_chunks"] == []
        # Never a human handoff from a retrieval miss.
        assert not out.get("requires_human_handoff")

    async def test_a_broad_question_with_zero_hits_does_not_raise_it(self, monkeypatch):
        expansion = self._Expansion()
        expansion.product_name = None
        rag = self._patch(monkeypatch, expansion, [])
        out = await rag.node_retrieve_technical_context({"messages": [("user", "what do you sell?")]})
        assert out["specs_unavailable"] is False

    async def test_a_successful_retrieval_lowers_the_flag(self, monkeypatch):
        class _Row:
            content = "Smart Door Lock Premium: fingerprint, PIN, card, key, app."
        rag = self._patch(monkeypatch, self._Expansion(), [_Row()])
        out = await rag.node_retrieve_technical_context({"messages": [("user", "premium lock?")]})
        assert out["specs_unavailable"] is False
        assert out["context_chunks"] == ["Smart Door Lock Premium: fingerprint, PIN, card, key, app."]

    async def test_chunks_that_never_mention_the_product_still_count_as_no_data(self, monkeypatch):
        """
        The case zero-results checking misses. Asymmetric RRF always returns its top-k, so a query
        about a product that does not exist comes back with chunks about neighbouring products —
        measured live with a fictional "SmartVault X9", which retrieved three switch/lock chunks.
        """
        class _Row:
            content = "Curtain Motor: 1.2Nm torque, app and voice control."
        rag = self._patch(monkeypatch, self._Expansion(), [_Row()])
        out = await rag.node_retrieve_technical_context({"messages": [("user", "premium lock specs?")]})
        assert out["specs_unavailable"] is True
        # The chunks are still passed through — they may be useful for the honest next step.
        assert out["context_chunks"]

    @pytest.mark.parametrize("chunk,product", [
        ("The 6 SW glass panel drives six circuits.", "6SW"),
        ("The 6-SW panel drives six circuits.", "6 SW"),
        ("Smart Door Lock (Premium) has five unlock methods.", "Smart Door Lock Premium"),
    ])
    def test_catalogue_shorthand_and_customer_phrasing_are_treated_as_the_same_product(self, chunk, product):
        from src.graph.nodes.rag import _mentions
        assert _mentions(chunk, product)

    def test_an_unrelated_chunk_is_not_a_mention(self):
        from src.graph.nodes.rag import _mentions
        assert not _mentions("Grande switches and curtain motors.", "SmartVault X9")
        assert not _mentions("", "anything")

    async def test_the_sales_node_answers_without_calling_the_model(self, monkeypatch):
        from src.graph.nodes import sales
        from src.logic.prompts import HIGH_INTENT_PROMPT

        async def _never(*a, **kw):
            raise AssertionError("no LLM call may happen when specs are unavailable")

        monkeypatch.setattr(sales, "execute_vendor_agnostic_node", _never)
        out = await sales._execute_sales_node({"specs_unavailable": True}, HIGH_INTENT_PROMPT)

        text = out["messages"][0].content
        assert "guess" in text
        # The flag is consumed, or every later turn would repeat this reply.
        assert out["specs_unavailable"] is False

    async def test_the_refusal_still_offers_a_way_forward(self, monkeypatch):
        from src.graph.nodes import sales
        from src.logic.prompts import HIGH_INTENT_PROMPT
        monkeypatch.setattr(sales, "execute_vendor_agnostic_node", lambda *a, **kw: None)
        out = await sales._execute_sales_node({"specs_unavailable": True}, HIGH_INTENT_PROMPT)
        options = out["messages"][0].response_metadata["options"]
        assert options, "a dead-end 'I don't know' is worse than the guess it replaced"
        assert any(o["postback_id"] == "CONNECT_NOW" for o in options)
        # Descriptions matter: 3 options render as buttons, which drop them — so this must list.
        assert all(o.get("description") for o in options)

    async def test_it_never_preempts_a_confirmed_checkout(self, monkeypatch):
        """Ordering guarantee: a grounding refusal must not swallow the pay turn."""
        from src.graph.nodes import sales
        from src.logic.prompts import HIGH_INTENT_PROMPT
        monkeypatch.setattr(settings, "AGENT_FULL_AUTONOMY", True)
        out = await sales._execute_sales_node(
            {
                "specs_unavailable": True,
                "checkout_confirmed": True,
                "pending_order": {"amount": 9500.0},
                "payment_link_sent": False,
                # A name on file, so the tap goes straight through — the hold for a missing name is
                # its own guarantee and is tested in TestTheNameIsAskedBeforeTheMoneyNotAfter.
                "customer_name": "Anil",
            },
            HIGH_INTENT_PROMPT,
        )
        assert "payment link" in out["messages"][0].content


class TestTheNameIsAskedBeforeTheMoneyNotAfter:
    """
    The receipt promised "someone will call you" while the system held no name and no city, so the
    ask was added right after it. Wrong moment: a question that arrives once the money is done gets
    ignored, and the team is left with an order nobody's name is on.

    The client's rule is that the name is **mandatory at the pay button**. So the confirm tap is held
    for exactly one turn while the agent asks, and the link mints on the same authorisation as soon as
    the name arrives — parsed in code, because an LLM turn at the pay button costs the customer
    seconds at the worst possible moment and could re-propose the order instead of completing it.
    """

    ORDER = {"amount": 19500.0, "line_items": [{"sku": "Smart Door Lock Base", "qty": 1}]}

    @staticmethod
    def _state(**over):
        return {
            "messages": [("user", "Confirm & pay [CONFIRM_CHECKOUT]")],
            "checkout_confirmed": True,
            "pending_order": TestTheNameIsAskedBeforeTheMoneyNotAfter.ORDER,
            **over,
        }

    async def test_the_tap_is_held_and_the_name_asked_when_we_do_not_have_one(self):
        from src.graph.nodes.sales import _execute_sales_node
        out = await _execute_sales_node(self._state(), None)
        assert out["awaiting_pay_details"] is True
        body = str(out["messages"][0].content)
        assert "name" in body.lower() and "city" in body.lower()
        assert "link" not in body.lower(), "nothing may promise a link before we can mint one"
        # A blank line under the question — this arrives at the most decisive moment of the sale.
        assert "\n\n" in body
        # And it must never tell them the city is optional: they would skip it, and the team would
        # have an order with a name on it and nowhere to send an installer.
        for hedge in ("if you'd rather", "optional", "if you like", "no problem", "don't have to"):
            assert hedge not in body.lower(), hedge

    async def test_only_a_name_gets_one_ask_for_where_it_is_going(self):
        from src.graph.nodes.sales import _execute_sales_node
        out = await _execute_sales_node(
            self._state(messages=[("user", "Anil")], awaiting_pay_details=True), None
        )
        assert out["customer_name"] == "Anil"
        assert out["awaiting_pay_details"] is True, "still waiting — but for the city now"
        assert out["checkout_confirmed"] is True, "the tap still stands"
        body = str(out["messages"][0].content)
        assert "city" in body.lower()
        assert "pin" in body.lower(), "the easiest way to answer is worth naming"
        assert "\n\n" in body

    @pytest.mark.parametrize("reply,expected_city", [
        ("Kochi", "Kochi"),
        ("Kozhikode, Kerala", "Kozhikode, Kerala"),
        ("MG Road, Bengaluru", "MG Road, Bengaluru"),   # a dropped pin arrives as an address
        ("no", None),
        ("skip", None),
        ("Confirm & pay [CONFIRM_CHECKOUT]", None),
    ])
    async def test_whatever_comes_back_the_link_goes_out(self, reply, expected_city):
        # The city is worth asking for once and never worth blocking a payment over.
        from src.graph.nodes.sales import _execute_sales_node
        out = await _execute_sales_node(
            self._state(messages=[("user", reply)], awaiting_pay_details=True, customer_name="Anil"), None
        )
        assert out["awaiting_pay_details"] is False
        assert out["checkout_confirmed"] is True
        assert out["city"] == expected_city
        assert "payment link" in str(out["messages"][0].content)

    async def test_a_tap_with_a_name_on_file_goes_straight_through(self):
        from src.graph.nodes.sales import _execute_sales_node
        out = await _execute_sales_node(self._state(customer_name="Anil"), None)
        assert out["awaiting_pay_details"] is False
        assert "payment link" in str(out["messages"][0].content)

    async def test_the_reply_carrying_the_name_mints_on_the_same_authorisation(self):
        from src.graph.nodes.sales import _execute_sales_node
        out = await _execute_sales_node(
            self._state(messages=[("user", "Anil, Kochi")], awaiting_pay_details=True), None
        )
        assert out["customer_name"] == "Anil"
        assert out["city"] == "Kochi"
        assert out["awaiting_pay_details"] is False, "name and city in one reply needs no second ask"
        assert out["checkout_confirmed"] is True, "the tap already authorised this order"
        assert "Anil" in str(out["messages"][0].content)

    async def test_a_dropped_pin_answers_the_city_question(self):
        # WhatsApp sends a `location` message; processing.py turns it into the place name, the address
        # or the coordinates before the graph sees it, so by here it is ordinary text.
        from src.graph.nodes.sales import _parse_city
        assert _parse_city("Kaloor, Kochi") == "Kaloor, Kochi"
        assert _parse_city("9.9312, 76.2673") == "9.9312, 76.2673"
        assert _parse_city("  ") is None

    async def test_tapping_pay_again_re_asks_rather_than_storing_the_button_as_a_name(self):
        from src.graph.nodes.sales import _execute_sales_node
        out = await _execute_sales_node(self._state(awaiting_pay_details=True), None)
        assert "customer_name" not in out
        assert out["awaiting_pay_details"] is True
        assert "name" in str(out["messages"][0].content).lower()

    @pytest.mark.parametrize("reply", [
        "why do you need my name?",
        "Actually can you tell me if it works with Alexa first",
        "1234567",
        "Confirm & pay [CONFIRM_CHECKOUT]",
    ])
    def test_it_refuses_to_treat_anything_else_as_a_name(self, reply):
        # Storing "why do you need my name?" would put it on the order, the sheet and the receipt.
        from src.graph.nodes.sales import _parse_name_and_city
        assert _parse_name_and_city(reply) == (None, None)

    @pytest.mark.parametrize("reply,expected", [
        ("Anil", ("Anil", None)),
        ("Anil, Kochi", ("Anil", "Kochi")),
        ("I'm Anil from Kochi", ("Anil", "Kochi")),
        ("my name is Anil Menon", ("Anil Menon", None)),
        ("this is Anjali in Bengaluru", ("Anjali", "Bengaluru")),
    ])
    def test_it_reads_the_shapes_people_actually_type(self, reply, expected):
        from src.graph.nodes.sales import _parse_name_and_city
        assert _parse_name_and_city(reply) == expected

    def test_a_held_tap_cannot_mint(self):
        # The mint gate is the money guarantee; the hold has to be visible to it, not just to the
        # node that sent the question.
        import inspect
        from src.tasks import processing
        gate = inspect.getsource(processing.taskiq_process_message)
        assert 'not fstate.get("awaiting_pay_details")' in gate

    def test_nothing_asks_for_details_after_the_receipt_any_more(self):
        import inspect
        from src.tasks import processing
        assert not hasattr(processing, "_ask_for_delivery_details")
        sequence = inspect.getsource(processing._send_payment_confirmation)
        assert "delivery_details" not in sequence
        assert "_format_payment_receipt" in sequence


class TestPostPaymentMessages:
    """
    The transaction is the emotional peak of the whole conversation and the point at which the
    customer most needs evidence. Those are two different jobs, so they are two messages.
    """

    class _Row:
        id = 12
        line_items = [{"sku": "Smart Door Lock Premium", "qty": 1}]
        product_summary = "1 x Smart Door Lock Premium"
        razorpay_payment_id = "pay_TWfJ7K2xQ9aLmN"
        razorpay_link_id = "plink_TWeI5D4FpGRRgD"

    def test_the_first_message_marks_the_moment_and_stops(self):
        from src.tasks.processing import _payment_celebration
        text = _payment_celebration(self._Row())
        assert "confirmed" in text.lower()
        assert len(text) < 200          # short enough to read at a glance
        assert "🙏" not in text
        assert text.count("\n\n") == 1  # exactly one paragraph break

    def test_the_receipt_carries_the_evidence_a_customer_can_quote_back(self):
        from src.tasks.processing import _format_payment_receipt
        text = _format_payment_receipt(self._Row(), 28100.0)
        assert "28,100" in text
        assert "OTO-12" in text
        assert "pay_TWfJ7K2xQ9aLmN" in text
        assert "Smart Door Lock Premium" in text

    def test_the_receipt_says_a_call_comes_next_not_an_install_date(self):
        from src.tasks.processing import _format_payment_receipt
        text = _format_payment_receipt(self._Row(), 28100.0).lower()
        assert "call you" in text
        # Promising a date before the team confirms it is the commitment GUARDRAIL_RULES forbids.
        assert "will be installed" not in text
        assert "installation date" not in text

    def test_the_receipt_degrades_without_line_items_or_a_payment_id(self):
        from src.tasks.processing import _format_payment_receipt

        class _Bare:
            id = 13
            line_items = None
            product_summary = "(order confirmed via webhook)"
            razorpay_payment_id = None
            razorpay_link_id = "plink_ABC"

        text = _format_payment_receipt(_Bare(), 0.0)
        assert "OTO-13" in text
        assert "plink_ABC" in text          # falls back to the link id as the reference
        assert "Amount paid" not in text    # never invents a figure it doesn't have

    def test_neither_message_uses_the_banned_emoji(self):
        from src.tasks.processing import _payment_celebration, _format_payment_receipt
        for text in (_payment_celebration(self._Row()), _format_payment_receipt(self._Row(), 100.0)):
            assert "🙏" not in text

    def test_the_reference_leads_the_receipt_in_copyable_monospace(self):
        """It is the one thing anyone comes back to this message for — theirs, the team's or their
        bank's — so it goes first, on its own line, long-press-copyable."""
        from src.tasks.processing import _format_payment_receipt
        text = _format_payment_receipt(self._Row(), 28100.0)
        assert "```OTO-12```" in text
        assert "```pay_TWfJ7K2xQ9aLmN```" in text
        assert text.index("OTO-12") < text.index("28,100")

    def test_no_figure_in_the_receipt_is_wrapped_in_bold(self):
        # WhatsApp bold adjacent to digits or punctuation renders unreliably, and the customer
        # then just sees the asterisks around the number they came to check.
        import re
        from src.tasks.processing import _format_payment_receipt
        for line in _format_payment_receipt(self._Row(), 28100.0).splitlines():
            if line.startswith("*") and line.endswith("*"):
                assert not re.search(r"[\d₹]", line), line

    def test_the_receipt_states_how_they_paid(self):
        from src.tasks.processing import _format_payment_receipt
        assert "Payment mode:" in _format_payment_receipt(self._Row(), 28100.0)

    def test_electrician_shorthand_is_glossed_the_way_the_quote_glossed_it(self):
        from src.tasks.processing import _format_payment_receipt

        class _Row:
            id = 14
            line_items = [{"sku": "6 SW", "qty": 2}]
            product_summary = None
            razorpay_payment_id = "pay_X"
            razorpay_link_id = None

        text = _format_payment_receipt(_Row(), 9200.0)
        assert "6 SW" in text
        assert "switch" in text.lower()   # the plain words, as on the quote

    def test_the_greeting_is_used_only_when_the_name_was_actually_given(self):
        # The name is asked for at the pay button, so it is normally present by now — but a link
        # resent from an older order may predate that, and a guessed name on a receipt is worse
        # than none.
        from src.tasks.processing import _payment_celebration
        assert "Thank you —" in _payment_celebration(self._Row())
        assert "Thank you, Anjali —" in _payment_celebration(self._Row(), "Anjali")
        assert "Thank you —" in _payment_celebration(self._Row(), "   ")


class TestTheConfirmationSequenceHasOneOwner:
    """
    Razorpay sends more than one event for the same money — `payment_link.paid` AND
    `payment.captured`, with different event ids — so the webhook's event-id dedup cannot collapse
    them and both reach the paid branch. Each send was individually idempotent, which was the wrong
    grain: live, the two runs interleaved, the loser's receipt was skipped as a duplicate while the
    winner's was still in flight, and the loser ran straight on to the name-and-city ask, which
    OVERTOOK the receipt. The customer read the question and then a long receipt on top of it.

    So the claim covers the whole sequence, and the ask is last by construction rather than by luck.
    """

    class _Row:
        id = 7
        amount = 30300.0
        line_items = [{"sku": "Smart Door Lock Base", "qty": 1}]
        product_summary = "1 x Smart Door Lock Base"
        razorpay_payment_id = "pay_TXdKA3UXfCnS9X"
        razorpay_link_id = "plink_TXd9"

    @staticmethod
    def _harness(monkeypatch, claims):
        """Records every dispatched message; `claims` is what the idempotency claim returns in turn."""
        from src.tasks import processing

        sent = []

        class _Cache:
            async def check_and_set_idempotency(self, *_a, **_k):
                return claims.pop(0) if claims else False

        class _WA:
            async def dispatch_message(self, **kw):
                sent.append(kw["text"])

        class _Graph:
            async def aget_state(self, *_a, **_k):
                return None

        monkeypatch.setattr(processing, "_cache", _Cache())
        monkeypatch.setattr(processing, "_whatsapp", _WA())
        monkeypatch.setattr(processing, "_graph_app", _Graph())
        # The real gaps are 0.6s each; the ordering under test is not the sleeping. `real_sleep` is
        # captured first because processing.asyncio IS this module's asyncio — patching it without
        # capturing makes the replacement call itself.
        real_sleep = asyncio.sleep
        monkeypatch.setattr(processing.asyncio, "sleep", lambda *_a, **_k: real_sleep(0))
        return processing, sent

    async def test_the_receipt_is_the_last_message(self, monkeypatch):
        processing, sent = self._harness(monkeypatch, [True, True])
        owned = await processing._send_payment_confirmation(
            "919812345678", {"configurable": {"thread_id": "919812345678"}},
            self._Row(), 30300.0, "", "pay_TXdKA3UXfCnS9X",
        )
        assert owned is True
        assert len(sent) == 2
        assert "confirmed" in sent[0].lower()          # the moment
        assert "OTO-7" in sent[1]                       # the evidence they keep, and nothing after it
        # The name-and-city question used to sit here, after the money. It runs at the pay button now.
        assert "city" not in sent[1].lower()

    async def test_a_second_event_for_the_same_payment_sends_nothing(self, monkeypatch):
        processing, sent = self._harness(monkeypatch, [False])
        owned = await processing._send_payment_confirmation(
            "919812345678", {"configurable": {"thread_id": "919812345678"}},
            self._Row(), 30300.0, "", "pay_TXdKA3UXfCnS9X",
        )
        assert owned is False
        assert sent == [], "the second event must not re-send, re-order or duplicate anything"

    def test_losing_the_race_cannot_leave_a_paid_order_pending(self):
        # The claim guards the CHAT only. The state write that clears pending_order/checkout_confirmed
        # is a money guarantee and has to run for every paid event, so it must sit in the branch
        # ABOVE the send helper — never inside it.
        import inspect
        from src.tasks import processing
        branch = inspect.getsource(processing.taskiq_confirm_payment)
        helper = inspect.getsource(processing._send_payment_confirmation)
        assert '"pending_order": None' in branch
        assert '"pending_order": None' not in helper, "the chat helper must not own a money guarantee"
        assert branch.index('"pending_order": None') < branch.index("_send_payment_confirmation(")


class TestPaymentSettlesTheOrderInState:
    """
    A paid order must stop being a PENDING one. The mint block fires on
    `checkout_confirmed and pending_order and not payment_link_sent`, and any later turn that
    rebuilds a quote sets `payment_link_sent` back to False — so an order left in state after payment
    sat one "Confirm & pay" tap away from a second live link for something already bought. Observed
    live: answering the post-payment name/city question re-sent the whole quote with its pay button.

    Clearing it is a money guarantee, not housekeeping, so it is asserted here rather than trusted.
    """

    class _Row:
        id = 4
        line_items = [{"sku": "Indoor Smart Camera", "qty": 1}]
        product_summary = "1 x Indoor Smart Camera"
        razorpay_payment_id = "pay_TXDmLBsFE0yT5Y"
        razorpay_link_id = "plink_TXDm"

    def test_the_paid_branch_clears_every_key_the_mint_gate_reads(self):
        import inspect
        from src.tasks import processing
        source = inspect.getsource(processing.taskiq_confirm_payment)
        for key in ('"pending_order": None', '"checkout_confirmed": False', '"payment_link_url": None'):
            assert key in source, key

    def test_the_paid_branch_records_what_was_bought(self):
        import inspect
        from src.tasks import processing
        assert '"paid_line_items": _paid_line_items(row)' in inspect.getsource(processing.taskiq_confirm_payment)

    def test_the_bought_items_carry_sku_and_quantity(self):
        from src.tasks.processing import _paid_line_items
        assert _paid_line_items(self._Row()) == {"Indoor Smart Camera": 1}

    def test_quantities_are_kept_so_a_larger_repeat_order_still_differs(self):
        class _R:
            line_items = [{"sku": "6 SW", "qty": 2}, {"sku": "Zigbee Hub", "qty": 1}]
        from src.tasks.processing import _paid_line_items
        assert _paid_line_items(_R()) == {"6 SW": 2, "Zigbee Hub": 1}

    @pytest.mark.parametrize("line_items", [None, [], "not a list", [{}], [{"sku": "  "}], [None]])
    def test_a_partial_audit_row_degrades_to_empty_rather_than_raising(self, line_items):
        """The audit write is fail-soft, so this runs against rows that may be incomplete. Raising
        here would abandon the rest of the paid handler — including the receipt."""
        class _R:
            pass
        row = _R()
        row.line_items = line_items
        from src.tasks.processing import _paid_line_items
        assert _paid_line_items(row) == {}

    def test_a_missing_quantity_counts_as_one(self):
        class _R:
            line_items = [{"sku": "Curtain Motor"}, {"sku": "Zigbee Hub", "qty": "bad"}]
        from src.tasks.processing import _paid_line_items
        assert _paid_line_items(_R()) == {"Curtain Motor": 1, "Zigbee Hub": 1}

    def test_the_settled_order_is_what_the_sales_node_guard_reads(self):
        """The two halves are only a guarantee together: state clearing removes the mint path, and
        this stops the model walking the customer back into a checkout they finished."""
        from src.core.schemas import CheckoutItem
        from src.graph.nodes.sales import _reproposes_paid_order
        from src.tasks.processing import _paid_line_items
        paid = _paid_line_items(self._Row())
        assert _reproposes_paid_order([CheckoutItem(sku="Indoor Smart Camera", qty=1)], paid)
        assert not _reproposes_paid_order([CheckoutItem(sku="Curtain Motor", qty=1)], paid)


class TestWhatTheCustomerIsTold:
    @pytest.mark.parametrize("reason", ["payment", "post_payment", "upset", "safety", "requested"])
    async def test_each_reason_has_its_own_words_and_holds_the_thread(self, reason):
        out = await triage.node_human_escalation(_state(handoff_reason=reason))
        text = out["messages"][0].content
        assert text.strip()
        assert out["handoff_active"] is True
        assert out["handoff_notified"] is True

    async def test_a_technical_error_tells_the_team_without_freezing_the_conversation(self):
        """
        Observed live: the model call hit a credit limit, the agent escalated with reason `error`, and
        every message after it got the holding line — "Hi" answered with "I've handed this to a
        colleague" — until somebody released the thread by hand. A failure of OUR machinery is not a
        customer who needs a person, so the team is still told and the customer still gets the
        apology, but the hold is not sticky: triage recomputes `requires_human_handoff` on the next
        message, so the agent retries by itself and a transient outage costs one turn.
        """
        out = await triage.node_human_escalation(_state(handoff_reason="error"))
        assert out["messages"][0].content.strip()
        assert out["requires_human_handoff"] is True, "the team is still told"
        assert out["handoff_active"] is False, "but the thread is not frozen behind a holding line"
        assert out["handoff_notified"] is False
        assert out["handoff_started_at"] is None

    async def test_the_reasons_do_not_all_say_the_same_thing(self):
        texts = set()
        for reason in ("payment", "post_payment", "upset", "safety", "error", "requested"):
            out = await triage.node_human_escalation(_state(handoff_reason=reason))
            texts.add(out["messages"][0].content)
        assert len(texts) == 6

    @pytest.mark.parametrize("reason", ["payment", "post_payment", "upset", "safety", "error", "requested"])
    async def test_no_emojis_when_someone_is_already_unhappy(self, reason):
        out = await triage.node_human_escalation(_state(handoff_reason=reason))
        text = out["messages"][0].content
        for ch in text:
            assert ord(ch) < 0x2190, f"emoji-range char {ch!r} in the {reason} handoff message"

    async def test_an_unknown_reason_falls_back_rather_than_crashing(self):
        out = await triage.node_human_escalation(_state(handoff_reason="something_new"))
        assert out["messages"][0].content.strip()

    async def test_the_second_message_is_a_short_holding_line_not_the_notice_again(self):
        first = await triage.node_human_escalation(_state(handoff_reason="payment"))
        second = await triage.node_human_escalation(
            _state(handoff_reason="payment", handoff_notified=True)
        )
        assert second["messages"][0].content != first["messages"][0].content
        assert len(second["messages"][0].content) < len(first["messages"][0].content)
        # Still held — a holding line must not look like a release.
        assert second["handoff_active"] is True

    async def test_the_announcement_carries_a_timing_expectation(self):
        out = await triage.node_human_escalation(_state(handoff_reason="requested"))
        text = out["messages"][0].content.lower()
        assert "working hour" in text or "working day" in text


class TestItCannotDeadEnd:
    async def test_a_forgotten_release_is_picked_up_by_the_max_hold(self, monkeypatch):
        monkeypatch.setattr(settings, "HANDOFF_MAX_HOLD_HOURS", 1.0)

        captured = {}

        async def _never_called(*a, **kw):
            raise AssertionError("triage must not need the LLM to auto-release")

        # Stale hold: started two hours ago, limit is one.
        state = _state(
            handoff_active=True,
            handoff_started_at=time.time() - 7200,
            handoff_notified=True,
            handoff_reason="payment",
            messages=[("user", "Confirm & pay [CONFIRM_CHECKOUT]")],
        )
        # The confirm fast path returns before any LLM call, so we can observe the release merged in.
        monkeypatch.setattr(triage, "execute_vendor_agnostic_node", _never_called)
        out = await triage.node_triage(state)
        captured.update(out)

        assert captured["handoff_active"] is False
        assert captured["requires_human_handoff"] is False
        assert any("Auto-released" in n for n in captured["handoff_notes"])

    async def test_zero_disables_the_safety_net(self, monkeypatch):
        monkeypatch.setattr(settings, "HANDOFF_MAX_HOLD_HOURS", 0.0)
        state = _state(
            handoff_active=True,
            handoff_started_at=time.time() - 90000,
            messages=[("user", "Confirm & pay [CONFIRM_CHECKOUT]")],
        )
        out = await triage.node_triage(state)
        assert "handoff_active" not in out or out.get("handoff_active") is not False


class TestContextComesBackOnlyFromTheReleaseNote:
    """
    The colleague is on another number, so nothing they said is visible here. handoff_notes is the
    entire channel — and the agent must be told plainly that it wasn't part of that conversation.
    """

    def test_no_notes_means_no_block_rather_than_a_claim_of_knowledge(self):
        from src.graph.nodes.sales import _build_handoff_block
        assert _build_handoff_block({}) == ""
        assert _build_handoff_block({"handoff_notes": []}) == ""

    def test_a_note_is_injected_with_an_instruction_not_to_re_ask(self):
        from src.graph.nodes.sales import _build_handoff_block
        block = _build_handoff_block({"handoff_notes": ["Paid by UPI on the phone, order confirmed."]})
        assert "Paid by UPI on the phone" in block
        assert "not part of it" in block
        assert "repeat" in block

    def test_only_the_most_recent_notes_are_injected(self):
        from src.graph.nodes.sales import _build_handoff_block
        block = _build_handoff_block({"handoff_notes": [f"note {i}" for i in range(10)]})
        assert "note 9" in block
        assert "note 0" not in block

    @pytest.mark.parametrize("name", [
        "HIGH_INTENT_PROMPT", "WINDOW_SHOPPER_PROMPT", "PROBLEM_SOLVER_PROMPT",
        "B2B_PROMPT", "SUPPORT_PROMPT", "REWARM_PROMPT",
    ])
    def test_every_selling_prompt_can_receive_the_handoff_context(self, name):
        # sales.py filters variables per template, so an undeclared block is silently dropped.
        from src.logic import prompts
        assert "handoff_block" in getattr(prompts, name).input_variables


class TestALeadAndAnEscalationAreTwoDifferentThings:
    """
    One payload used to fire for both, so the sheet filled with rows holding nothing but a phone
    number and the word HUMAN_ESCALATION — a salesperson looking for callbacks had to read past every
    escalation to find one. They are two KINDS now, in one sheet and one inbox (the client's call:
    two tabs is two places to forget to look), so the team filters instead of hunting.
    """

    LEAD = {
        "current_archetype": "SALES_HIGH_INTENT",
        "primary_interest": "Smart Door Lock Base",
        "city": "Kochi",
        "language_preference": "English",
    }
    HELD = {**LEAD, "requires_human_handoff": True, "handoff_active": True, "handoff_reason": "payment"}

    def test_a_thread_with_a_product_interest_is_a_lead(self):
        from src.services.crm_handoff import CRMHandoffService
        row = CRMHandoffService.build_summary("91", self.LEAD)
        assert row["kind"] == "LEAD"
        assert row["escalation_reason"] is None

    def test_somebody_waiting_for_a_person_is_an_escalation_with_the_reason(self):
        from src.services.crm_handoff import CRMHandoffService
        row = CRMHandoffService.build_summary("91", self.HELD)
        assert row["kind"] == "ESCALATION"
        assert row["escalation_reason"] == "payment"

    def test_support_and_a_thread_with_no_interest_are_not_leads_either(self):
        from src.services.crm_handoff import CRMHandoffService
        support = CRMHandoffService.build_summary("91", {"current_archetype": "POST_SALE_SUPPORT"})
        assert (support["kind"], support["route"]) == ("SUPPORT", "SUPPORT")
        bare = CRMHandoffService.build_summary("91", {"current_archetype": "SALES_WINDOW_SHOPPER"})
        assert bare["kind"] == "PARTIAL"

    def test_an_explicit_kind_wins_over_inference(self):
        # The delivery blocks force the kind so each dedup flag governs its own row: a hot lead that
        # later needs a person belongs in the sheet twice, once as each.
        from src.services.crm_handoff import CRMHandoffService, KIND_LEAD
        assert CRMHandoffService.build_summary("91", self.HELD, kind=KIND_LEAD)["kind"] == "LEAD"

    @pytest.mark.parametrize("dropped", [
        "property_type", "budget", "timeline", "preferred_contact_time",
        "deferred_purchase_intent", "history_summary",
    ])
    def test_the_columns_that_were_always_empty_are_gone(self, dropped):
        from src.services.crm_handoff import CRMHandoffService
        assert dropped not in CRMHandoffService.build_summary("91", self.HELD)

    @pytest.mark.parametrize("column", ["kind", "stage", "order", "escalation_reason", "summary"])
    def test_and_what_code_always_knows_took_their_place(self, column):
        from src.services.crm_handoff import CRMHandoffService
        assert column in CRMHandoffService.build_summary("91", self.HELD)

    async def test_everything_goes_to_one_destination(self, monkeypatch):
        from src.services import lead_sink
        from src.services.crm_handoff import CRMHandoffService
        posted = []

        class _Resp:
            def raise_for_status(self):
                return None

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json):
                posted.append((url, json["kind"]))
                return _Resp()

        monkeypatch.setattr(lead_sink.settings, "LEADS_WEBHOOK_URL", "https://example.test/hook")
        monkeypatch.setattr(lead_sink.settings, "SMTP_HOST", "")
        monkeypatch.setattr(lead_sink.httpx, "AsyncClient", lambda **kw: _Client())
        for state in (self.LEAD, self.HELD):
            await lead_sink.LeadSink.deliver_lead(CRMHandoffService.build_summary("91", state))
        assert posted == [
            ("https://example.test/hook", "LEAD"),
            ("https://example.test/hook", "ESCALATION"),
        ]

    async def test_nothing_configured_is_still_silent_rather_than_an_error(self, monkeypatch):
        from src.services import lead_sink
        from src.services.crm_handoff import CRMHandoffService
        monkeypatch.setattr(lead_sink.settings, "LEADS_WEBHOOK_URL", "")
        monkeypatch.setattr(lead_sink.settings, "SMTP_HOST", "")
        result = await lead_sink.LeadSink.deliver_lead(CRMHandoffService.build_summary("91", self.HELD))
        assert result == {"webhook": False, "email": False}


class TestTheEmailIsWrittenToBeReadOnAPhone:
    """
    The old body was every key as `k: v` followed by the entire payload as indented JSON, so the one
    line that mattered — what to do next — sat under two hundred characters of punctuation. One
    format for every kind now: who and which kind in the subject, the facts a salesperson acts on,
    the code-built summary, and for anything waiting on a person the exact handback command.
    """

    def _row(self, **over):
        from src.services.crm_handoff import CRMHandoffService
        state = {
            "current_archetype": "SALES_HIGH_INTENT",
            "primary_interest": "Smart Door Lock Base",
            "city": "Kochi",
            "customer_name": "Anil",
            "language_preference": "English",
            "pain_points": ["front door safety"],
            "pending_order": {"product_summary": "1 x Smart Door Lock Base", "amount": 19500.0},
            "messages": [("user", "worried about my front door")],
            **over,
        }
        return CRMHandoffService.build_summary("919812345678", state)

    def test_a_lead_reads_as_a_lead_and_carries_no_json(self):
        from src.services.lead_sink import LeadSink
        subject, body = LeadSink.build_email(self._row())
        assert subject == "[Otohom lead] Anil — 919812345678"
        assert "captured a lead" in body
        assert "{" not in body and "}" not in body, "no JSON in an inbox"
        # Labels are padded to a fixed width, so assert on the pieces rather than the spacing.
        assert "Interested in" in body and "Smart Door Lock Base" in body
        assert "₹19,500" in body
        assert "#done" not in body, "a lead needs no handback instruction"

    def test_an_escalation_says_why_and_how_to_hand_it_back(self, monkeypatch):
        from src.services import lead_sink
        monkeypatch.setattr(lead_sink.settings, "STAFF_WHATSAPP_NUMBERS", "9812345678")
        subject, body = lead_sink.LeadSink.build_email(
            self._row(requires_human_handoff=True, handoff_active=True, handoff_reason="payment")
        )
        assert subject == "[Otohom escalation · payment] Anil — 919812345678"
        assert "waiting for a person" in body
        assert "Reason" in body and "payment" in body
        assert "#done 919812345678" in body

    def test_it_falls_back_to_the_script_when_no_allowlist_is_set(self, monkeypatch):
        from src.services import lead_sink
        monkeypatch.setattr(lead_sink.settings, "STAFF_WHATSAPP_NUMBERS", "")
        _, body = lead_sink.LeadSink.build_email(
            self._row(requires_human_handoff=True, handoff_reason="upset")
        )
        assert "resolve_handoff" in body and "#done" not in body

    def test_empty_fields_are_left_out_rather_than_printed_blank(self):
        from src.services.lead_sink import LeadSink
        _, body = LeadSink.build_email(
            {"kind": "PARTIAL", "mobile_number": "919812345678", "summary": ""}
        )
        assert "City" not in body and "Order" not in body
        assert "919812345678" in body

    def test_the_summary_is_indented_under_its_own_heading(self):
        from src.services.lead_sink import LeadSink
        _, body = LeadSink.build_email(self._row())
        assert "\nSummary\n  Asked about:" in body


class TestTheSummaryColumnFinallyFills:
    """
    `conversation_summary` was declared in state.py and written by NOTHING, anywhere in the repo —
    read once in crm_handoff and never set — so the sheet's Summary column had always been blank.
    It is built in code rather than by a model: this runs on the customer's turn, so an extra LLM
    call would be latency they wait through, and a paraphrase of a sale is exactly the kind of text
    that quietly invents a commitment.
    """

    FULL = {
        "primary_interest": "Smart Door Lock Base, Video Door Phone",
        "city": "Kochi",
        "pain_points": ["front door safety", "Front door safety", "keys with the maid"],
        "pending_order": {"product_summary": "1 x Smart Door Lock Base", "amount": 19500.0},
        "messages": [
            ("user", "worried about my front door"),
            ("assistant", "ignored"),
            ("user", "can I see the price"),
        ],
        "handoff_notes": ["Card blocked for international; paid by UPI on the phone."],
    }

    def test_it_names_the_product_the_concerns_the_stage_and_the_order(self):
        from src.services.crm_handoff import build_digest
        digest = build_digest(self.FULL)
        assert "Smart Door Lock Base" in digest
        assert "In Kochi" in digest
        assert "front door safety; keys with the maid" in digest   # deduped, first spelling kept
        assert "saw the price" in digest
        assert "19,500" in digest

    def test_it_quotes_the_customer_in_their_own_words(self):
        # What a salesperson wants on a callback is what the customer actually said, not a paraphrase.
        from src.services.crm_handoff import build_digest
        digest = build_digest(self.FULL)
        assert '"worried about my front door"' in digest
        assert '"can I see the price"' in digest
        assert "ignored" not in digest, "the agent's own turns are not the customer's words"

    def test_it_carries_the_colleagues_note_so_the_sheet_shows_the_outcome(self):
        from src.services.crm_handoff import build_digest
        assert "paid by UPI" in build_digest(self.FULL)

    def test_a_bare_state_produces_something_rather_than_crashing(self):
        from src.services.crm_handoff import build_digest
        digest = build_digest({})
        assert "browsing" in digest
        assert "₹" not in digest, "no order means no figure"

    def test_it_stays_small_enough_for_a_spreadsheet_cell(self):
        from src.services.crm_handoff import build_digest
        long_state = {
            **self.FULL,
            "messages": [("user", "please " * 400)],
            "handoff_notes": ["x" * 2000],
        }
        assert len(build_digest(long_state)) < 1200

    def test_the_stage_is_a_phrase_a_salesperson_can_read(self):
        from src.services.crm_handoff import stage_phrase
        assert stage_phrase({}) == "browsing"
        assert stage_phrase({"primary_interest": "lock"}) == "asked about a product"
        assert stage_phrase({"consult_stage": 2}) == "chose a product"
        assert stage_phrase({"pending_order": {"amount": 1}}) == "saw the price"
        assert stage_phrase({"pending_order": {"amount": 1}, "checkout_confirmed": True}) == "payment link sent"
        assert stage_phrase({"last_payment_status": "paid"}) == "paid"
        assert stage_phrase({"last_payment_status": "failed", "payment_failure_count": 3}) == "payment failed (3x)"

    def test_pain_points_stop_repeating_themselves(self):
        from src.core.text import dedupe_keeping_first
        got = dedupe_keeping_first(["front door safety", "Front door safety", "FRONT DOOR SAFETY", "keys"])
        assert got == ["front door safety", "keys"]
        assert dedupe_keeping_first([f"concern {i}" for i in range(20)]) == [f"concern {i}" for i in range(8)]
        assert dedupe_keeping_first(["  ", None, "real"]) == ["real"]


class TestAColleagueCanHandTheThreadBackFromWhatsApp:
    """
    Before this, `resolve_handoff.py --note` was the only channel and it needs a terminal — so in
    practice a hold ran to the 24-hour safety net and the outcome was never recorded at all. A
    colleague has a phone, so the same thing is now one message to the agent's own number.

    The allowlist is the authorisation, and that is the part worth testing hardest: anyone who
    learned the syntax could otherwise release any hold on any thread.
    """

    STAFF = "919812345678"
    CUSTOMER = "919812345678"

    @staticmethod
    def _fakes(monkeypatch, held=True, allowlist=None):
        from src.tasks import processing

        sent, updates = [], []

        class _Snap:
            values = {
                "handoff_active": held,
                "handoff_reason": "payment",
                "handoff_notes": ["earlier note"],
                "messages": [("user", "hi")],
            }

        class _Graph:
            async def aget_state(self, config):
                return _Snap()

            async def aupdate_state(self, config, values):
                updates.append((config["configurable"]["thread_id"], values))

        class _WA:
            async def dispatch_message(self, **kw):
                sent.append((kw["thread_id"], kw["text"]))
                return True

        monkeypatch.setattr(processing, "_graph_app", _Graph())
        monkeypatch.setattr(processing, "_whatsapp", _WA())
        monkeypatch.setattr(
            processing.settings, "STAFF_WHATSAPP_NUMBERS",
            allowlist if allowlist is not None else TestAColleagueCanHandTheThreadBackFromWhatsApp.STAFF,
        )
        return processing, sent, updates

    @staticmethod
    def _msg(sender, body):
        return {"from": sender, "type": "text", "id": "wamid.X", "text": {"body": body}}

    async def test_done_releases_the_target_thread_and_never_the_senders_own(self, monkeypatch):
        processing, sent, updates = self._fakes(monkeypatch)
        consumed = await processing._handle_staff_command(
            self.STAFF, self._msg(self.STAFF, f"#done {self.CUSTOMER} Paid by UPI, order confirmed."), "[t]"
        )
        assert consumed is True
        assert [t for t, _ in updates] == [self.CUSTOMER], "the colleague's own thread must not be written"
        written = updates[0][1]
        assert written["handoff_active"] is False
        assert written["handoff_notes"][-1] == "Paid by UPI, order confirmed."
        # Both dedup flags reset, so a later escalation or lead is delivered again.
        assert written["lead_sent"] is False and written["escalation_sent"] is False
        # The colleague is told what happened; the customer is not messaged by #done.
        assert [t for t, _ in sent] == [self.STAFF]

    async def test_back_also_tells_the_customer_the_agent_is_back(self, monkeypatch):
        processing, sent, updates = self._fakes(monkeypatch)
        await processing._handle_staff_command(
            self.STAFF, self._msg(self.STAFF, f"#back {self.CUSTOMER} Site visit booked Thursday."), "[t]"
        )
        recipients = [t for t, _ in sent]
        assert self.CUSTOMER in recipients and self.STAFF in recipients
        assert any("back with you" in text for t, text in sent if t == self.CUSTOMER)

    async def test_status_changes_nothing(self, monkeypatch):
        processing, sent, updates = self._fakes(monkeypatch)
        await processing._handle_staff_command(
            self.STAFF, self._msg(self.STAFF, f"#status {self.CUSTOMER}"), "[t]"
        )
        assert updates == []
        assert "held by a person" in sent[0][1]

    async def test_the_same_text_from_anyone_else_is_an_ordinary_customer_message(self, monkeypatch):
        # Not an error and not a hint: a stranger must not learn that the commands exist.
        processing, sent, updates = self._fakes(monkeypatch)
        consumed = await processing._handle_staff_command(
            "919999999999", self._msg("919999999999", f"#done {self.CUSTOMER} let me in"), "[t]"
        )
        assert consumed is False
        assert sent == [] and updates == []

    async def test_the_commands_do_not_exist_when_no_allowlist_is_configured(self, monkeypatch):
        processing, sent, updates = self._fakes(monkeypatch)
        monkeypatch.setattr(processing.settings, "STAFF_WHATSAPP_NUMBERS", "")
        assert await processing._handle_staff_command(
            self.STAFF, self._msg(self.STAFF, f"#done {self.CUSTOMER} anything"), "[t]"
        ) is False

    async def test_a_command_with_no_number_gets_the_help_text(self, monkeypatch):
        processing, sent, updates = self._fakes(monkeypatch)
        await processing._handle_staff_command(self.STAFF, self._msg(self.STAFF, "#done"), "[t]")
        assert "#done <number>" in sent[0][1]
        assert updates == []

    async def test_releasing_a_thread_nobody_is_holding_says_so(self, monkeypatch):
        processing, sent, updates = self._fakes(monkeypatch, held=False)
        await processing._handle_staff_command(
            self.STAFF, self._msg(self.STAFF, f"#done {self.CUSTOMER} whatever"), "[t]"
        )
        assert "not currently held" in sent[0][1]
        assert updates == []

    async def test_a_release_with_no_note_is_refused(self, monkeypatch):
        # The note is the entire channel back to the agent; releasing without one hands it nothing.
        processing, sent, updates = self._fakes(monkeypatch)
        await processing._handle_staff_command(
            self.STAFF, self._msg(self.STAFF, f"#done {self.CUSTOMER}"), "[t]"
        )
        assert "what happened" in sent[0][1]
        assert updates == []

    async def test_a_staff_number_can_still_have_an_ordinary_conversation(self, monkeypatch):
        processing, sent, updates = self._fakes(monkeypatch)
        assert await processing._handle_staff_command(
            self.STAFF, self._msg(self.STAFF, "hi, what do smart switches cost?"), "[t]"
        ) is False

    def test_the_parser_accepts_a_number_typed_the_way_a_phone_offers_it(self):
        from src.services.handoff_control import parse_staff_command
        assert parse_staff_command("#back +91 98123 45678 Sorted") == ("back", "919812345678", "Sorted")
        assert parse_staff_command("#DONE 919812345678  fixed  it ") == ("done", "919812345678", "fixed it")
        assert parse_staff_command("hello") is None

    async def test_the_allowlist_matches_however_the_colleague_wrote_their_number(self, monkeypatch):
        """
        The client supplied "9812345678, 9887654321" — the way anyone says their own number — while
        Meta delivers `from` as "919812345678". Comparing the raw strings meant a correctly configured
        allowlist matched nothing, and silently, because a non-match is indistinguishable by design
        from an ordinary customer message. Matching is on the last ten digits.
        """
        processing, sent, updates = self._fakes(monkeypatch, allowlist="9812345678, 9887654321")
        for sender in ("919812345678", "919887654321"):
            consumed = await processing._handle_staff_command(
                sender, self._msg(sender, f"#status {self.CUSTOMER}"), "[t]"
            )
            assert consumed is True, sender
        assert await processing._handle_staff_command(
            "919999999999", self._msg("919999999999", f"#status {self.CUSTOMER}"), "[t]"
        ) is False

    async def test_a_number_typed_without_its_country_code_still_finds_the_thread(self, monkeypatch):
        """
        The live failure: `#done 9812345678` answered "No conversation found" while the customer was
        visibly still being held, because the conversation is keyed `919812345678`. The command looked
        broken when only the number was. The country prefix comes from the colleague's OWN number, so
        it is not hard-coded to India.
        """
        processing, sent, updates = self._fakes(monkeypatch)
        await processing._handle_staff_command(
            self.STAFF, self._msg(self.STAFF, "#done 9812345678 Sorted on the phone"), "[t]"
        )
        assert [t for t, _ in updates] == ["919812345678"]
        assert "Released 919812345678" in sent[0][1]

    async def test_a_number_with_no_conversation_says_what_it_looked_for(self, monkeypatch):
        processing, sent, updates = self._fakes(monkeypatch)

        class _Empty:
            values = {}

        class _Graph:
            async def aget_state(self, config):
                return _Empty()

        monkeypatch.setattr(processing, "_graph_app", _Graph())
        await processing._handle_staff_command(
            self.STAFF, self._msg(self.STAFF, "#done 9812345678 anything"), "[t]"
        )
        assert "couldn't find a conversation for 9812345678" in sent[0][1]
        assert "919812345678" in sent[0][1], "say what else was tried, so the number can be checked"
        assert updates == []

    def test_a_space_after_the_hash_is_still_a_command(self):
        # "# done 9812345678" went through to the customer path live and got the holding line.
        from src.services.handoff_control import parse_staff_command, looks_like_staff_command
        assert looks_like_staff_command("# done 9812345678")
        assert parse_staff_command("#  status  919812345678") == ("status", "919812345678", "")

    def test_the_candidates_are_tried_most_likely_first(self):
        from src.services.handoff_control import thread_candidates
        assert thread_candidates("9812345678", "919812345678") == ["919812345678", "9812345678"]
        assert thread_candidates("919812345678", "919812345678") == ["919812345678", "9812345678"]
        # No sender context: nothing to prefix with, so the digits stand as given.
        assert thread_candidates("9812345678") == ["9812345678"]
        assert thread_candidates("") == []

    def test_both_channels_call_one_implementation(self):
        # Two ways in, one release path — otherwise the CLI and WhatsApp drift and only one of them
        # resets the dedup flags.
        import inspect
        from src.scripts import resolve_handoff
        from src.tasks import processing
        assert "release_handoff" in inspect.getsource(resolve_handoff.resolve)
        assert "release_handoff" in inspect.getsource(processing._handle_staff_command)
