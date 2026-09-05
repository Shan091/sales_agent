"""
Autonomy tests — proven in BOTH directions.

The design promise is narrow and easy to get wrong in either direction, so both halves are
asserted here:

  ON  (AGENT_FULL_AUTONOMY=True): pricing / quoting / closing are the agent's own work and do
      NOT route to a human. The checkout capability is present in the prompt.
  OFF (AGENT_FULL_AUTONOMY=False): the original Otohom client behaviour is restored — no
      checkout instructions reach the model at all, so the closing play falls back to the
      human handoff.
  BOTH: the CRITICAL escalation safety valve is still wired. Human escalation was narrowed,
      never deleted — a repeated payment failure, a persistent human request, a post-payment
      dispute or a safety/legal ask must still reach a person in either mode.

No DB / LLM / Redis. Everything asserted here is a pure function.
"""
import pytest

from config.settings import settings
from src.graph.nodes.sales import _build_pricing_policy_block
from src.graph.workflow import route_after_triage, route_after_rag
from src.logic import discounts
from src.logic import prompts


def _state(**kw):
    base = {
        "current_archetype": None,
        "requires_human_handoff": False,
        "data_routing_flag": "NONE",
    }
    base.update(kw)
    return base


@pytest.fixture
def autonomy_on(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_FULL_AUTONOMY", True)


@pytest.fixture
def autonomy_off(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_FULL_AUTONOMY", False)


class TestPricingPolicyIsFlagGated:
    """Exactly one pricing policy reaches the model per turn. Money is the one topic where a mixed
    signal is expensive, so the two blocks must be mutually exclusive — never both, never neither."""

    def test_on_grants_full_autonomy_and_shows_the_closed_offer_menu(self, autonomy_on):
        block = _build_pricing_policy_block({})
        assert "you own the sale end to end" in block
        # The model must see the offer ids as a closed set it can choose from.
        for offer_id in discounts.OFFER_IDS:
            if offer_id != "NONE":
                assert offer_id in block
        # Autonomy is about WHAT is sold, never about what it costs.
        assert "NEVER invent, type, calculate, estimate, hint at, or repeat a rupee amount" in block
        # Negotiation is explicitly in scope, and explicitly bounded.
        assert "PRICE OBJECTION" in block
        assert "do NOT invent a bigger discount" in block
        # The legacy prohibition must NOT also be present.
        assert "handled by the Otohom team, not by you" not in block

    def test_off_restores_the_merchant_pricing_policy(self, autonomy_off):
        block = _build_pricing_policy_block({})
        assert "handled by the Otohom team, not by you" in block
        assert "Never quote, estimate, or hint at a price" in block
        assert "lead_ready_for_handoff = True" in block
        # No checkout authority leaks through.
        assert "you own the sale end to end" not in block
        assert "Confirm & pay" not in block

    def test_off_never_shows_the_offer_menu(self, autonomy_off):
        """A discount the agent is forbidden to give must not even be visible to it."""
        block = _build_pricing_policy_block({})
        for offer_id in discounts.OFFER_IDS:
            if offer_id != "NONE":
                assert offer_id not in block

    def test_a_live_link_suppresses_re_proposing_the_order(self, autonomy_on):
        block = _build_pricing_policy_block({"payment_link_sent": True, "last_payment_status": "link_created"})
        assert "already been sent" in block

    def test_a_paid_order_suppresses_re_proposing_the_order(self, autonomy_on):
        block = _build_pricing_policy_block({"payment_link_sent": True, "last_payment_status": "paid"})
        # Asserted on the NOTE's own opening words. "already PAID" was too loose: the standing policy
        # text also discusses what to do when an order is already paid, so the fragment matched
        # whether or not the runtime NOTE had been appended at all.
        assert "PAID and settled" in block

    def test_a_failed_payment_does_not_suppress_a_retry(self, autonomy_on):
        # After a decline the customer must still be able to complete the purchase.
        block = _build_pricing_policy_block({"payment_link_sent": True, "last_payment_status": "failed"})
        assert "already been sent" not in block
        assert "PAID and settled" not in block


class TestEverySellingPromptCarriesThePricingPolicy:
    """A prompt that can reach a buying moment but does not declare {pricing_policy_block} would be
    told nothing about money at all — free to improvise a figure. sales.py filters the variable set
    per template, so a missing declaration fails silently rather than raising."""

    @pytest.mark.parametrize("name", [
        "HIGH_INTENT_PROMPT", "WINDOW_SHOPPER_PROMPT", "PROBLEM_SOLVER_PROMPT",
        "B2B_PROMPT", "SUPPORT_PROMPT", "REWARM_PROMPT",
    ])
    def test_selling_prompt_declares_the_pricing_policy_variable(self, name):
        assert "pricing_policy_block" in getattr(prompts, name).input_variables

    def test_no_prompt_hardcodes_a_pricing_prohibition(self):
        """The shared guardrails must stay policy-neutral on money, or the autonomy block would be
        arguing with them inside the same system prompt."""
        assert "Never quote" not in prompts.GUARDRAIL_RULES
        assert "never negotiate" not in prompts.GUARDRAIL_RULES.lower()


class TestTheModelIsNeverShownAFigureItCouldRepeat:
    """
    The agent is forbidden to state a percentage. The strongest way to hold that line is to not
    show it one — a number in the prompt is a number that can be echoed to a customer before code
    has verified the order even qualifies for it.
    """

    def test_the_offer_menu_carries_no_percentages(self):
        menu = discounts.offer_menu_for_prompt()
        assert "%" not in menu
        # Offer ids are still published — the model has to be able to name one.
        for offer_id in discounts.OFFERS:
            if offer_id != "NONE":
                assert offer_id in menu

    def test_offer_labels_carry_no_percentages(self):
        for offer_id, spec in discounts.OFFERS.items():
            assert "%" not in spec["label"], offer_id

    def test_the_quote_does_print_the_real_percentage(self):
        """Code prints what the model may not: the figure appears where it was computed."""
        items = [{"sku": "A", "qty": 1, "unit_price": 5000.0, "installation_fee": 0.0, "line_total": 5000.0},
                 {"sku": "B", "qty": 1, "unit_price": 5000.0, "installation_fee": 0.0, "line_total": 5000.0}]
        text = discounts.format_quote_message(discounts.apply_offer(items, "BUNDLE8"))
        assert "8% off" in text


class TestTheCatalogueIsAClosedSet:
    """
    The model is told to copy skus verbatim, so it must actually be shown them. Left unshown, it
    invents plausible-but-unresolvable strings and the customer silently gets no quote.
    """

    def test_catalogue_block_lists_every_name_and_demands_verbatim_copying(self):
        from src.graph.nodes.sales import catalogue_for_prompt
        block = catalogue_for_prompt(["6 SW", "Smart Door Lock Premium"])
        assert "- 6 SW" in block
        assert "- Smart Door Lock Premium" in block
        assert "character-for-character" in block

    def test_an_unreadable_catalogue_yields_no_block_rather_than_a_lie(self):
        from src.graph.nodes.sales import catalogue_for_prompt
        assert catalogue_for_prompt([]) == ""
        assert catalogue_for_prompt(None) == ""

    def test_autonomy_off_shows_no_catalogue(self, autonomy_off):
        block = _build_pricing_policy_block(_state(), ["6 SW"])
        assert "PRICEABLE CATALOGUE" not in block

    def test_the_prompt_points_at_the_catalogue_instead_of_asking_for_exact_names(self):
        assert "PRICEABLE CATALOGUE" in prompts.PRICING_AUTONOMY
        # The old instruction asked for "exact catalogue names" without ever supplying them.
        assert "(exact catalogue names)" not in prompts.PRICING_AUTONOMY


class TestConversationStyleRules:
    """The interaction rules the live session showed being broken."""

    def test_one_question_per_message_is_stated(self):
        assert "ONE QUESTION PER MESSAGE" in prompts.CONVERSATION_STYLE

    def test_typing_is_always_allowed(self):
        style = prompts.CONVERSATION_STYLE
        assert "ACCELERATOR, NOT A CAGE" in style
        assert "in your own words" in style

    def test_labels_must_be_complete_phrases(self):
        assert "FINISHED thought" in prompts.CONVERSATION_STYLE

    def test_need_comes_before_product(self):
        assert "UNDERSTAND THE NEED BEFORE YOU RECOMMEND" in prompts.CONVERSATION_STYLE


class TestMessageCraft:
    """
    The live transcripts showed walls of text, praise padding, and turns spent asking permission to
    be helpful. All three are prompt-level rules, so they are asserted at prompt level.
    """

    def test_short_paragraphs_are_required(self):
        assert "SHORT PARAGRAPHS, ALWAYS" in prompts.CONVERSATION_STYLE
        assert "ONE IDEA PER PARAGRAPH" in prompts.CONVERSATION_STYLE

    def test_whatsapp_formatting_is_taught_structurally(self):
        style = prompts.CONVERSATION_STYLE
        assert "*bold*" in style and "_italic_" in style
        # The failure mode from the transcript: "*1. The lock:*" rendered as literal asterisks.
        assert "next to a number or punctuation" in style

    def test_praise_padding_is_banned(self):
        style = prompts.CONVERSATION_STYLE
        assert "NEVER OPEN BY EVALUATING THEIR CHOICE" in style
        # The specific phrases the live transcripts produced, named so the rule isn't abstract.
        for filler in ("Great choice!", "Wonderful!", "great way to boost your security"):
            assert filler in style, f"the rule should name {filler!r} explicitly"

    def test_the_wasted_turn_is_shown_as_a_worked_example(self):
        """An abstract "don't stall" rule didn't hold; the WRONG/RIGHT pair from the real
        transcript did."""
        style = prompts.CONVERSATION_STYLE
        assert "WRONG:" in style and "RIGHT:" in style
        # The lesson rather than the product: the wrong version asks permission and puts two
        # versions of one product on screen; the right one recommends and lets the options confirm.
        assert "RECOMMENDS one" in style
        assert "Yes, that one / What's different? / Something else" in style

    def test_no_worked_example_teaches_a_menu_of_two_price_tiers(self):
        """
        Handing the customer both models of one product is the "raw price list" the behavioural
        notes warn about: it asks them to choose on price before they have been shown a price, and
        it spends the step-up beat — which the system renders itself, with the exact difference
        attached — a full turn early. The example here used to teach exactly that.
        """
        style = prompts.CONVERSATION_STYLE
        assert "does NOT do is put two versions of the same product side by side" in style
        # Every worked example is a non-lock product on purpose: with a Smart Door Lock in all of
        # them the model learns the rule as being about locks, and the same sequence has to read
        # right for a panel, a camera or a curtain motor.
        for menu in ("Base lock", "Premium lock", "Smart Door Lock Base", "Smart Door Lock Premium"):
            assert menu not in style, f"{menu!r} teaches the price-tier menu back into discovery"

    def test_asking_permission_to_be_useful_is_banned(self):
        """"We have two models — would you like me to tell you more?" is a wasted turn: they
        already said what they wanted."""
        style = prompts.CONVERSATION_STYLE
        assert "NEVER ASK PERMISSION TO BE USEFUL" in style
        assert "would you like to know more?" in style

    def test_the_brand_voice_is_defined_not_left_to_chance(self):
        style = prompts.CONVERSATION_STYLE
        assert "BRAND VOICE" in style
        assert "Specific, not adjectival" in style

    def test_the_banned_emoji_is_named(self):
        assert "🙏" in prompts.CONVERSATION_STYLE


class TestTheOpeningTurnIsDiscoveryNotACatalogue:
    def _greeting_play(self):
        """
        Only the greeting's OWN instructions — everything before the shared blocks are appended.
        The shared OTOHOM_OVERVIEW legitimately discusses the wiring story (gated on relevance);
        what matters here is that the opening turn itself doesn't lead with it.
        """
        text = "\n".join(
            m.prompt.template for m in prompts.GENERAL_GREETING_PROMPT.messages
            if hasattr(m, "prompt")
        )
        marker = "WHO OTOHOM IS FOR:"
        return text.split(marker)[0]

    def test_the_greeting_offers_no_product_category_menu(self):
        text = self._greeting_play()
        # The options the live screenshot showed — a taxonomy the customer must self-diagnose into.
        assert "Smart switches & lighting" not in text
        assert "Curtains & blinds" not in text
        assert "NO product names in the options" in text

    def test_the_greeting_asks_exactly_one_question(self):
        play = self._greeting_play()
        assert "ONE question mark" in play

    def test_the_greeting_names_the_reasons_as_a_statement_not_as_questions(self):
        """Naming the three reasons is what makes the list worth opening — but it has to be a
        statement, or one question silently becomes three."""
        play = self._greeting_play()
        assert "NAME the reasons" in play
        assert "not as questions" in play

    def test_otohom_is_never_described_by_nationality(self):
        """Otohom operates across regions; pinning it to one country is inaccurate and smaller
        than the truth. The prohibition lives in the shared overview so it can't be lost by
        editing one prompt."""
        assert "NEVER describe Otohom by nationality" in prompts.OTOHOM_OVERVIEW
        for claim in ("Made in India", "made in India", "Indian company", "Indian brand", "in India"):
            assert claim not in self._greeting_play(), claim

    def test_the_greeting_invites_a_typed_answer(self):
        assert "type it" in self._greeting_play()

    def test_the_greeting_does_not_pitch_the_wiring_story(self):
        """It answers a concern; opening with it means nobody raised the concern yet."""
        play = self._greeting_play().lower()
        assert "retrofit" not in play
        assert "wall-breaking" not in play

    def test_the_greeting_leads_with_outcomes_people_actually_arrive_with(self):
        play = self._greeting_play().lower()
        assert "bills" in play
        assert "front door" in play


class TestTheWiringPitchIsConditional:
    def test_the_overview_gates_it_on_relevance(self):
        assert "WHEN TO BRING UP THE NO-REWIRING POINT" in prompts.OTOHOM_OVERVIEW
        assert "lead with it" not in prompts.OTOHOM_OVERVIEW

    @pytest.mark.parametrize("name", ["HIGH_INTENT_PROMPT", "B2B_PROMPT"])
    def test_archetypes_no_longer_pitch_it_unconditionally(self, name):
        text = "\n".join(
            m.prompt.template for m in getattr(prompts, name).messages if hasattr(m, "prompt")
        )
        # The plain-language explanation stays; the unconditional trigger goes.
        assert "existing switch wiring" in text
        assert "When the retrofit angle fits" not in text
        assert "When the fit is retrofit" not in text


class TestSalesIntentsNeverEscalate:
    """Autonomy's whole point: money talk is ordinary sales work, not an escalation trigger."""

    @pytest.mark.parametrize("archetype,expected", [
        ("SALES_HIGH_INTENT", "high_intent"),
        ("SALES_WINDOW_SHOPPER", "window_shopper"),
        ("SALES_PROBLEM_SOLVER", "problem_solver"),
        ("B2B_ENTERPRISE", "b2b_enterprise"),
    ])
    def test_pricing_and_closing_intents_route_to_the_agent(self, autonomy_on, archetype, expected):
        assert route_after_triage(_state(current_archetype=archetype)) == expected

    def test_a_checkout_confirmation_turn_stays_with_the_agent(self, autonomy_on):
        s = _state(current_archetype="SALES_HIGH_INTENT", checkout_confirmed=True,
                   pending_order={"amount": 9500.0})
        assert route_after_triage(s) == "high_intent"

    def test_an_active_checkout_does_not_bypass_adversarial_deflection(self, autonomy_on):
        # Precedence must hold even mid-checkout: a hostile turn is deflected locally, never
        # escalated and never allowed to reach retrieval.
        s = _state(current_archetype="MALICIOUS_ADVERSARIAL", data_routing_flag="TECHNICAL_RAG",
                   checkout_confirmed=True, pending_order={"amount": 9500.0})
        assert route_after_triage(s) == "adversarial_block"


class TestCriticalSafetyValveSurvivesInBothModes:
    """Human escalation was NARROWED, not removed. These prove it is still reachable."""

    @pytest.mark.parametrize("autonomy", [True, False])
    def test_repeated_payment_failure_reaches_a_human(self, monkeypatch, autonomy):
        monkeypatch.setattr(settings, "AGENT_FULL_AUTONOMY", autonomy)
        # This is the state _handle_payment_failure writes once MAX_PAYMENT_FAILURES is hit.
        s = _state(current_archetype="SALES_HIGH_INTENT", requires_human_handoff=True,
                   last_payment_status="failed",
                   payment_failure_count=settings.MAX_PAYMENT_FAILURES)
        assert route_after_triage(s) == "human_escalation"

    @pytest.mark.parametrize("autonomy", [True, False])
    def test_persistent_human_request_reaches_a_human(self, monkeypatch, autonomy):
        monkeypatch.setattr(settings, "AGENT_FULL_AUTONOMY", autonomy)
        s = _state(current_archetype="HUMAN_ESCALATION", requires_human_handoff=True)
        assert route_after_triage(s) == "human_escalation"

    @pytest.mark.parametrize("autonomy", [True, False])
    def test_post_payment_dispute_reaches_a_human(self, monkeypatch, autonomy):
        monkeypatch.setattr(settings, "AGENT_FULL_AUTONOMY", autonomy)
        # A refund/dispute after a paid order: triage classifies HUMAN_ESCALATION.
        s = _state(current_archetype="HUMAN_ESCALATION", requires_human_handoff=True,
                   last_payment_status="paid")
        assert route_after_triage(s) == "human_escalation"

    @pytest.mark.parametrize("autonomy", [True, False])
    def test_escalation_flag_wins_over_a_technical_question(self, monkeypatch, autonomy):
        # Precedence: an escalating customer is not sent through RAG first.
        monkeypatch.setattr(settings, "AGENT_FULL_AUTONOMY", autonomy)
        s = _state(current_archetype="SALES_HIGH_INTENT", requires_human_handoff=True,
                   data_routing_flag="TECHNICAL_RAG")
        assert route_after_triage(s) == "human_escalation"

    @pytest.mark.parametrize("autonomy", [True, False])
    def test_valve_is_reachable_after_the_rag_hop_too(self, monkeypatch, autonomy):
        monkeypatch.setattr(settings, "AGENT_FULL_AUTONOMY", autonomy)
        s = _state(current_archetype="SALES_HIGH_INTENT", requires_human_handoff=True)
        assert route_after_rag(s) == "human_escalation"

    @pytest.mark.parametrize("autonomy", [True, False])
    def test_the_calm_first_ask_still_probes_instead_of_escalating(self, monkeypatch, autonomy):
        # The narrowing must not become over-eagerness in the other direction either: a polite
        # first "can I talk to someone?" is probed, not forwarded.
        monkeypatch.setattr(settings, "AGENT_FULL_AUTONOMY", autonomy)
        s = _state(current_archetype="HUMAN_PROBE", requires_human_handoff=False)
        assert route_after_triage(s) == "human_probe"

    def test_the_graph_still_has_both_escalation_nodes_wired(self):
        # A topology regression (deleting the valve outright) would break the client's
        # non-autonomy mode as well, so assert the nodes exist.
        from src.graph.workflow import create_workflow
        nodes = set(create_workflow().nodes)
        assert {"human_escalation", "human_probe", "adversarial_block"} <= nodes
