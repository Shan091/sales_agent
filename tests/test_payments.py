"""
Money-path tests — the bounded / gated / audited guarantees, provable without infrastructure.

These are the tests that back the Track-01 claim "the LLM never decides an amount". Nothing here
touches Postgres, Redis, Razorpay or an LLM: the pricing math, the discount clamp, the payment
guardrail, the webhook signature check and the checkout gate are all pure functions by design,
precisely so they can be verified like this.

Grouped by the property each one defends:
  TestDiscountClamp        — a jailbroken model cannot buy a bigger discount than policy allows.
  TestPriceLineItems       — quantity/SKU proposals cannot inflate an order.
  TestPaymentGuardrail     — a tampered or stale order is refused before any link is minted.
  TestWebhookSignature     — a forged "paid" event is rejected fail-closed.
  TestCheckoutGate         — a link is only ever minted after the explicit tap.
  TestMemoryBlock          — recalled facts are injected safely (and absent when there are none).
"""
import hashlib
import hmac
import re
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from config.settings import settings
from src.core.guardrails import Guardrails
from src.graph.nodes.triage import _is_confirm_checkout, _is_connect_now
from src.logic import discounts
from src.memory.semantic import format_memory_block
from src.services.razorpay_service import RazorpayService


# A trusted catalogue stand-in, shaped like PricingEngine's PriceResult (dicts are accepted
# by the same code paths, which is what makes these tests infra-free).
CATALOGUE = {
    "6 SW": {"product_name": "6 SW", "base_price": 4200.0, "installation_fee": 400.0},
    "Zigbee Hub": {"product_name": "Zigbee Hub", "base_price": 4500.0, "installation_fee": 400.0},
    "Indoor Smart Camera": {"product_name": "Indoor Smart Camera", "base_price": 3500.0, "installation_fee": 500.0},
}


def build_order(items=(("6 SW", 2), ("Zigbee Hub", 1)), offer=None):
    raw = [{"sku": sku, "qty": qty} for sku, qty in items]
    line_items, _unresolved, _notes = discounts.price_line_items(raw, CATALOGUE)
    return discounts.apply_offer(line_items, offer)


class TestDiscountClamp:
    """The single most important property: code owns the discount, not the model."""

    def test_unknown_offer_degrades_to_list_price(self):
        # A model inventing "MEGA90" must not get a discount — the registry is a closed set.
        order = build_order(offer="MEGA90")
        assert order["discount_pct"] == 0.0
        assert order["discount_amount"] == 0.0
        assert order["applied_offer"] is None
        assert any("not in the code-owned registry" in n for n in order["audit_notes"])

    def test_no_offer_can_ever_exceed_the_policy_ceiling(self):
        # Every registry entry, applied to an order that qualifies on paper.
        for offer_id in discounts.OFFER_IDS:
            order = build_order(
                items=(("6 SW", 20), ("Zigbee Hub", 20), ("Indoor Smart Camera", 20)),
                offer=offer_id,
            )
            assert order["discount_pct"] <= settings.MAX_DISCOUNT_PCT + 1e-9, offer_id

    def test_registry_pct_above_ceiling_is_clamped(self, monkeypatch):
        # Simulate a policy tightening (or a bad registry edit): the clamp, not the registry,
        # decides what is honoured.
        monkeypatch.setattr(settings, "MAX_DISCOUNT_PCT", 5.0)
        order = build_order(items=(("6 SW", 1), ("Zigbee Hub", 1), ("Indoor Smart Camera", 1)), offer="BUNDLE10")
        assert order["discount_pct"] == 5.0
        assert any("clamped" in n for n in order["audit_notes"])

    def test_ineligible_offer_is_dropped_not_applied(self):
        # BUNDLE10 needs 3+ distinct products; a single-product order must not receive it.
        order = build_order(items=(("6 SW", 1),), offer="BUNDLE10")
        assert order["discount_pct"] == 0.0
        assert any("distinct products" in n for n in order["audit_notes"])

    def test_project12_requires_the_subtotal_floor(self):
        order = build_order(items=(("6 SW", 1), ("Zigbee Hub", 1)), offer="PROJECT12")
        assert order["discount_pct"] == 0.0
        assert any("subtotal of at least" in n for n in order["audit_notes"])

    def test_installation_fee_is_never_discounted(self):
        order = build_order(items=(("6 SW", 1), ("Zigbee Hub", 1)), offer="BUNDLE8")
        # Discountable value excludes the two installation fees (400 + 400).
        discountable = (4200.0 + 4500.0)
        assert order["discount_amount"] == pytest.approx(discountable * 8 / 100.0, abs=0.01)
        assert order["subtotal"] == pytest.approx(4600.0 + 4900.0, abs=0.01)
        assert order["amount"] == pytest.approx(order["subtotal"] - order["discount_amount"], abs=0.01)
        assert any("installation not discounted" in n for n in order["audit_notes"])

    def test_every_order_carries_its_policy_and_audit_trail(self):
        order = build_order(offer="FESTIVE5")
        assert order["policy"]["max_discount_pct"] == float(settings.MAX_DISCOUNT_PCT)
        assert order["policy"]["order_cap"] == float(settings.RAZORPAY_MAX_AMOUNT)
        assert isinstance(order["audit_notes"], list)


class TestPriceLineItems:
    def test_quantity_is_clamped_to_the_per_line_cap(self):
        line_items, _u, notes = discounts.price_line_items([{"sku": "6 SW", "qty": 9999}], CATALOGUE)
        assert line_items[0]["qty"] == settings.MAX_LINE_QTY
        assert any("clamped" in n for n in notes)

    def test_duplicate_skus_merge_and_stay_capped(self):
        raw = [{"sku": "6 SW", "qty": 15}, {"sku": "6 SW", "qty": 15}]
        line_items, _u, notes = discounts.price_line_items(raw, CATALOGUE)
        assert len(line_items) == 1
        assert line_items[0]["qty"] == settings.MAX_LINE_QTY
        assert any("Merged quantity" in n for n in notes)

    def test_unpriced_sku_is_dropped_not_guessed(self):
        line_items, unresolved, notes = discounts.price_line_items(
            [{"sku": "6 SW", "qty": 1}, {"sku": "Flux Capacitor", "qty": 1}], CATALOGUE
        )
        assert unresolved == ["Flux Capacitor"]
        assert [li["sku"] for li in line_items] == ["6 SW"]
        assert any("Dropped unpriced SKU" in n for n in notes)

    def test_line_total_includes_installation_and_multiplies_by_qty(self):
        line_items, _u, _n = discounts.price_line_items([{"sku": "6 SW", "qty": 3}], CATALOGUE)
        assert line_items[0]["line_total"] == pytest.approx((4200.0 + 400.0) * 3, abs=0.01)

    def test_zero_or_negative_quantity_floors_at_one(self):
        line_items, _u, _n = discounts.price_line_items([{"sku": "6 SW", "qty": -5}], CATALOGUE)
        assert line_items[0]["qty"] == 1


class TestPaymentGuardrail:
    """The last gate before money moves. It must reject anything it cannot re-derive itself."""

    def test_a_clean_code_built_order_passes(self):
        order = build_order(offer="BUNDLE8")
        ok, reasons = Guardrails.validate_payment_request(order, CATALOGUE)
        assert ok, reasons

    def test_empty_or_missing_order_is_refused(self):
        for bad in (None, {}, {"line_items": []}):
            ok, reasons = Guardrails.validate_payment_request(bad, CATALOGUE)
            assert not ok and reasons

    def test_tampered_total_is_refused(self):
        order = build_order()
        order["amount"] = 1.0  # someone edited the checkpoint
        ok, reasons = Guardrails.validate_payment_request(order, CATALOGUE)
        assert not ok
        assert any("Grand total" in r for r in reasons)

    def test_unit_price_not_matching_the_catalogue_is_refused(self):
        order = build_order()
        order["line_items"][0]["unit_price"] = 1.0
        order["line_items"][0]["line_total"] = (1.0 + 400.0) * order["line_items"][0]["qty"]
        order["subtotal"] = round(sum(li["line_total"] for li in order["line_items"]), 2)
        order["amount"] = order["subtotal"]
        ok, reasons = Guardrails.validate_payment_request(order, CATALOGUE)
        assert not ok
        assert any("does not match the catalogue" in r for r in reasons)

    def test_sku_absent_from_the_catalogue_is_refused(self):
        order = build_order()
        ok, reasons = Guardrails.validate_payment_request(order, {"Zigbee Hub": CATALOGUE["Zigbee Hub"]})
        assert not ok
        assert any("not in the trusted price catalogue" in r for r in reasons)

    def test_over_ceiling_discount_is_refused_even_if_internally_consistent(self):
        # An order whose arithmetic is self-consistent but whose discount breaks policy.
        order = build_order()
        discountable = sum(li["unit_price"] * li["qty"] for li in order["line_items"])
        order["discount_pct"] = settings.MAX_DISCOUNT_PCT + 40.0
        order["discount_amount"] = round(discountable * order["discount_pct"] / 100.0, 2)
        order["amount"] = round(order["subtotal"] - order["discount_amount"], 2)
        ok, reasons = Guardrails.validate_payment_request(order, CATALOGUE)
        assert not ok
        assert any("policy ceiling" in r for r in reasons)

    def test_discount_may_not_eat_the_installation_fee(self):
        order = build_order()
        order["discount_pct"] = 0.0
        order["discount_amount"] = order["subtotal"]  # discounting labour too
        order["amount"] = 0.0
        ok, reasons = Guardrails.validate_payment_request(order, CATALOGUE)
        assert not ok

    def test_total_above_the_order_cap_is_refused(self, monkeypatch):
        monkeypatch.setattr(settings, "RAZORPAY_MAX_AMOUNT", 100.0)
        order = build_order()
        ok, reasons = Guardrails.validate_payment_request(order, CATALOGUE)
        assert not ok
        assert any("per-order cap" in r for r in reasons)

    def test_quantity_beyond_the_cap_is_refused(self):
        order = build_order()
        order["line_items"][0]["qty"] = settings.MAX_LINE_QTY + 1
        ok, reasons = Guardrails.validate_payment_request(order, CATALOGUE)
        assert not ok

    def test_without_a_trusted_map_internal_consistency_is_still_enforced(self):
        # The mint-time fallback path (catalogue lookup failed): arithmetic must still hold.
        order = build_order()
        ok, _ = Guardrails.validate_payment_request(order, None)
        assert ok
        order["subtotal"] = order["subtotal"] + 1000.0
        ok, reasons = Guardrails.validate_payment_request(order, None)
        assert not ok
        assert any("Subtotal" in r for r in reasons)


class TestWebhookSignature:
    """A forged 'paid' event must never be believed."""

    BODY = b'{"event":"payment_link.paid","payload":{}}'

    def test_valid_signature_accepted(self, monkeypatch):
        secret = "whsec_test_value"
        monkeypatch.setattr(settings, "RAZORPAY_WEBHOOK_SECRET", secret)
        sig = hmac.new(secret.encode(), self.BODY, hashlib.sha256).hexdigest()
        assert RazorpayService.verify_webhook_signature(self.BODY, sig)

    def test_wrong_signature_rejected(self, monkeypatch):
        monkeypatch.setattr(settings, "RAZORPAY_WEBHOOK_SECRET", "whsec_test_value")
        assert not RazorpayService.verify_webhook_signature(self.BODY, "deadbeef")

    def test_signature_over_different_bytes_rejected(self, monkeypatch):
        secret = "whsec_test_value"
        monkeypatch.setattr(settings, "RAZORPAY_WEBHOOK_SECRET", secret)
        sig = hmac.new(secret.encode(), b'{"event":"something.else"}', hashlib.sha256).hexdigest()
        assert not RazorpayService.verify_webhook_signature(self.BODY, sig)

    def test_missing_secret_fails_closed(self, monkeypatch):
        monkeypatch.setattr(settings, "RAZORPAY_WEBHOOK_SECRET", "")
        sig = hmac.new(b"anything", self.BODY, hashlib.sha256).hexdigest()
        assert not RazorpayService.verify_webhook_signature(self.BODY, sig)

    def test_missing_header_rejected(self, monkeypatch):
        monkeypatch.setattr(settings, "RAZORPAY_WEBHOOK_SECRET", "whsec_test_value")
        assert not RazorpayService.verify_webhook_signature(self.BODY, "")

    def test_parse_event_extracts_thread_id_from_notes(self):
        payload = {
            "event": "payment_link.paid",
            "payload": {
                "payment_link": {"entity": {"id": "plink_123", "amount": 950000,
                                            "notes": {"thread_id": "919812345678"}}},
                "payment": {"entity": {"id": "pay_456", "method": "card"}},
            },
        }
        parsed = RazorpayService.parse_event(payload)
        assert parsed["event"] == "payment_link.paid"
        assert parsed["link_id"] == "plink_123"
        assert parsed["payment_id"] == "pay_456"
        assert parsed["thread_id"] == "919812345678"

    def test_parse_event_tolerates_a_junk_payload(self):
        parsed = RazorpayService.parse_event({})
        assert parsed["event"] == ""
        assert parsed["link_id"] is None
        assert parsed["thread_id"] is None

    def test_to_paise_rounds_rather_than_truncates(self):
        assert RazorpayService.to_paise(15119.99) == 1511999
        assert RazorpayService.to_paise(4600.0) == 460000


class TestCheckoutGate:
    """No tap, no link — the gate is decided in code, never by the model."""

    def _state(self, text, human=True):
        msg = HumanMessage(content=text) if human else AIMessage(content=text)
        return {"messages": [msg]}

    def test_confirm_button_title_opens_the_gate(self):
        assert _is_confirm_checkout(self._state("Confirm & pay"))

    def test_postback_id_also_opens_the_gate(self):
        assert _is_confirm_checkout(self._state("CONFIRM_CHECKOUT"))

    @pytest.mark.parametrize("text", [
        "Explore more", "yes", "ok go ahead", "how much is it?", "I want to pay",
        "send me the payment link", "confirm my order",
    ])
    def test_ordinary_conversation_does_not_open_the_gate(self, text):
        assert not _is_confirm_checkout(self._state(text))

    def test_an_agent_message_can_never_open_the_gate(self):
        # The quote message itself contains the button label — it must not self-trigger.
        assert not _is_confirm_checkout(self._state("Tap Confirm & pay for a secure link", human=False))

    def test_empty_state_does_not_open_the_gate(self):
        assert not _is_confirm_checkout({"messages": []})
        assert not _is_confirm_checkout({})

    def test_connect_now_escape_hatch_still_works(self):
        # The critical-escalation hatch must remain reachable in autonomy mode.
        assert _is_connect_now(self._state("Connect me now"))
        assert _is_connect_now(self._state("CONNECT_NOW"))
        assert not _is_connect_now(self._state("Confirm & pay"))


class TestGatesAcceptEveryInboundMessageShape:
    """
    Regression: the gates must not depend on the inbound message already being a
    HumanMessage. The worker seeds a turn as ("user", text) — if a gate type-checks for
    HumanMessage and the reducer hasn't coerced it, the gate returns False forever and the
    failure looks exactly like a customer who never tapped the button. Every shape that can
    reach state is asserted here, for the checkout gate and the human escape hatch alike.
    """

    SHAPES = [
        pytest.param(lambda t: ("user", t), id="tuple"),
        pytest.param(lambda t: HumanMessage(content=t), id="human-message"),
        pytest.param(lambda t: {"role": "user", "content": t}, id="dict"),
        pytest.param(lambda t: {"role": "user", "content": [{"type": "text", "text": t}]}, id="content-blocks"),
    ]

    @pytest.mark.parametrize("shape", SHAPES)
    def test_confirm_gate_opens_for_every_shape(self, shape):
        assert _is_confirm_checkout({"messages": [shape("Confirm & pay [CONFIRM_CHECKOUT]")]})

    @pytest.mark.parametrize("shape", SHAPES)
    def test_escape_hatch_opens_for_every_shape(self, shape):
        assert _is_connect_now({"messages": [shape("Connect me now [CONNECT_NOW]")]})

    @pytest.mark.parametrize("shape", SHAPES)
    def test_ordinary_prose_still_closed_for_every_shape(self, shape):
        assert not _is_confirm_checkout({"messages": [shape("what would 4 of those cost?")]})

    @pytest.mark.parametrize("shape", SHAPES)
    def test_walkthrough_gates_open_for_every_shape(self, shape):
        from src.graph.nodes.triage import _is_swap_upgrade, _is_consult_next, _is_quote_now
        assert _is_swap_upgrade({"messages": [shape("Switch to Premium [SWAP_UPGRADE]")]})
        assert _is_consult_next({"messages": [shape("Keep the Base [CONSULT_NEXT]")]})
        assert _is_quote_now({"messages": [shape("Yes, show the price [QUOTE_NOW]")]})

    def test_walkthrough_gates_stay_shut_on_ordinary_prose(self):
        from src.graph.nodes.triage import _is_consult_next, _is_quote_now
        # These two are postback-id only on purpose: their labels are ordinary phrases, and a
        # customer typing "just this for now" mid-conversation must not silently advance a
        # walkthrough beat they cannot see.
        assert not _is_consult_next({"messages": [("user", "just this for now")]})
        assert not _is_quote_now({"messages": [("user", "yes")]})

    def test_the_reducer_coerces_the_workers_seed_into_a_message(self):
        # The other half of the guarantee: what processing.py puts in becomes a real message,
        # so history handed to the LLM is well-formed too.
        from langgraph.graph.message import add_messages
        merged = add_messages([], [("user", "Confirm & pay")])
        assert isinstance(merged[-1], HumanMessage)
        assert _is_confirm_checkout({"messages": merged})

    def test_an_assistant_tuple_never_opens_the_gate(self):
        assert not _is_confirm_checkout({"messages": [("assistant", "tap Confirm & pay")]})
        assert not _is_confirm_checkout({"messages": [{"role": "assistant", "content": "Confirm & pay"}]})

    def test_non_text_content_is_ignored_not_crashed_on(self):
        assert not _is_confirm_checkout({"messages": [("user", None)]})
        assert not _is_confirm_checkout({"messages": [object()]})


class TestQuoteMessage:
    def test_quote_shows_itemized_lines_and_the_code_computed_total(self):
        order = build_order(items=(("6 SW", 2),), offer="FESTIVE5")
        text = discounts.format_quote_message(order)
        assert "6 SW" in text
        assert f"{order['amount']:,.0f}" in text
        assert "Subtotal" in text
        assert "Total" in text

    def test_quote_omits_the_discount_line_when_there_is_no_discount(self):
        order = build_order(items=(("6 SW", 1),), offer="NONE")
        text = discounts.format_quote_message(order)
        # No deduction line, and the total equals the subtotal. An upsell hint may still
        # mention a percentage the customer COULD reach — that is a different claim from
        # "a discount was applied", so assert on the deduction itself.
        assert "−₹" not in text
        assert f"*Total*   ₹{order['subtotal']:,.0f}" in text

    def test_an_applied_discount_names_the_offer_it_came_from(self):
        """The deduction must not be anonymous. The offer LABEL carries the why ("Bundle offer" =
        because you bought a bundle), which is enough without a third explanatory line — the
        quote sits above the pay button and every extra line pushes the total off screen."""
        order = build_order(items=(("6 SW", 2), ("Zigbee Hub", 1)), offer="BUNDLE8")
        text = discounts.format_quote_message(order)
        assert "−₹" in text
        assert discounts.OFFERS["BUNDLE8"]["label"] in text
        assert "8% off" in text

    def test_next_tier_gap_is_stated_when_one_is_genuinely_close(self):
        # Two distinct products: one more would reach the 3-product tier.
        order = build_order(items=(("6 SW", 2), ("Zigbee Hub", 1)), offer="BUNDLE8")
        text = discounts.format_quote_message(order)
        assert "One more product" in text

    def test_the_top_tier_gets_no_upsell_nudge(self):
        # Two products past the project-bracket subtotal: nothing better to reach for.
        order = build_order(items=(("6 SW", 20), ("Zigbee Hub", 10)), offer="PROJECT12")
        assert order["applied_offer"] == "PROJECT12", order["audit_notes"]
        assert "qualifies for" not in discounts.format_quote_message(order)


class TestThePerUnitFigureReconcilesWithTheLine:
    """
    line_total = (base_price + installation_fee) * qty — the fitting fee is per unit and ADDITIONAL,
    not a component of the unit price. The quote used to print "₹3,500 each, incl. ₹500 fitting"
    against a ₹4,000 line, which said the ₹500 was already inside the ₹3,500 and left the ₹4,000
    unexplainable. On the one message where every figure has to be defensible, `each × qty` must
    equal the printed line.
    """

    def test_the_each_figure_times_qty_equals_the_line_total(self):
        cam = CATALOGUE["Indoor Smart Camera"]
        all_in = cam["base_price"] + cam["installation_fee"]
        for qty in (2, 7):
            order = build_order(items=(("Indoor Smart Camera", qty),), offer="NONE")
            assert order["line_items"][0]["line_total"] == all_in * qty
            assert f"₹{all_in:,.0f} each" in discounts.format_quote_message(order)

    def test_at_a_quantity_of_one_the_same_figure_is_not_printed_three_times(self):
        # "₹4,000 each" against a ₹4,000 line and a ₹4,000 Total is the same number three times, and
        # the word "each" means nothing when there is one of them. What the customer still can't see
        # anywhere else is the split, so that is all the detail line carries.
        text = discounts.format_quote_message(
            build_order(items=(("Indoor Smart Camera", 1),), offer="NONE")
        )
        assert "_(₹3,500 + ₹500 fitting)_" in text
        assert "each" not in text
        assert text.count("₹4,000") == 2  # the line and the Total

    def test_the_split_is_shown_so_the_fitting_fee_is_still_visible(self):
        text = discounts.format_quote_message(build_order(items=(("Indoor Smart Camera", 2),), offer="NONE"))
        assert "₹4,000 each (₹3,500 + ₹500 fitting)" in text
        # The old wording claimed the fee was already inside the unit price.
        assert "incl. ₹500 fitting" not in text

    def test_a_product_with_no_fitting_fee_prints_a_bare_each_figure(self):
        catalogue = {"Widget": {"product_name": "Widget", "base_price": 1000.0, "installation_fee": 0.0}}
        line_items, _u, _n = discounts.price_line_items([{"sku": "Widget", "qty": 3}], catalogue)
        text = discounts.format_quote_message(discounts.apply_offer(line_items, "NONE"))
        assert "₹1,000 each" in text
        assert "fitting" not in text


class TestASettledOrderIsNeverQuotedAgain:
    """
    A quote carries a pay button, so building one for something already bought asks for the money
    twice. Observed live: after a completed payment the customer answered the name/city question and
    received the whole quote back, pay CTA and buttons included — because the re-quote path reset
    `payment_link_sent` to False and the mint gate needs nothing more than a pending order and a tap.

    Layer one is state: the paid webhook clears `pending_order`, which removes the mint path outright.
    This is layer two — the sales node refusing to rebuild an order the customer already owns.
    """

    def _items(self, *pairs):
        from src.core.schemas import CheckoutItem
        return [CheckoutItem(sku=sku, qty=qty) for sku, qty in pairs]

    def test_an_exact_restatement_of_the_paid_order_is_suppressed(self):
        from src.graph.nodes.sales import _reproposes_paid_order
        assert _reproposes_paid_order(self._items(("Indoor Smart Camera", 1)), {"Indoor Smart Camera": 1})

    def test_case_and_whitespace_do_not_let_it_through(self):
        from src.graph.nodes.sales import _reproposes_paid_order
        assert _reproposes_paid_order(self._items(("  indoor smart camera ", 1)), {"Indoor Smart Camera": 1})

    def test_a_product_they_do_not_own_still_gets_quoted(self):
        """Repeat and follow-on purchases are the point of the channel, so this guard must not become
        a rule that the customer may only ever buy once."""
        from src.graph.nodes.sales import _reproposes_paid_order
        assert not _reproposes_paid_order(self._items(("Zigbee Hub", 1)), {"Indoor Smart Camera": 1})

    def test_more_of_something_they_own_still_gets_quoted(self):
        from src.graph.nodes.sales import _reproposes_paid_order
        assert not _reproposes_paid_order(self._items(("Indoor Smart Camera", 2)), {"Indoor Smart Camera": 1})

    def test_a_mixed_order_containing_something_new_still_gets_quoted(self):
        from src.graph.nodes.sales import _reproposes_paid_order
        items = self._items(("Indoor Smart Camera", 1), ("Zigbee Hub", 1))
        assert not _reproposes_paid_order(items, {"Indoor Smart Camera": 1})

    def test_nothing_paid_yet_is_a_no_op(self):
        from src.graph.nodes.sales import _reproposes_paid_order
        assert not _reproposes_paid_order(self._items(("Indoor Smart Camera", 1)), {})

    def test_no_proposal_is_a_no_op(self):
        from src.graph.nodes.sales import _reproposes_paid_order
        assert not _reproposes_paid_order(None, {"Indoor Smart Camera": 1})
        assert not _reproposes_paid_order([], {"Indoor Smart Camera": 1})


class TestALeadInMayOnlyTalkAboutTheBeatItPrecedes:
    """
    The model's own words arrive above a code-built message, and in the live transcript they undid it
    three separate ways at once: "Good choice!" congratulated the customer for spending before a
    price existed, a Video Door Phone was pitched in prose a full turn BEFORE its button was ever
    sent, and a question sat above a screen of buttons so nobody could tell whether to answer or tap.
    The prompt forbids all three; beat_lead_in is what makes that true when the model forgets.
    """

    ORDER = {
        "line_items": [{"sku": "Smart Door Lock Base", "qty": 1}],
        "suggested_complement": {"sku": "Video Door Phone", "display_name": "Video Door Phone"},
    }

    def test_a_praise_opener_is_stripped_and_what_follows_is_re_capitalised(self):
        from src.graph.nodes.sales import beat_lead_in
        out = beat_lead_in("Good choice! that lock covers the front door nicely.", "upsell", self.ORDER)
        assert out == "That lock covers the front door nicely."

    def test_a_sentence_naming_a_product_this_beat_is_not_about_is_dropped(self):
        # The defect this exists for: the door phone was pitched a turn before its button, so the
        # customer was asked about something they had no way to say yes to.
        from src.graph.nodes.sales import beat_lead_in
        out = beat_lead_in(
            "That lock is a sound fit for a front door. "
            "A Video Door Phone would let you see who is outside before opening.",
            "upsell",
            self.ORDER,
        )
        assert "Video Door Phone" not in out
        assert out == "That lock is a sound fit for a front door."

    def test_a_catalogue_product_outside_this_order_is_dropped_too(self):
        from src.graph.nodes.sales import beat_lead_in
        out = beat_lead_in(
            "The lock handles the door itself. An Indoor Smart Camera would cover the hallway.",
            "crosssell",
            self.ORDER,
            ["Indoor Smart Camera", "Smart Door Lock Base"],
        )
        assert "Indoor Smart Camera" not in out
        assert out == "The lock handles the door itself."

    def test_a_trailing_question_goes_and_the_quote_beat_says_what_follows(self):
        from src.graph.nodes.sales import beat_lead_in
        out = beat_lead_in(
            "Understood - just the one lock it is. "
            "To help with this, could you share your name and city?",
            "quote",
            self.ORDER,
        )
        assert "?" not in out
        assert "just the one lock it is" in out
        assert out.endswith("Here's what it comes to:")
        # The wording it replaces promised a breakdown above every beat, including a step-up card
        # with no breakdown anywhere near it.
        assert "full breakdown" not in out

    def test_only_the_quote_beat_gets_a_stitched_hand_off(self):
        from src.graph.nodes.sales import beat_lead_in
        text = "Understood - just the one lock it is. Anything else on it?"
        assert "comes to" not in beat_lead_in(text, "upsell", self.ORDER)
        assert "comes to" not in beat_lead_in(text, "crosssell", self.ORDER)
        assert beat_lead_in(text, "quote", self.ORDER).endswith("Here's what it comes to:")

    def test_an_untouched_statement_is_left_exactly_as_written(self):
        # Nothing was removed, so it still reads as an introduction and needs no stitching.
        from src.graph.nodes.sales import beat_lead_in
        text = "Here's everything for your setup so far:"
        assert beat_lead_in(text, "quote", self.ORDER) == text

    def test_nothing_substantive_left_means_no_bubble_at_all(self):
        from src.graph.nodes.sales import beat_lead_in
        assert beat_lead_in("Sure!", "crosssell", self.ORDER) == ""
        assert beat_lead_in("Good choice!", "upsell", self.ORDER) == ""
        assert beat_lead_in("Could you share your name and city?", "quote", self.ORDER) == ""

    def test_it_never_raises_on_junk(self):
        from src.graph.nodes.sales import beat_lead_in
        for junk in ("", None, "?", "???"):
            assert isinstance(beat_lead_in(junk, "quote", self.ORDER), str)


class TestTheCustomerIsToldWhichDiscountIsAvailable:
    """
    A list-price quote leaves two questions hanging: "is there a discount I'm not getting?" and
    "how do I get a better one?" Both are answered in code, with figures that match what the
    button will actually do — a preview that disagreed with the outcome would be worse than none.
    """

    def test_the_preview_matches_what_applying_it_produces(self):
        order = build_order(items=(("6 SW", 2), ("Zigbee Hub", 1)), offer="NONE")
        preview = discounts.available_offer_preview(order["line_items"])
        assert preview is not None
        applied = discounts.apply_offer(order["line_items"], preview["offer_id"])
        assert preview["saving"] == applied["discount_amount"]
        assert preview["new_total"] == applied["amount"]
        assert preview["pct"] == applied["discount_pct"]

    def test_the_button_label_states_the_real_percentage(self):
        from src.graph.nodes.sales import _quote_options
        order = build_order(items=(("6 SW", 1),), offer="NONE")
        preview = discounts.available_offer_preview(order["line_items"])
        labels = [o["label"] for o in _quote_options(order)]
        assert f"Apply {preview['pct']:.0f}% off" in labels

    def test_the_quote_names_the_offer_and_the_saving_but_never_a_second_total(self):
        order = build_order(items=(("6 SW", 1),), offer="NONE")
        text = discounts.format_quote_message(order)
        preview = discounts.available_offer_preview(order["line_items"])
        assert "You qualify for" in text
        assert f"{preview['saving']:,.0f}" in text
        assert preview["label"].lower() in text.lower()
        # Deliberately NOT the discounted total. Printing a better total under the *Total* line
        # gives one quote two totals, and the customer has to work out which one they would pay.
        # The saving is the decision — they tap Apply and the single total changes.
        assert f"{preview['new_total']:,.0f}" not in text

    def test_no_apply_button_once_the_offer_is_already_on(self):
        from src.graph.nodes.sales import _quote_options
        order = build_order(items=(("6 SW", 2), ("Zigbee Hub", 1)), offer="BUNDLE8")
        labels = [o["label"] for o in _quote_options(order)]
        assert not any(l.startswith("Apply") for l in labels)
        assert "You qualify for" not in discounts.format_quote_message(order)

    def test_the_route_to_a_bigger_discount_is_stated(self):
        order = build_order(items=(("6 SW", 2), ("Zigbee Hub", 1)), offer="BUNDLE8")
        assert "Want a bigger discount?" in discounts.format_quote_message(order)

    def test_no_preview_when_nothing_is_eligible(self):
        assert discounts.available_offer_preview([]) is None

    def test_every_quote_stays_within_three_buttons(self):
        from src.graph.nodes.sales import _quote_options
        for items, offer in [
            ((("6 SW", 1),), "NONE"),
            ((("6 SW", 2), ("Zigbee Hub", 1)), "NONE"),
            ((("6 SW", 2), ("Zigbee Hub", 1)), "BUNDLE8"),
        ]:
            assert len(_quote_options(build_order(items=items, offer=offer))) <= 3


class TestTheUpsellIsNamedValidatedAndRewarded:
    """
    The agent picks WHICH product to suggest; code decides whether that suggestion may reach the
    customer, and code writes every figure attached to it. These tests pin both halves — the
    validator's refusals, and the fact that the nudge sits where it will actually be read.
    """

    def _validate(self, sku, reason, items=(("6 SW", 1),), trusted=None):
        from src.graph.nodes.sales import _validate_complement
        order = build_order(items=items, offer="NONE")
        names = trusted if trusted is not None else {n: n for n in CATALOGUE}
        return _validate_complement(sku, reason, order["line_items"], names)

    def test_a_suggestion_the_catalogue_cannot_price_is_dropped(self):
        # The failure mode this prevents: a button that resolves to nothing when tapped.
        assert self._validate("SmartVault X9", "so it locks itself") == {}

    def test_a_product_already_in_the_order_is_dropped(self):
        assert self._validate("6 SW", "more switches") == {}

    def test_a_suggestion_with_no_reachable_tier_is_dropped(self):
        # Three distinct products already sit on the top tier: there is no reward left to claim,
        # and this block claims one.
        assert self._validate(
            "Indoor Smart Camera", "to see the hallway",
            items=(("6 SW", 1), ("Zigbee Hub", 1), ("Indoor Smart Camera", 1)),
        ) == {}

    @pytest.mark.parametrize("reason", [
        "saves you 8% on the order",
        "only ₹3,500 more",
        "it costs about 3500",
        "adds 10 percent off",
    ])
    def test_a_reason_carrying_a_figure_loses_the_reason_not_the_button(self, reason):
        out = self._validate("Zigbee Hub", reason)
        assert out["reason"] == ""
        assert out["button_label"]

    def test_a_valid_suggestion_keeps_its_reason_and_gets_a_button(self):
        out = self._validate("Zigbee Hub", "so everything talks to each other")
        assert out["sku"] == "Zigbee Hub"
        assert out["reason"] == "so everything talks to each other"
        assert out["button_label"].startswith("Add ")

    def test_the_button_label_fits_and_never_ends_mid_word(self):
        out = self._validate("Indoor Smart Camera", "to keep an eye on the hallway")
        assert len(out["button_label"]) <= 20
        if out["button_label"].endswith("…"):
            assert not out["button_label"][:-1].rstrip().endswith("came")

    def test_the_nudge_is_read_before_the_total_not_after_it(self):
        order = build_order(items=(("6 SW", 1),), offer="NONE")
        order["suggested_complement"] = {
            "sku": "Zigbee Hub", "display_name": "Zigbee Hub",
            "reason": "so everything talks to each other", "button_label": "Add Zigbee Hub",
        }
        text = discounts.format_quote_message(order)
        assert text.index("*Add the Zigbee Hub?*") < text.index("*Total*")

    def test_the_nudge_names_the_product_and_the_reward_and_replaces_the_vague_ask(self):
        order = build_order(items=(("6 SW", 1),), offer="NONE")
        order["suggested_complement"] = {
            "sku": "Zigbee Hub", "display_name": "Zigbee Hub",
            "reason": "so everything talks to each other", "button_label": "Add Zigbee Hub",
        }
        text = discounts.format_quote_message(order)
        hint = discounts.next_offer_hint(order["line_items"])
        assert "Zigbee Hub" in text
        assert f"{hint['pct']:.0f}% off" in text
        # The generic fallback asks the customer to pick from a catalogue they cannot see; it is
        # what happens when there was nothing specific to name, not the default.
        assert "tap *Explore more*" not in text

    def test_the_fallback_nudge_points_at_a_button_not_at_homework(self):
        order = build_order(items=(("6 SW", 1),), offer="NONE")
        text = discounts.format_quote_message(order)
        # With nothing validated to name, the quote still moves — but it hands the customer a
        # button to tap rather than asking them to name a product from a catalogue they have
        # never seen. "Tell me what else you're weighing up" was that homework.
        assert "tap *Explore more*" in text
        assert "weighing up" not in text

    def test_apply_and_add_never_compete_for_the_same_tap(self):
        from src.graph.nodes.sales import _quote_options
        comp = {"sku": "Zigbee Hub", "display_name": "Zigbee Hub", "reason": "",
                "button_label": "Add Zigbee Hub"}

        at_list_price = build_order(items=(("6 SW", 1),), offer="NONE")
        at_list_price["suggested_complement"] = comp
        labels = [o["label"] for o in _quote_options(at_list_price)]
        assert any(l.startswith("Apply") for l in labels)
        assert "Add Zigbee Hub" not in labels

        discounted = build_order(items=(("6 SW", 1),), offer="FESTIVE5")
        discounted["suggested_complement"] = comp
        labels = [o["label"] for o in _quote_options(discounted)]
        assert "Add Zigbee Hub" in labels
        assert not any(l.startswith("Apply") for l in labels)
        assert len(labels) <= 3

    def test_the_quote_buttons_are_exactly_the_three_expected(self):
        from src.graph.nodes.sales import _quote_options
        order = build_order(items=(("6 SW", 1),), offer="NONE")
        preview = discounts.available_offer_preview(order["line_items"])
        assert [o["label"] for o in _quote_options(order)] == [
            f"Apply {preview['pct']:.0f}% off", "Confirm & pay", "Explore more",
        ]


class TestAStepUpIsOnlyEverAVerifiedPair:
    """
    The upsell's soundness rests on `discounts.UPGRADES` being a closed registry rather than on the
    model's judgement or on a price comparison. These tests are mostly about what must NOT happen:
    every rejected pair here costs MORE than what the customer picked, so a "is it dearer" test
    waves all of them through, and each one is a different way of selling the wrong part.
    """

    UPGRADE_CATALOGUE = {
        "Smart Door Lock Base": {"product_name": "Smart Door Lock Base", "base_price": 18000.0, "installation_fee": 1500.0},
        "Smart Door Lock Premium": {"product_name": "Smart Door Lock Premium", "base_price": 28000.0, "installation_fee": 1500.0},
        "Indoor Smart Camera": {"product_name": "Indoor Smart Camera", "base_price": 3500.0, "installation_fee": 500.0},
        "Smart Flood Light Camera": {"product_name": "Smart Flood Light Camera", "base_price": 6500.0, "installation_fee": 800.0},
        "Energy Meter Single Phase": {"product_name": "Energy Meter Single Phase", "base_price": 3500.0, "installation_fee": 400.0},
        "Energy Meter 3 Phase": {"product_name": "Energy Meter 3 Phase", "base_price": 5500.0, "installation_fee": 600.0},
        "4 SW": {"product_name": "4 SW", "base_price": 3200.0, "installation_fee": 350.0},
        "6 SW": {"product_name": "6 SW", "base_price": 4200.0, "installation_fee": 400.0},
        "6 SW FAN": {"product_name": "6 SW FAN", "base_price": 4800.0, "installation_fee": 400.0},
        "PIR Motion Sensor": {"product_name": "PIR Motion Sensor", "base_price": 1800.0, "installation_fee": 200.0},
        "Microwave Sensor": {"product_name": "Microwave Sensor", "base_price": 2200.0, "installation_fee": 200.0},
        "Biometric Access Control": {"product_name": "Biometric Access Control", "base_price": 22000.0, "installation_fee": 1500.0},
        "Touch Screen Control Panel 7 inch": {"product_name": "Touch Screen Control Panel 7 inch", "base_price": 15000.0, "installation_fee": 1000.0},
        "Touch Screen Control Panel 10 inch": {"product_name": "Touch Screen Control Panel 10 inch", "base_price": 22000.0, "installation_fee": 1200.0},
    }

    # The brief asks for two or three benefits separated by semicolons, because the card leads on the
    # first and lists the rest as bullets. A step-up with nothing usable here is dropped outright.
    GAINS = (
        "so you can let a guest in from your phone while you're away; "
        "a log of who came in and when; "
        "one-time codes for guests instead of a spare key"
    )

    def _validate(self, to_sku, replaces, reason=GAINS, qty=1, prior=""):
        from src.graph.nodes.sales import _validate_upgrade
        raw = [{"sku": replaces, "qty": qty}]
        line_items, _u, _n = discounts.price_line_items(raw, self.UPGRADE_CATALOGUE)
        trusted = {n: SimpleNamespace(**spec) for n, spec in self.UPGRADE_CATALOGUE.items()}
        return _validate_upgrade(to_sku, replaces, reason, line_items, trusted, prior)

    # ── The two pairs that are real ──────────────────────────────────────────────────────────────

    def test_the_lock_pair_is_accepted(self):
        out = self._validate("Smart Door Lock Premium", "Smart Door Lock Base")
        assert out["sku"] == "Smart Door Lock Premium"
        assert out["replaces_sku"] == "Smart Door Lock Base"
        # ₹29,500 all-in against ₹19,500 — the delta is code's, computed from both catalogue rows.
        assert out["unit_delta"] == 10000.0
        assert out["line_delta"] == 10000.0

    def test_every_registry_pair_prices_and_steps_upward(self):
        # Guards the registry itself: a pair naming a product the catalogue doesn't sell, or one
        # that isn't actually dearer, is a registry bug and should fail here rather than silently
        # never fire in production.
        for frm, spec in discounts.UPGRADES.items():
            assert frm in self.UPGRADE_CATALOGUE, f"{frm} missing from the test catalogue"
            to = spec["to"]
            assert to in self.UPGRADE_CATALOGUE, f"{to} missing from the test catalogue"
            a = self.UPGRADE_CATALOGUE[frm]
            b = self.UPGRADE_CATALOGUE[to]
            assert (b["base_price"] + b["installation_fee"]) > (a["base_price"] + a["installation_fee"])
            assert spec["gains"] and not re.search(r"[\d₹%]", spec["gains"])

    # ── The pairs that must never be proposed ───────────────────────────────────────────────────

    @pytest.mark.parametrize("to_sku,replaces,why", [
        ("Energy Meter 3 Phase", "Energy Meter Single Phase", "phase is set by the building's supply"),
        ("Smart Flood Light Camera", "Indoor Smart Camera", "indoor and outdoor are different jobs"),
        ("6 SW", "4 SW", "gang count is fitment, not grade"),
        ("6 SW FAN", "6 SW", "a fan variant depends on what's on that circuit"),
        ("Microwave Sensor", "PIR Motion Sensor", "different sensing technology"),
        ("Biometric Access Control", "Smart Door Lock Base", "a different product class"),
    ])
    def test_an_unverified_pair_is_refused_however_much_dearer_it_is(self, to_sku, replaces, why):
        a = self.UPGRADE_CATALOGUE[replaces]
        b = self.UPGRADE_CATALOGUE[to_sku]
        # The premise of the test: a price check alone would accept every one of these.
        assert (b["base_price"] + b["installation_fee"]) > (a["base_price"] + a["installation_fee"])
        assert self._validate(to_sku, replaces) == {}, why

    def test_a_downgrade_is_refused(self):
        assert self._validate("Smart Door Lock Base", "Smart Door Lock Premium") == {}

    # ── The structural checks ───────────────────────────────────────────────────────────────────

    def test_a_pair_whose_left_side_is_not_in_the_order_is_refused(self):
        # The model naming a real pair while the order contains neither product.
        from src.graph.nodes.sales import _validate_upgrade
        raw = [{"sku": "6 SW", "qty": 2}]
        line_items, _u, _n = discounts.price_line_items(raw, self.UPGRADE_CATALOGUE)
        trusted = {n: SimpleNamespace(**spec) for n, spec in self.UPGRADE_CATALOGUE.items()}
        out = _validate_upgrade(
            "Smart Door Lock Premium", "Smart Door Lock Base", "so you can", line_items, trusted
        )
        assert out == {}

    def test_an_upgrade_already_in_the_order_is_refused(self):
        from src.graph.nodes.sales import _validate_upgrade
        raw = [{"sku": "Smart Door Lock Base", "qty": 1}, {"sku": "Smart Door Lock Premium", "qty": 1}]
        line_items, _u, _n = discounts.price_line_items(raw, self.UPGRADE_CATALOGUE)
        trusted = {n: SimpleNamespace(**spec) for n, spec in self.UPGRADE_CATALOGUE.items()}
        out = _validate_upgrade(
            "Smart Door Lock Premium", "Smart Door Lock Base", "so you can", line_items, trusted
        )
        assert out == {}

    def test_a_missing_replaces_field_is_refused_rather_than_guessed(self):
        assert self._validate("Smart Door Lock Premium", "") == {}

    @pytest.mark.parametrize("reason", [
        "only ₹10,000 more", "just 10000 extra", "saves 8% overall",
    ])
    def test_a_reason_carrying_a_figure_drops_the_whole_step_up(self, reason):
        # The figure has to go — no amount may originate in the model — and what is left is a price
        # difference with nothing behind it, which is a bill with no reason attached. So the card goes
        # too, rather than being shown weak.
        assert self._validate("Smart Door Lock Premium", "Smart Door Lock Base", reason=reason) == {}

    @pytest.mark.parametrize("reason", [
        "so you can set up scenes for the whole house",
        "so you can cut the standby load on every circuit",
        "retrofit it behind the existing plate",
    ])
    def test_a_reason_in_trade_words_drops_the_whole_step_up(self, reason):
        # A benefit the customer has to decode is not a benefit. The same closed list guards the
        # pairing card's reason and the explore hook, so the rule cannot hold on one card and lapse
        # on another.
        assert self._validate("Smart Door Lock Premium", "Smart Door Lock Base", reason=reason) == {}

    @pytest.mark.parametrize("reason", ["", None, "   ", ";", ".", "•"])
    def test_no_usable_benefit_means_no_step_up_at_all(self, reason):
        assert self._validate("Smart Door Lock Premium", "Smart Door Lock Base", reason=reason) == {}

    def test_only_a_step_up_card_already_shown_is_treated_as_spent(self):
        # The narrow rule, and the reason it is narrow. What must never repeat is the DECIDED
        # either/or — the card, with its exact difference, its bullets and its Switch/Keep buttons.
        # A rendered card is recognisable by its closing sentence, so that is what we look for.
        card = "\n".join(discounts.upgrade_pitch({"suggested_upgrade": self._validate(
            "Smart Door Lock Premium", "Smart Door Lock Base",
        )}))
        assert any(m in card for m in discounts.STEP_UP_CARD_MARKERS), "fixture: card must be recognisable"
        assert self._validate("Smart Door Lock Premium", "Smart Door Lock Base", prior=card) == {}

        # Matched on the normalised name, so spacing and case cannot smuggle a repeat past it.
        spaced = card.replace("Smart Door Lock Premium", "Smart Door Lock  PREMIUM")
        assert self._validate("Smart Door Lock Premium", "Smart Door Lock Base", prior=spaced) == {}

        # The live regression this replaced a wider check to fix: discovery described BOTH models in
        # prose, the any-mention version read that as a decided either/or, and the customer was
        # never shown the difference at all. Prose can say a dearer model exists; only the card is a
        # decision, so a prose mention must leave the beat standing.
        prose = "We have the Smart Door Lock Premium as well as the base model."
        assert self._validate(
            "Smart Door Lock Premium", "Smart Door Lock Base", prior=prose,
        )["sku"] == "Smart Door Lock Premium"

        # A card marker for some OTHER product is not this product's card either.
        other = card.replace("Smart Door Lock Premium", "Touch Screen Control Panel 10 inch")
        assert self._validate(
            "Smart Door Lock Premium", "Smart Door Lock Base", prior=other,
        )["sku"] == "Smart Door Lock Premium"

        # And nothing said yet is plainly a live question.
        assert self._validate(
            "Smart Door Lock Premium", "Smart Door Lock Base",
            prior="Let's start with the front door.",
        )["sku"] == "Smart Door Lock Premium"

    async def test_the_same_history_that_drops_the_step_up_leaves_the_pairing_product_alone(
        self, monkeypatch
    ):
        # The asymmetry, end to end on one order. A step-up that was shown and passed over is a
        # DECIDED either/or, so raising it again is a repeat. A pairing product that came up in
        # conversation and was never bought is still a live, un-asked question — and dropping it
        # because the agent once said the words would silence the only cross-sell moment there is.
        from src.graph.nodes import sales

        catalogue = self.UPGRADE_CATALOGUE

        class _Engine:
            def __init__(self, session):
                pass
            async def get_product_prices_batch(self, skus):
                return {
                    s: (SimpleNamespace(**catalogue[s]) if s in catalogue else None) for s in skus
                }

        class _Session:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False

        monkeypatch.setattr(sales, "PricingEngine", _Engine)
        monkeypatch.setattr(sales, "async_session_maker", lambda: _Session())

        # A rendered step-up card for the lock, plus a passing prose mention of the camera. Only the
        # first is a decision the customer has already made.
        prior = "\n".join(discounts.upgrade_pitch({"suggested_upgrade": self._validate(
            "Smart Door Lock Premium", "Smart Door Lock Base",
        )})) + "\nWe also do an Indoor Smart Camera for the hallway."
        order = await sales._build_pending_order(
            [SimpleNamespace(sku="Smart Door Lock Base", qty=1)],
            "NONE",
            suggested_complement="Indoor Smart Camera",
            complement_reason="so you can look in on the hallway; and see who came to the door",
            suggested_upgrade="Smart Door Lock Premium",
            upgrade_replaces="Smart Door Lock Base",
            upgrade_reason=self.GAINS,
            prior_agent_text=prior,
        )
        assert order
        assert "suggested_upgrade" not in order, "a step-up already shown must not come back"
        assert order["suggested_complement"]["sku"] == "Indoor Smart Camera"

    # ── The rendered block ──────────────────────────────────────────────────────────────────────

    def _quote_with_step_up(self, qty=1, reason=GAINS):
        order = build_order(items=(("6 SW", 1),), offer="NONE")
        order["suggested_upgrade"] = self._validate(
            "Smart Door Lock Premium", "Smart Door Lock Base", reason=reason, qty=qty
        )
        assert order["suggested_upgrade"], "fixture: the step-up should have validated"
        return discounts.format_quote_message(order)

    def test_the_step_up_leads_on_what_they_gain_and_lists_the_rest(self):
        text = self._quote_with_step_up()
        assert "*Let a guest in from your phone while you're away*" in text
        assert "The Smart Door Lock Premium _(+₹10,000)_ also gives you:" in text
        assert "• a log of who came in and when" in text
        assert "• one-time codes for guests instead of a spare key" in text
        assert text.index("_(+₹10,000)_") < text.index("*Total*")

    def test_one_benefit_renders_no_bullet_block_and_a_fourth_is_dropped_not_wrapped(self):
        one = self._quote_with_step_up(reason="so you can open it from your phone")
        assert "*Open it from your phone*" in one
        assert "That's the Smart Door Lock Premium _(+₹10,000)_." in one
        assert "•" not in one
        four = self._quote_with_step_up(
            reason="so you can open it from your phone; one; two; three; a fourth thing"
        )
        assert four.count("• ") == 3
        assert "a fourth thing" not in four

    def test_the_difference_is_a_tag_on_the_product_never_a_second_price(self):
        # Valuation is relative: the decision is about the increment, so the figure rides inside the
        # product line as a small italic tag. Bold, or a line of its own, reads as another amount
        # being asked for.
        text = self._quote_with_step_up()
        assert "_(+₹10,000)_" in text
        assert "*+₹10,000*" not in text
        assert "₹10,000 more" not in text
        assert "more than the one you picked" not in text
        for line in text.splitlines():
            assert line.strip() != "_(+₹10,000)_", "the difference must not be a line of its own"
        assert not text.startswith("_(+")

    def test_the_block_reads_as_information_and_never_labels_itself(self):
        # Buyers inherently mistrust vendors and fear being up-sold, and the method here is one
        # where the customer believes they are browsing options. A heading that announces an
        # upsell forfeits that; so does one that announces mere availability, which tells them
        # nothing they wanted to know. Leading with the gain informs instead of announcing.
        text = self._quote_with_step_up()
        assert "Step up to" not in text
        assert "Also available" not in text
        assert "grade" not in text.lower()
        assert "just ₹" not in text and "only ₹" not in text

    def test_declining_is_spelled_out_because_there_is_no_button_to_ignore(self):
        # Inside a quote the one affordance a tap would carry for free: that doing nothing keeps what
        # they chose. "Quoted" is a word one business says to another, so it is not the word used.
        text = self._quote_with_step_up()
        assert "Your Smart Door Lock Base stays as it is" in text
        assert "just say the word" in text
        assert "as quoted" not in text

    def test_the_difference_is_stated_per_unit_first_when_they_want_several(self):
        # "+₹10,000" for four locks understates the ask by ₹30,000, and the per-unit figure is the
        # one the decision turns on — so it comes first, with the line figure after it.
        text = self._quote_with_step_up(qty=4)
        assert "_(+₹10,000 each, ₹40,000 for all 4)_" in text

    def test_a_pair_of_them_reads_as_both_rather_than_all_2(self):
        text = self._quote_with_step_up(qty=2)
        assert "_(+₹10,000 each, ₹20,000 for both)_" in text
        assert "all 2" not in text

    def test_a_quote_never_carries_both_a_step_up_and_an_add_on(self):
        order = build_order(items=(("6 SW", 1),), offer="NONE")
        order["suggested_upgrade"] = self._validate(
            "Smart Door Lock Premium", "Smart Door Lock Base"
        )
        order["suggested_complement"] = {
            "sku": "Zigbee Hub", "display_name": "Zigbee Hub",
            "reason": "so you can run everything from one place", "button_label": "Add Zigbee Hub",
        }
        text = discounts.format_quote_message(order)
        assert "*Let a guest in from your phone while you're away*" in text
        assert "*Add the Zigbee Hub?*" not in text

    def test_the_injected_pair_list_carries_no_figures_and_names_the_rejected_shapes(self):
        menu = discounts.upgrade_menu_for_prompt(list(self.UPGRADE_CATALOGUE))
        assert "Smart Door Lock Base  ->  Smart Door Lock Premium" in menu
        assert not re.search(r"[₹%]|\d{3,}", menu)
        for shape in ("switch panel", "indoor camera", "energy meter"):
            assert shape in menu

    def test_a_pair_the_live_catalogue_cannot_price_is_not_offered(self):
        assert discounts.upgrade_menu_for_prompt(["6 SW", "Zigbee Hub"]) == ""


class TestTheThirdQuoteButtonIsNotADeadEnd:
    """
    "Explore more" replaced "Not yet" so that the one slot which can carry momentum carries a
    request instead of a shrug. Live, the tap produced a single sentence and the same quote again —
    functionally the button it replaced. The instruction is what makes the slot worth having, so it
    is pinned: the reply must name products, and it must not abandon a priced order in a tour.
    """

    def test_the_tap_is_instructed_to_name_specific_products(self):
        from src.logic.prompts import PRICING_AUTONOMY
        assert "TWO OR THREE specific ones" in PRICING_AUTONOMY
        assert "none of them already in the order" in PRICING_AUTONOMY

    def test_the_tap_is_instructed_to_return_to_the_order(self):
        from src.logic.prompts import PRICING_AUTONOMY
        assert "COME BACK TO THE ORDER" in PRICING_AUTONOMY
        assert "still saved" in PRICING_AUTONOMY

    def test_a_settled_order_is_not_returned_to(self):
        """There is nothing to come back to once it's paid, and re-offering it reads as a second bill."""
        from src.logic.prompts import PRICING_AUTONOMY
        assert "nothing to come back to" in PRICING_AUTONOMY

    def test_asking_permission_to_quote_is_forbidden(self):
        """The wasted turn observed live: "…would you like me to put together a quote?" after the
        customer had already named the product and asked the price."""
        from src.logic.prompts import PRICING_AUTONOMY
        assert "QUOTE IT IN THAT SAME TURN" in PRICING_AUTONOMY
        assert "shall I put a quote together?" in PRICING_AUTONOMY

    def test_a_comparison_has_to_end_in_a_recommendation(self):
        from src.logic.prompts import CONVERSATION_STYLE
        assert "AFTER A COMPARISON" in CONVERSATION_STYLE

    def test_the_reason_shape_is_pinned_in_both_places(self):
        """
        Each card leads on the first benefit and bullets the rest, so a one-clause reason makes a
        one-line message — and a reason that bolts on an unsupported claim makes a dangerous one (the
        live reason was "To know if a door or window is opened, which can be great for elderly care").
        The shape is pinned in the prompt AND on the schema field, because the model reads both.
        """
        from src.core.schemas import NodeExecutionSchema
        from src.logic.prompts import PRICING_AUTONOMY
        assert PRICING_AUTONOMY.count("SEPARATED BY SEMICOLONS") == 2
        for field in ("complement_reason", "upgrade_reason"):
            desc = NodeExecutionSchema.model_fields[field].description
            assert "SEPARATED BY SEMICOLONS" in desc, field
            assert "strongest first" in desc, field
            assert "care-monitoring" in desc, field
        # A step-up with nothing behind it is a bill with no reason attached, so the reason is not
        # optional — and the prompt says what happens if it comes back thin.
        assert "`upgrade_reason` IS REQUIRED" in PRICING_AUTONOMY
        assert "drops the step up altogether" in PRICING_AUTONOMY


class TestTheAddTapIsArithmeticNotAConversation:
    """
    The add-a-product tap re-prices in code: no LLM call, the guardrail runs again, and the new
    total needs a new authorisation. The gate that starts it must fire whatever shape the inbound
    message arrives in — a gate that silently never fires is indistinguishable from a customer who
    never tapped.
    """

    SHAPES = TestGatesAcceptEveryInboundMessageShape.SHAPES

    @pytest.mark.parametrize("shape", SHAPES)
    def test_the_add_gate_fires_for_every_shape(self, shape):
        from src.graph.nodes.triage import _is_add_complement
        assert _is_add_complement({"messages": [shape("Add Zigbee Hub [ADD_COMPLEMENT]")]})

    def test_the_stored_label_also_opens_it(self):
        from src.graph.nodes.triage import _is_add_complement
        assert _is_add_complement({
            "messages": [HumanMessage(content="Add Zigbee Hub")],
            "pending_order": {"suggested_complement": {"button_label": "Add Zigbee Hub"}},
        })

    def test_typing_about_a_different_product_is_left_to_the_agent(self):
        # "add a curtain motor too" names something specific; this path would add whatever
        # complement happens to be stored, which could be a different product entirely.
        from src.graph.nodes.triage import _is_add_complement
        assert not _is_add_complement({
            "messages": [HumanMessage(content="add a curtain motor too")],
            "pending_order": {"suggested_complement": {"button_label": "Add Zigbee Hub"}},
        })

    def test_the_agents_own_quote_can_never_open_it(self):
        from src.graph.nodes.triage import _is_add_complement
        assert not _is_add_complement({
            "messages": [AIMessage(content="Tap Add Zigbee Hub to include it")],
            "pending_order": {"suggested_complement": {"button_label": "Add Zigbee Hub"}},
        })

    def test_the_add_tap_never_opens_the_pay_gate(self):
        assert not _is_confirm_checkout({"messages": [HumanMessage(content="Add Zigbee Hub [ADD_COMPLEMENT]")]})

    async def test_adding_the_product_reaches_the_next_tier_and_re_arms_the_gate(self, monkeypatch):
        from src.graph.nodes import sales

        order = build_order(items=(("6 SW", 1),), offer="FESTIVE5")
        order["suggested_complement"] = {
            "sku": "Zigbee Hub", "display_name": "Zigbee Hub",
            "reason": "so everything talks to each other", "button_label": "Add Zigbee Hub",
        }

        class _Engine:
            def __init__(self, session):
                pass
            async def get_product_prices_batch(self, skus):
                return {s: CATALOGUE.get(s) for s in skus}

        class _Session:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False

        def _no_llm(*a, **kw):
            raise AssertionError("the add tap must not call an LLM")

        monkeypatch.setattr(sales, "PricingEngine", _Engine)
        monkeypatch.setattr(sales, "async_session_maker", lambda: _Session())
        monkeypatch.setattr(sales.LLMFactory, "get_llm", _no_llm)

        out = await sales._add_complement_to_order({
            "messages": [HumanMessage(content="Add Zigbee Hub [ADD_COMPLEMENT]")],
            "pending_order": order,
        })

        grown = out["pending_order"]
        assert [li["sku"] for li in grown["line_items"]] == ["6 SW", "Zigbee Hub"]
        assert grown["applied_offer"] == "BUNDLE8"
        assert grown["discount_pct"] > order["discount_pct"]
        # A bigger order at a better rate still has to clear the ceiling and the guardrail.
        assert grown["discount_pct"] <= settings.MAX_DISCOUNT_PCT
        ok, reasons = Guardrails.validate_payment_request(grown, CATALOGUE)
        assert ok, reasons
        # A new total is a new authorisation, and it must stay mintable.
        assert out["checkout_confirmed"] is False
        assert out["payment_link_sent"] is False
        assert out["add_complement_requested"] is False

    async def test_a_complement_that_vanished_from_the_catalogue_leaves_the_order_payable(self, monkeypatch):
        from src.graph.nodes import sales

        order = build_order(items=(("6 SW", 1),), offer="FESTIVE5")
        order["suggested_complement"] = {
            "sku": "Discontinued Thing", "display_name": "Discontinued Thing",
            "reason": "", "button_label": "Add Discontinued",
        }

        class _Engine:
            def __init__(self, session):
                pass
            async def get_product_prices_batch(self, skus):
                return {s: None for s in skus}

        class _Session:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False

        monkeypatch.setattr(sales, "PricingEngine", _Engine)
        monkeypatch.setattr(sales, "async_session_maker", lambda: _Session())

        out = await sales._add_complement_to_order({
            "messages": [HumanMessage(content="Add Discontinued [ADD_COMPLEMENT]")],
            "pending_order": order,
        })

        # The order survives untouched and is still payable; the dead button is gone rather than
        # left there to fail a second time.
        assert out["pending_order"]["line_items"] == order["line_items"]
        assert "suggested_complement" not in out["pending_order"]
        labels = [o["label"] for o in out["messages"][0].response_metadata["options"]]
        assert "Confirm & pay" in labels
        assert not any(l.startswith("Add ") for l in labels)


class TestAgentClaimsAboutCareMonitoringAreFlagged:
    """
    A live chat produced "PIR Motion Sensors … can be set up to alert you if there's no movement for
    a long period" to a customer asking about an elderly parent. Nothing in docs/catalog supports it:
    the sensors report an event as it happens and none watches for the absence of one.
    `specs_unavailable` could not catch it — that guard fires when a NAMED PRODUCT appears in no
    retrieved chunk, and here the product was real and the chunks discussed it. The invented part was
    the capability.

    The prohibition itself is in prompts.GUARDRAIL_RULES. What follows is the visibility backstop, and
    it is deliberately a log line rather than a block: a keyword list cannot tell an invented claim
    from a correct refusal built out of the same words.
    """

    @pytest.mark.parametrize("text", [
        "The PIR sensor can alert you if there's no movement for a long period.",
        "It does fall detection, so you'll know straight away.",
        "You can set a medical alert for her.",
        "It monitors vital signs while you're out.",
    ])
    def test_a_care_claim_is_detected(self, text):
        from src.graph.nodes.sales import _care_claim_in
        assert _care_claim_in(text) is not None

    @pytest.mark.parametrize("text", [
        "The camera streams to your phone and records to a card.",
        "It tells the app the moment the door opens.",
        "Got it — one indoor camera for the living room.",
        "",
    ])
    def test_ordinary_selling_text_is_not_flagged(self, text):
        from src.graph.nodes.sales import _care_claim_in
        assert _care_claim_in(text) is None

    def test_the_prohibition_is_in_the_prompt_every_selling_node_receives(self):
        """The log line is observability; this is the actual defence, so assert it exists."""
        from src.logic.prompts import GUARDRAIL_RULES
        for phrase in ("fall detection", "no movement", "CARE-MONITORING", "report EVENTS"):
            assert phrase in GUARDRAIL_RULES

    def test_the_catalogue_makes_no_inactivity_claim(self):
        """If this ever fails the prompt rule is the thing that is wrong, not the model."""
        from pathlib import Path
        catalog = Path("docs/catalog")
        if not catalog.exists():
            pytest.skip("catalog folder not present in this checkout")
        blob = " ".join(p.read_text(encoding="utf-8").lower() for p in catalog.rglob("*.md"))
        for claim in ("no movement", "inactivity", "fall detect"):
            assert claim not in blob


class TestJargonIsGlossedOnTheQuote:
    """Catalogue names are written for electricians. A customer who doesn't understand a line
    usually goes quiet rather than asking, so the plain words go next to it."""

    @pytest.mark.parametrize("sku,expected", [
        ("6 SW", "6-switch glass panel"),
        ("1 SW", "1-switch glass panel"),
        ("6 SW - DIMMER", "6-switch glass panel with dimming"),
        ("6 SW FAN", "6-switch glass panel with fan speed control"),
    ])
    def test_switch_codes_become_plain_words(self, sku, expected):
        assert discounts.plain_product_name(sku) == expected

    def test_two_way_explains_what_it_means(self):
        gloss = discounts.plain_product_name("2 Way 2 SW")
        assert "two-way" in gloss
        assert "two places" in gloss

    @pytest.mark.parametrize("sku", ["Curtain Motor", "Smart Door Lock Premium", "Video Door Phone"])
    def test_names_that_already_read_normally_get_no_gloss(self, sku):
        assert discounts.plain_product_name(sku) == ""

    def test_the_gloss_appears_on_the_rendered_quote(self):
        text = discounts.format_quote_message(build_order(items=(("6 SW", 2),), offer="NONE"))
        assert "6 SW" in text            # the traceable catalogue name is still printed
        assert "6-switch glass panel" in text

    def test_it_never_raises_on_junk(self):
        for junk in ("", None, "   ", "SW", "abc SW xyz"):
            assert isinstance(discounts.plain_product_name(junk), str)

    @pytest.mark.parametrize("sku,expected_word", [
        # "PIR" is trade jargon, "Microwave Sensor" actively misleads (people think of the oven), and
        # "Door Window Sensor" is catalogue spelling nobody says out loud. All three reached a live
        # quote raw, in a nudge line the agent had described in plain words two messages earlier.
        ("PIR Motion Sensor", "movement"),
        ("Microwave Sensor", "presence"),
        ("Door Window Sensor", "opens"),
    ])
    def test_sensor_names_are_translated(self, sku, expected_word):
        assert expected_word in discounts.plain_product_name(sku)

    def test_a_glossed_sensor_reads_plainly_in_the_nudge(self):
        catalogue = {"Door Window Sensor": {"product_name": "Door Window Sensor", "base_price": 1500.0, "installation_fee": 200.0}}
        line_items, _u, _n = discounts.price_line_items([{"sku": "Door Window Sensor", "qty": 1}], catalogue)
        text = discounts.format_quote_message(discounts.apply_offer(line_items, "NONE"))
        assert "tells the app when a door or window opens" in text


class TestOptionsRenderWithTheirExplanations:
    """
    WhatsApp reply buttons show ONLY a title — the description is dropped. Presenting "Base lock" /
    "Premium lock" as bare buttons is what left a customer choosing between two products whose
    difference was never shown, so the layout is decided by whether the descriptions carry meaning.
    """

    async def _payload(self, options, monkeypatch):
        import httpx
        from src.services.whatsapp import WhatsAppService

        captured = {}

        class _Cache:
            async def check_and_set_idempotency(self, *a, **kw):
                return True

        class _Resp:
            status_code = 200
            text = "{}"
            def json(self):
                return {}
            def raise_for_status(self):
                return None

        class _Client:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            async def post(self, url, json=None, headers=None):
                captured.update(json or {})
                return _Resp()

        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: _Client())
        await WhatsAppService(_Cache()).dispatch_message(
            thread_id="919812345678", webhook_msg_id="m1", node_name="n", msg_index=0,
            text="pick one", options=options, last_user_message_timestamp=9e9,
        )
        return captured

    async def test_two_options_with_descriptions_become_a_list_so_the_detail_shows(self, monkeypatch):
        payload = await self._payload([
            {"label": "Base lock", "description": "Fingerprint, PIN, card or key", "postback_id": "A"},
            {"label": "Premium lock", "description": "Adds app unlock from anywhere", "postback_id": "B"},
        ], monkeypatch)
        assert payload["interactive"]["type"] == "list"
        rows = payload["interactive"]["action"]["sections"][0]["rows"]
        assert rows[0]["description"] == "Fingerprint, PIN, card or key"
        assert rows[1]["description"] == "Adds app unlock from anywhere"

    async def test_plain_action_choices_stay_as_buttons(self, monkeypatch):
        payload = await self._payload([
            {"label": "Confirm & pay", "postback_id": "CONFIRM_CHECKOUT"},
            {"label": "Explore more", "postback_id": "CHECKOUT_NOT_YET"},
        ], monkeypatch)
        assert payload["interactive"]["type"] == "button"

    async def test_a_row_without_a_description_omits_the_key_entirely(self, monkeypatch):
        payload = await self._payload([
            {"label": "Base lock", "description": "Fingerprint, PIN, card or key", "postback_id": "A"},
            {"label": "Compare them", "postback_id": "B"},
        ], monkeypatch)
        rows = payload["interactive"]["action"]["sections"][0]["rows"]
        assert "description" not in rows[1]

    async def test_no_options_is_plain_text(self, monkeypatch):
        payload = await self._payload(None, monkeypatch)
        assert payload["type"] == "text"


class TestOptionLabelsFitAButton:
    """Three or fewer options render as WhatsApp reply buttons, which cut at 20 characters. The
    model can't know which it will get, so 20 is the only safe bound — and the shortener must
    never leave half a word behind."""

    @pytest.mark.parametrize("raw", [
        "Save on electricity bills",
        "No, just the switches and motor",
        "Yes, add the 7-inch panel",
        "Tell me about the touch panel",
    ])
    def test_long_labels_are_shortened_not_rejected(self, raw):
        from src.core.schemas import WhatsAppOption
        opt = WhatsAppOption(label=raw, postback_id="X")
        assert len(opt.label) <= 20

    def test_a_shortened_label_never_ends_mid_word(self):
        from src.core.text import fit_label
        out = fit_label("Save on electricity bills", 20)
        assert out.endswith("…")
        assert not out[:-1].rstrip().endswith("electricit")

    def test_short_labels_are_untouched(self):
        from src.core.schemas import WhatsAppOption
        for raw in ("Confirm & pay", "Explore more", "Apply 5% off", "Add door phone"):
            assert WhatsAppOption(label=raw, postback_id="X").label == raw

    def test_the_description_carries_the_meaning(self):
        from src.core.schemas import WhatsAppOption
        opt = WhatsAppOption(
            label="Save on my bills",
            description="Cut what lights and AC waste when nobody's in the room",
            postback_id="X",
        )
        assert opt.description and len(opt.description) <= 72

    def test_the_add_button_keeps_the_tail_of_the_name_rather_than_cutting_it(self):
        # "Add Indoor Smart…" went out in slot one of a priced order — the one message where a
        # half-read label sits beside the pay button. The distinguishing noun is at the END of every
        # catalogue name, which is the same thing _upgrade_button_labels uses from the other side.
        from src.graph.nodes.sales import _add_button_label
        assert _add_button_label("Indoor Smart Camera") == "Add Smart Camera"
        assert _add_button_label("Video Door Phone") == "Add Video Door Phone"
        assert _add_button_label("Touch Screen Control Panel") == "Add Control Panel"

    def test_no_add_label_is_ever_over_length_or_ellipsised(self):
        from src.graph.nodes.sales import _add_button_label
        from src.logic import discounts
        from src.scripts.seed_pricing import SEED_PRICES
        names = [row[0] for row in SEED_PRICES]
        for name in names:
            for variant in (name, discounts.plain_product_name(name) or name):
                label = _add_button_label(variant)
                assert len(label) <= 20, (variant, label)
                assert "…" not in label, (variant, label)
                assert label.startswith("Add ") and len(label) > 4, (variant, label)


class TestOfferEligibilityHelpers:
    """best_eligible_offer / next_offer_hint must agree with what apply_offer actually grants —
    a promise the code can't keep is worse than no promise."""

    def _items(self, n):
        return [
            {"sku": f"P{i}", "qty": 1, "unit_price": 5000.0,
             "installation_fee": 0.0, "line_total": 5000.0}
            for i in range(n)
        ]

    def test_best_offer_matches_what_apply_offer_grants(self):
        for n in (1, 2, 3, 4):
            items = self._items(n)
            best = discounts.best_eligible_offer(items)
            assert best is not None, n
            order = discounts.apply_offer(items, best[0])
            # If it's reported eligible, applying it must actually produce a discount.
            assert order["applied_offer"] == best[0], (n, order["audit_notes"])
            assert order["discount_amount"] > 0

    def test_no_offer_for_an_empty_order(self):
        assert discounts.best_eligible_offer([]) is None
        assert discounts.next_offer_hint([]) is None

    def test_hint_prefers_the_cheaper_step(self):
        # 2 products, small subtotal: one more product (reachable) beats a huge spend jump.
        hint = discounts.next_offer_hint(self._items(2))
        assert hint["offer_id"] == "BUNDLE10"
        assert hint["needs_products"] == 1

    def test_hint_is_dropped_when_the_gap_would_more_than_double_the_order(self):
        # A single ₹5,000 line is nowhere near the ₹1,00,000 project bracket.
        hint = discounts.next_offer_hint(self._items(1))
        assert hint is None or hint["offer_id"] != "PROJECT12"

    def test_no_hint_once_the_best_tier_is_reached(self):
        items = self._items(3)
        items[0]["line_total"] = 200000.0  # pushes subtotal past every threshold
        assert discounts.next_offer_hint(items) is None

    def test_hint_carries_no_internal_scoring_field(self):
        hint = discounts.next_offer_hint(self._items(2))
        assert "_effort" not in hint


class TestMemoryBlock:
    def test_no_facts_yields_an_empty_block(self):
        assert format_memory_block([]) == ""
        assert format_memory_block(None or []) == ""

    def test_facts_are_rendered_with_a_do_not_recite_instruction(self):
        block = format_memory_block(["Lives in Kochi", "Has a 3BHK apartment"])
        assert "Lives in Kochi" in block
        assert "Has a 3BHK apartment" in block
        assert "Do NOT recite" in block

    def test_block_is_bounded_so_memory_cannot_flood_the_prompt(self):
        block = format_memory_block([f"fact {i}" for i in range(50)])
        assert block.count("\n- ") <= 5


class TestTheConsultativeWalkthrough:
    """
    A price is the LAST thing a customer sees, not the first.

    A live transcript had the agent name a lock AND a video door phone for front-door safety, the
    customer choose the lock, and a full quote arrive in the same breath — the door phone never
    mentioned again, with a better discount tier one already-discussed product away. Choosing a
    product is not asking what it costs, so the order is priced and HELD while three beats run in
    order: the dearer model of what they picked, then the product that pairs with it, then the offer
    to show them the price. Every beat is code-written from data validated when the order was built,
    so none of them costs an LLM call or can invent a figure.
    """

    UPGRADE = {
        "sku": "Smart Door Lock Premium", "display_name": "Smart Door Lock Premium",
        "replaces_sku": "Smart Door Lock Base", "replaces_display": "Smart Door Lock Base",
        "qty": 1, "unit_delta": 9000.0, "line_delta": 9000.0,
        "reason": "so you can let a guest in from your phone while you're away; "
                  "a log of who came in and when; one-time codes for guests",
        "button_label": "Switch to Premium", "keep_label": "Keep the Base",
    }
    COMPLEMENT = {
        "sku": "Zigbee Hub", "display_name": "Zigbee Hub",
        "reason": "so you can run everything from one place; open the door for a guest while "
                  "you're out; ask for the lights without getting up",
        "button_label": "Add Zigbee Hub",
    }

    def _order(self, upgrade=True, complement=True, hook="Want to see what's costing you most?"):
        order = build_order(items=(("6 SW", 1),), offer="NONE")
        if upgrade:
            order["suggested_upgrade"] = dict(self.UPGRADE)
        if complement:
            order["suggested_complement"] = dict(self.COMPLEMENT)
        if hook:
            order["explore_hook"] = hook
        return order

    def test_the_beats_run_upsell_then_crosssell_then_the_ask_then_the_price(self):
        from src.graph.nodes.sales import _next_beat
        order = self._order()
        assert _next_beat(order, 0) == "upsell"
        assert _next_beat(order, 1) == "crosssell"
        assert _next_beat(order, 2) == "quote_ask"
        assert _next_beat(order, 3) == "quote"

    def test_a_beat_with_nothing_verified_to_say_is_skipped_not_padded(self):
        from src.graph.nodes.sales import _next_beat
        # No listed step-up pair and no pairing that fits: the customer is asked once whether to be
        # priced, and that is the whole walkthrough. Three messages of filler is worse than one.
        bare = self._order(upgrade=False, complement=False)
        assert _next_beat(bare, 0) == "quote_ask"
        assert _next_beat(self._order(upgrade=False), 0) == "crosssell"

    def test_the_stage_only_ever_moves_forward(self):
        from src.graph.nodes.sales import _advance
        order = self._order()
        stages, stage = [], 0
        for _ in range(4):
            stage = _advance(order, stage)["consult_stage"]
            stages.append(stage)
        assert stages == [1, 2, 3, 4]

    def test_no_beat_carries_a_figure_the_model_could_have_written(self):
        from src.graph.nodes.sales import _beat_message
        order = self._order()
        # The two selling beats DO print figures — a price difference and an offer percentage — but
        # both are computed by code from the catalogue and the offer registry. What the model wrote
        # is the list of benefits, which the card splits into a heading and bullets, so each clause
        # has to survive and none of them may add anything numeric of its own.
        upsell = _beat_message(order, "upsell").content.lower()
        for clause in ("let a guest in from your phone while you're away",
                       "a log of who came in and when", "one-time codes for guests"):
            assert clause in upsell, clause
        assert not re.search(r"[₹\d%]", order["suggested_upgrade"]["reason"])
        assert not re.search(r"[₹\d%]", order["suggested_complement"]["reason"])
        # The ask beat has no figures at all: it is the message that exists BECAUSE no price has
        # been shown yet, and the hook the model wrote for it is checked for digits at build time.
        ask = _beat_message(order, "quote_ask").content
        assert not re.search(r"[₹\d%]", ask), ask

    def test_the_upsell_beat_offers_the_swap_and_the_way_past_it(self):
        from src.graph.nodes.sales import _beat_message, SWAP_UPGRADE_POSTBACK, CONSULT_NEXT_POSTBACK
        options = _beat_message(self._order(), "upsell").response_metadata["options"]
        assert [o["postback_id"] for o in options][:2] == [SWAP_UPGRADE_POSTBACK, CONSULT_NEXT_POSTBACK]
        assert options[0]["label"] == "Switch to Premium"
        assert options[1]["label"] == "Keep the Base"
        assert len(options) <= 3  # WhatsApp's reply-button ceiling
        # No description on any of them: whatsapp.py flips to a LIST layout the moment one is
        # present, which buries the first button behind a "Choose one" tap.
        assert not any(o.get("description") for o in options)

    def test_the_crosssell_beat_offers_the_add_and_the_way_past_it(self):
        from src.graph.nodes.sales import _beat_message, ADD_COMPLEMENT_POSTBACK, CONSULT_NEXT_POSTBACK
        msg = _beat_message(self._order(), "crosssell")
        options = msg.response_metadata["options"]
        assert [o["postback_id"] for o in options][:2] == [ADD_COMPLEMENT_POSTBACK, CONSULT_NEXT_POSTBACK]
        assert options[0]["label"] == "Add Zigbee Hub"
        assert "Zigbee Hub" in msg.content
        assert len(options) <= 3

    def test_the_ask_beat_pairs_the_price_button_with_a_benefit_led_way_out(self):
        from src.graph.nodes.sales import _beat_message, QUOTE_NOW_POSTBACK
        msg = _beat_message(self._order(hook="Want to save on your electricity bill?"), "quote_ask")
        options = msg.response_metadata["options"]
        assert options[0]["postback_id"] == QUOTE_NOW_POSTBACK
        # The second slot is the hook, labelled by what it is ABOUT — a benefit, not a refusal.
        # "Not yet" made the customer say no at the moment momentum matters most.
        assert options[1]["label"] == "Save on electricity"
        assert "not yet" not in " ".join(o["label"].lower() for o in options)
        assert "Want to save on your electricity bill?" in msg.content
        assert all(len(o["label"]) <= 20 for o in options)

    def test_the_ask_beat_still_works_when_the_model_gave_no_hook(self):
        from src.graph.nodes.sales import _beat_message, _HOOK_FALLBACK_LABEL
        msg = _beat_message(self._order(hook=None), "quote_ask")
        assert msg.content.strip()
        assert msg.response_metadata["options"][1]["label"] == _HOOK_FALLBACK_LABEL

    def test_an_explicit_price_request_skips_straight_to_the_quote(self):
        from src.graph.nodes.sales import _advance
        update = _advance(self._order(), 0, beat="quote")
        assert update["consult_stage"] == 4
        text = update["messages"][0].content
        assert "*Total*" in text
        # Nothing was tapped through, so the suggestion rides along in the quote body instead —
        # still one ask above the total, never two.
        assert "Smart Door Lock Premium" in text

    def test_a_walked_through_quote_does_not_re_pitch_what_the_beats_showed(self):
        from src.graph.nodes.sales import _advance
        text = _advance(self._order(), 3)["messages"][0].content
        assert "*Total*" in text
        assert "Zigbee Hub" not in text
        assert "Premium" not in text

    def test_every_walkthrough_step_leaves_the_order_needing_a_fresh_authorisation(self):
        from src.graph.nodes.sales import _advance
        for stage in (0, 1, 2, 3):
            update = _advance(self._order(), stage)
            assert update["checkout_confirmed"] is False, stage
            assert update["payment_link_sent"] is False, stage
            assert update["pending_order"]["amount"] > 0

    def test_a_changed_order_gets_a_lead_in_so_the_tap_is_visibly_acknowledged(self):
        from src.graph.nodes.sales import _advance
        update = _advance(self._order(), 1, lead_in="Added to your order.")
        assert update["messages"][0].content == "Added to your order."
        assert update["messages"][0].response_metadata["options"] is None
        assert len(update["messages"]) == 2

    def test_the_hook_label_is_chosen_from_what_the_hook_is_about(self):
        from src.graph.nodes.sales import _hook_label, _HOOK_FALLBACK_LABEL
        assert _hook_label("Curious what makes a place feel safe when nobody's home?") == "Make it safer"
        assert _hook_label("Want your curtains to open on their own each morning?") == "Curtains at a tap"
        assert _hook_label("") == _HOOK_FALLBACK_LABEL

    def test_the_swap_buttons_name_only_the_part_that_differs(self):
        from src.graph.nodes.sales import _upgrade_button_labels
        # Both names share everything but their tail, so that tail is the only thing a 20-character
        # button can spend its characters on.
        assert _upgrade_button_labels("Smart Door Lock Base", "Smart Door Lock Premium") == (
            "Switch to Premium", "Keep the Base"
        )
        take, keep = _upgrade_button_labels(
            "Touch Screen Control Panel 7 inch", "Touch Screen Control Panel 10 inch"
        )
        assert (take, keep) == ("Switch to 10 inch", "Keep the 7 inch")
        for label in (take, keep):
            assert len(label) <= 20 and not label.endswith("…")

    def test_both_suggestions_survive_on_one_order_because_each_gets_its_own_beat(self):
        # The old rule kept ONE and threw the other away — which is how a door phone already on the
        # table went unmentioned for the rest of the conversation.
        order = self._order()
        assert order["suggested_upgrade"] and order["suggested_complement"]
        assert discounts.upgrade_pitch(order) and discounts.complement_pitch(order)

    def test_a_standalone_pitch_never_claims_the_order_is_already_quoted(self):
        order = self._order()
        # Before a price exists "stays as quoted" is simply false, and neither block may read as a
        # question stapled above a total — the buttons under them are the ask.
        assert not any("as quoted" in line for line in discounts.upgrade_pitch(order))
        assert not any("*Add the" in line for line in discounts.complement_pitch(order))
        assert not any(line.rstrip().endswith("?") for line in discounts.upgrade_pitch(order))

    def test_the_step_up_beat_leads_on_the_gain_and_closes_on_the_swap(self):
        lines = discounts.upgrade_pitch(self._order())
        assert lines[0] == "*Let a guest in from your phone while you're away*"
        # Blank line under the heading and above the closing line: on a phone the card is five short
        # lines with no punctuation to break them up, and run together it reads as one block.
        assert lines[1] == ""
        assert lines[2] == "The Smart Door Lock Premium _(+₹9,000)_ also gives you:"
        assert lines[3:5] == ["• a log of who came in and when", "• one-time codes for guests"]
        assert lines[-2] == ""
        assert lines[-1] == "_You'd get this one instead of the one you picked._"
        # Not a hedge in sight: the Keep button is what makes declining free.
        text = "\n".join(lines)
        assert "either way" not in text.lower()
        assert "no commitment" not in text.lower()
        assert "Also available" not in text

    def test_the_in_quote_block_stays_tight(self):
        # Inside a quote the block is one item among several, so the spacing that helps a standalone
        # card would push the total off the first screen.
        lines = discounts._upgrade_lines(self._order(), standalone=False)
        assert "" not in lines

    def test_swapping_several_of_them_says_these_rather_than_this_one(self):
        order = self._order()
        order["suggested_upgrade"].update(qty=4, unit_delta=9000.0, line_delta=36000.0)
        lines = discounts.upgrade_pitch(order)
        assert "The Smart Door Lock Premium _(+₹9,000 each, ₹36,000 for all 4)_ also gives you:" in lines
        assert lines[-1] == "_You'd get these instead of the ones you picked._"

    def test_the_pairing_beat_says_most_people_and_puts_the_reward_last(self):
        lines = discounts.complement_pitch(self._order())
        assert lines[0] == "*Run everything from one place*"
        # Blank line under the heading, and above the reward: the benefit has to land on its own.
        assert lines[1] == ""
        assert lines[2].startswith("Most people fitting a 6-switch glass panel want that too.")
        assert "The Zigbee Hub also gives you:" in lines[2]
        assert lines[3].startswith("• open the door for a guest")
        assert lines[-2] == ""
        assert lines[-1] == "Adding it takes this order to 8% off."
        text = "\n".join(lines)
        assert "Goes well with it" not in text
        assert "common pairing" not in text

    @pytest.mark.parametrize(
        "flag,expected_stage",
        [("consult_next_requested", 2), ("quote_now_requested", 4)],
    )
    async def test_a_walkthrough_tap_is_answered_without_ever_building_a_model(
        self, monkeypatch, flag, expected_stage
    ):
        """
        The whole point of a code-owned walkthrough: three extra beats that cost zero LLM calls.
        A model turn here would add seconds to each step and give the model a chance to re-propose
        the order mid-checkout — the failure that made the paid-order guards necessary.
        """
        from src.core.llm_factory import LLMFactory
        from src.graph.nodes import sales as sales_node
        from src.logic.prompts import HIGH_INTENT_PROMPT

        monkeypatch.setattr(settings, "AGENT_FULL_AUTONOMY", True)

        def _explode(*a, **kw):
            raise AssertionError("a walkthrough tap must never build an LLM")

        monkeypatch.setattr(LLMFactory, "get_llm", _explode)

        state = {
            "messages": [HumanMessage(content="Just this for now [CONSULT_NEXT]")],
            "pending_order": self._order(),
            "consult_stage": 1,
            flag: True,
        }
        update = await sales_node._execute_sales_node(state, HIGH_INTENT_PROMPT)
        assert update["consult_stage"] == expected_stage
        assert update["messages"] and update["messages"][-1].response_metadata["options"]
        assert update["checkout_confirmed"] is False


class TestTheWalkthroughReadsRightForAnyProduct:
    """
    Exactly two products in the catalogue have a verified step-up, so for everything else beat one
    does not exist and the pairing card is the FIRST thing the customer sees after choosing. The
    transcript that prompted this change set was a lock, and wording written while looking at one
    lock is the easiest way to end up with a sequence that only reads correctly for locks.
    """

    CATALOGUE = {
        "Indoor Smart Camera": {"product_name": "Indoor Smart Camera", "base_price": 3500.0, "installation_fee": 500.0},
        "Video Door Phone": {"product_name": "Video Door Phone", "base_price": 12000.0, "installation_fee": 1200.0},
    }
    GAINS = (
        "so you can know who's at the gate before you walk down; "
        "a photo of whoever rang while you were out; "
        "a look outside before you open up at night"
    )

    def _camera_order(self, reason=GAINS):
        line_items, _u, _n = discounts.price_line_items(
            [{"sku": "Indoor Smart Camera", "qty": 1}], self.CATALOGUE
        )
        order = discounts.apply_offer(line_items, "NONE")
        order["suggested_complement"] = {
            "sku": "Video Door Phone", "display_name": "Video Door Phone",
            "reason": reason, "button_label": "Add door phone",
        }
        order["explore_hook"] = "Want to know who's been at your gate while you were out?"
        return order

    def test_a_product_with_no_verified_pair_opens_on_the_pairing_card(self):
        from src.graph.nodes.sales import _next_beat, _STAGE_AFTER, _advance
        order = self._camera_order()
        assert not discounts.upgrade_pitch(order)  # there is no pair to step up to
        assert _next_beat(order, 0) == "crosssell"
        # No empty beat and no menu of camera models: the skipped beat costs a message, not a stumble.
        update = _advance(order, 0)
        assert update["consult_stage"] == _STAGE_AFTER["crosssell"]
        assert "instead of the one you picked" not in update["messages"][0].content

    def test_the_card_names_the_product_in_full_with_the_right_article(self):
        lines = discounts.complement_pitch(self._camera_order())
        assert lines[0] == "*Know who's at the gate before you walk down*"
        assert lines[1] == ""
        assert lines[2] == (
            "Most people fitting an Indoor Smart Camera want that too. "
            "The Video Door Phone also gives you:"
        )
        assert lines[3] == "• a photo of whoever rang while you were out"
        assert lines[-1] == "Adding it takes this order to 8% off."

    def test_a_name_in_no_registry_pair_is_never_clipped(self):
        # family_name walks a shared prefix, and there is nothing to share here — "Video Door Phone"
        # must not become "Video Door", nor the camera "Indoor Smart".
        text = "\n".join(discounts.complement_pitch(self._camera_order()))
        assert "Video Door\n" not in text and "Video Door " not in text.replace("Video Door Phone", "")
        assert "Indoor Smart " not in text.replace("Indoor Smart Camera", "")

    def test_the_pairing_beat_is_the_same_three_parts_whatever_the_product(self):
        from src.graph.nodes.sales import _beat_message, ADD_COMPLEMENT_POSTBACK
        msg = _beat_message(self._camera_order(), "crosssell")
        assert msg.response_metadata["options"][0]["postback_id"] == ADD_COMPLEMENT_POSTBACK
        assert msg.response_metadata["options"][0]["label"] == "Add door phone"
        body = msg.content
        assert body.index("Know who's at the gate") < body.index("Most people fitting")
        assert body.index("Most people fitting") < body.index("8% off")


class TestAProductIsNamedByItsFamilyInASentence:
    """
    "Most people fitting a Smart Door Lock Base want that too" tells a customer they have been sorted
    into a tier — and nobody who asked for a door lock thinks of themselves as having picked "the
    Base". In prose the product is named the way a person would name it. The tier word survives only
    where the contrast IS the subject: the step-up card's own target, and its two buttons.
    """

    def test_a_registry_pair_collapses_to_what_the_two_share(self):
        assert discounts.family_name("Smart Door Lock Base") == "Smart Door Lock"
        assert discounts.family_name("Smart Door Lock Premium") == "Smart Door Lock"
        assert discounts.family_name("Touch Screen Control Panel 7 inch") == "Touch Screen Control Panel"
        assert discounts.family_name("Touch Screen Control Panel 10 inch") == "Touch Screen Control Panel"

    def test_a_product_in_no_pair_keeps_every_word_of_its_name(self):
        for name in ("Video Door Phone", "Indoor Smart Camera", "Zigbee Hub", "Curtain Motor"):
            assert discounts.family_name(name) == name

    def test_trade_shorthand_is_said_the_way_a_customer_would_understand_it(self):
        # "Most people fitting a 6 SW want that too" went out live and means nothing to anyone who
        # hasn't wired a switchboard. Only the shorthand family is replaced — a product with a
        # readable name keeps it, because that is what the customer says back to the team.
        assert discounts.family_name("6 SW") == "6-switch glass panel"
        assert discounts.speakable_name("6 SW FAN") == "6-switch glass panel with fan speed control"
        assert discounts.speakable_name("Zigbee Hub") == "Zigbee Hub"
        assert discounts.speakable_name("Video Door Phone") == "Video Door Phone"

    def test_it_never_raises_on_junk(self):
        for junk in ("", None, "   ", "Smart", "Smart Door Lock"):
            assert isinstance(discounts.family_name(junk), str)

    def test_the_step_up_card_still_names_its_target_in_full(self):
        # The one place the tier word belongs: the customer is choosing between two models, so
        # "the Smart Door Lock" would name both of them and answer nothing.
        order = build_order(items=(("6 SW", 1),), offer="NONE")
        order["suggested_upgrade"] = {
            "sku": "Smart Door Lock Premium", "display_name": "Smart Door Lock Premium",
            "replaces_sku": "Smart Door Lock Base", "replaces_display": "Smart Door Lock Base",
            "qty": 1, "unit_delta": 10000.0, "line_delta": 10000.0,
            "reason": "so you can open it from your phone; one-time codes for guests",
            "button_label": "Switch to Premium", "keep_label": "Keep the Base",
        }
        text = "\n".join(discounts.upgrade_pitch(order))
        assert "The Smart Door Lock Premium" in text
        # …and no sentence leans on the bare tier word to carry the meaning.
        assert "the Base" not in text and "the Premium" not in text


class TestBothSuggestionCardsSpeakInOneVoice:
    """
    The two cards arrive one after the other, so a difference in voice between them reads as two
    different senders. One helper builds the heading for both, which is also the only way the
    "no usable benefit" rule can be checked at validation time and rendered consistently later.
    """

    LONG = (
        "so you can see exactly who came to your door while you were away at work for the whole day"
    )

    def test_the_same_benefit_produces_the_same_heading_on_either_card(self):
        reason = "so you can see who's at the door before you open it; and a photo of them after"
        up = build_order(items=(("6 SW", 1),), offer="NONE")
        up["suggested_upgrade"] = {
            "sku": "Smart Door Lock Premium", "display_name": "Smart Door Lock Premium",
            "replaces_sku": "Smart Door Lock Base", "replaces_display": "Smart Door Lock Base",
            "qty": 1, "unit_delta": 10000.0, "line_delta": 10000.0, "reason": reason,
            "button_label": "Switch to Premium", "keep_label": "Keep the Base",
        }
        comp = build_order(items=(("6 SW", 1),), offer="NONE")
        comp["suggested_complement"] = {
            "sku": "Zigbee Hub", "display_name": "Zigbee Hub",
            "reason": reason, "button_label": "Add Zigbee Hub",
        }
        heading = "*See who's at the door before you open it*"
        assert discounts.upgrade_pitch(up)[0] == heading
        assert discounts.complement_pitch(comp)[0] == heading

    def test_the_briefs_own_scaffolding_is_stripped_from_the_heading(self):
        assert discounts.benefit_heading("so you can open it from your phone") == "Open it from your phone"
        assert discounts.benefit_heading("you'll never hunt for keys again") == "Never hunt for keys again"
        assert discounts.benefit_heading("• so that you can see the gate.") == "See the gate"
        assert discounts.benefit_heading("a log of who came in; and more") == "A log of who came in"

    def test_a_heading_too_long_to_bold_is_left_plain_and_whole(self):
        # A claim cut in half reads worse than an unbolded one, so it is never trimmed.
        heading = discounts.benefit_heading(self.LONG)
        assert len(heading) > 64
        rendered = discounts._heading_line(heading)
        assert rendered == heading
        assert not rendered.startswith("*")
        assert rendered.endswith("whole day")

    def test_no_heading_at_all_means_the_card_falls_back_rather_than_opens_blank(self):
        order = build_order(items=(("6 SW", 1),), offer="NONE")
        order["suggested_complement"] = {
            "sku": "Zigbee Hub", "display_name": "Zigbee Hub", "reason": "", "button_label": "Add Zigbee Hub",
        }
        lines = discounts.complement_pitch(order)
        assert lines[0] == "*Worth adding: Zigbee Hub*"
        assert lines[-1] == "Adding it takes this order to 8% off."


class TestTheAskBeatIsTheAskAndNothingElse:
    """
    Between the ask and the hook there used to be two more lines: an explainer describing the message
    that was about to arrive, and a reassurance that answered an objection nobody had made while
    putting the word "commitment" in front of a customer who hadn't thought of one. Both went. What is
    left is two questions, each with its own button — which is why two is fine here and nowhere else.
    """

    HOOK = "Want to see what the lights left on all day are costing you?"

    def _ask(self, hook=HOOK):
        from src.graph.nodes.sales import _beat_message
        order = build_order(items=(("6 SW", 1),), offer="NONE")
        if hook:
            order["explore_hook"] = hook
        return _beat_message(order, "quote_ask")

    def test_the_body_is_the_ask_then_the_hook_and_nothing_between_them(self):
        assert self._ask().content == f"*Shall I show you the price?*\n\n{self.HOOK}"

    def test_the_padding_and_the_back_office_word_are_both_gone(self):
        text = self._ask().content.lower()
        for banned in ("commitment", "either way", "break it down", "line by line", "quot"):
            assert banned not in text, banned

    def test_two_buttons_and_the_price_is_the_first_one(self):
        from src.graph.nodes.sales import QUOTE_NOW_POSTBACK, CHECKOUT_NOT_YET_OPTION
        options = self._ask().response_metadata["options"]
        assert len(options) == 2
        assert options[0] == {"label": "Yes, show the price", "postback_id": QUOTE_NOW_POSTBACK}
        # The way out of being priced, labelled by what it is about rather than as a refusal.
        assert options[1]["label"] == "Save on electricity"
        assert options[1]["postback_id"] == CHECKOUT_NOT_YET_OPTION["postback_id"]

    def test_the_second_button_answers_the_question_printed_above_it(self):
        from src.graph.nodes.sales import _hook_label, _HOOK_FALLBACK_LABEL
        for hook, label in [
            ("Ever wonder who rang the doorbell while you were out?", "Make it safer"),
            ("Want the curtains to open on their own each morning?", "Curtains at a tap"),
            ("Would you know if a tank overflowed while you were asleep?", "Stop water damage"),
            ("Want to stop getting up to switch the lights off?", "Better lighting"),
            ("Want to check on the place while you're travelling?", "Control it anywhere"),
            ("Tired of the same little chore every single night?", "Make life easier"),
        ]:
            assert _hook_label(hook) == label, hook
        # Unthemed, so it answers the question rather than guessing at a subject.
        assert _hook_label("Curious about something most people miss?") == _HOOK_FALLBACK_LABEL

    def test_no_label_on_this_message_can_ever_arrive_half_cut(self):
        # This is the message that decides whether a price gets shown, so a trimmed label here is not
        # recoverable. Every one is hand-written inside the limit rather than trimmed down to it.
        from src.core.text import fit_label
        from src.graph.nodes.sales import (
            _HOOK_LABELS, _HOOK_FALLBACK_LABEL, QUOTE_NOW_LABEL, CONSULT_NEXT_LABEL,
        )
        labels = [label for _keys, label in _HOOK_LABELS]
        labels += [_HOOK_FALLBACK_LABEL, QUOTE_NOW_LABEL, CONSULT_NEXT_LABEL]
        for label in labels:
            assert len(label) <= 20, label
            assert fit_label(label, 20) == label, label
            assert not label.endswith("…"), label

    async def test_a_hook_in_trade_words_or_carrying_a_figure_never_reaches_the_customer(
        self, monkeypatch
    ):
        # The prompt asks for plain words; this is what holds when the model reaches for the trade
        # term anyway. Failure is a downgrade to a code-written question, never a button-less beat.
        from src.graph.nodes import sales
        from src.graph.nodes.sales import _beat_message, _HOOK_FALLBACK_TEXT, _HOOK_FALLBACK_LABEL

        class _Engine:
            def __init__(self, session):
                pass
            async def get_product_prices_batch(self, skus):
                return {s: (SimpleNamespace(**CATALOGUE[s]) if s in CATALOGUE else None) for s in skus}

        class _Session:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False

        monkeypatch.setattr(sales, "PricingEngine", _Engine)
        monkeypatch.setattr(sales, "async_session_maker", lambda: _Session())

        for hook in (
            "Most homes lose more on standby power than on lighting — want to see?",
            "Want to know which scenes people set up first?",
            "Want to see how ₹500 a month disappears?",
            "Want to cut 30% off what you're spending?",
        ):
            order = await sales._build_pending_order(
                [SimpleNamespace(sku="6 SW", qty=1)], "NONE", explore_hook=hook
            )
            assert order and "explore_hook" not in order, hook
            msg = _beat_message(order, "quote_ask")
            assert _HOOK_FALLBACK_TEXT in msg.content
            assert msg.response_metadata["options"][1]["label"] == _HOOK_FALLBACK_LABEL

        kept = await sales._build_pending_order(
            [SimpleNamespace(sku="6 SW", qty=1)], "NONE", explore_hook=self.HOOK
        )
        assert kept["explore_hook"] == self.HOOK


class TestThePricedOrderCanBeChangedByTyping:
    """
    Typing an edit always worked — it is an ordinary turn that rebuilds the items and re-prices
    through the same guardrail — and nothing told the customer so. Four messages of tapping teaches a
    thumb to look for a button, so the line names the gesture rather than the possibility, and it is
    a line in the body rather than a fourth button: the quote's three buttons each do one obvious
    thing, and `Explore more` keeps the slot it has in every screenshot.
    """

    EDIT_LINE = (
        "_Need anything different — a different quantity, another product, one taken off? "
        "Just type it in the message box._"
    )

    def test_the_line_names_the_gesture_and_the_place_to_make_it(self):
        text = discounts.format_quote_message(build_order())
        assert self.EDIT_LINE in text
        assert "message box" in text

    def test_it_sits_under_the_total_and_above_the_call_to_action(self):
        # Below the pay prompt it competes with Confirm & pay; above the Total it reads as an apology
        # for a price the customer hasn't been shown yet.
        text = discounts.format_quote_message(build_order())
        assert text.index("*Total*") < text.index(self.EDIT_LINE) < text.index("_Test-mode order")

    def test_it_carries_no_examples_and_promises_no_redo(self):
        text = discounts.format_quote_message(build_order())
        assert "e.g." not in text
        assert "redo" not in text.lower()
        assert '"' not in self.EDIT_LINE and "'" not in self.EDIT_LINE

    def test_no_re_price_can_drop_it(self):
        # The customer most likely to change something is the one who just changed something, so this
        # has to survive every path back through the renderer, including the post-walkthrough quote
        # that shows no suggestions at all.
        for offer in (None, "NONE", "FESTIVE5", "BUNDLE8", "BUNDLE10"):
            for show in (True, False):
                order = build_order(items=(("6 SW", 2), ("Zigbee Hub", 1)), offer=offer)
                text = discounts.format_quote_message(order, show_suggestions=show)
                assert self.EDIT_LINE in text, (offer, show)

    def test_the_quote_still_has_three_buttons_and_explore_more_is_the_last(self):
        from src.graph.nodes.sales import _quote_options
        # Both slot-one shapes: the Apply button (an eligible offer unapplied) and the Add button.
        unapplied = build_order(items=(("6 SW", 1), ("Zigbee Hub", 1)), offer="NONE")
        applied = build_order(items=(("6 SW", 1), ("Zigbee Hub", 1)), offer="BUNDLE8")
        applied["suggested_complement"] = {
            "sku": "Indoor Smart Camera", "display_name": "Indoor Smart Camera",
            "reason": "so you can see the room while you're away",
            "button_label": "Add camera",
        }
        for order in (unapplied, applied):
            options = _quote_options(order)
            assert len(options) == 3
            assert options[-1]["label"] == "Explore more"
            assert options[-1]["postback_id"] == "CHECKOUT_NOT_YET"
            assert not any("description" in o for o in options)

    async def test_a_typed_change_re_prices_through_the_same_money_gate(self, monkeypatch):
        # "make it two" is not a tap: it comes back as a normal turn with fresh checkout_items, so the
        # promise the line makes is only kept if that path re-prices and re-validates like any other.
        from src.graph.nodes import sales

        class _Engine:
            def __init__(self, session):
                pass
            async def get_product_prices_batch(self, skus):
                return {s: (SimpleNamespace(**CATALOGUE[s]) if s in CATALOGUE else None) for s in skus}

        class _Session:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False

        monkeypatch.setattr(sales, "PricingEngine", _Engine)
        monkeypatch.setattr(sales, "async_session_maker", lambda: _Session())

        one = await sales._build_pending_order([SimpleNamespace(sku="6 SW", qty=1)], "NONE")
        two = await sales._build_pending_order([SimpleNamespace(sku="6 SW", qty=2)], "NONE")
        assert two["amount"] == one["amount"] * 2
        ok, reasons = Guardrails.validate_payment_request(two, CATALOGUE)
        assert ok, reasons
        assert self.EDIT_LINE in discounts.format_quote_message(two)

        # And the re-priced order is a fresh authorisation, not the old one carried forward.
        updates = sales._advance(two, sales._STAGE_AFTER["quote_ask"])
        assert updates["checkout_confirmed"] is False
        assert updates["payment_link_sent"] is False


class TestABeatBelongsToTheOrderItIsWalking:
    """
    `consult_stage` alone could not say WHICH sale it had walked. A customer who came back a week
    later and picked something else landed on a price, because the stage was still 3 from the last
    order — and that product's step-up had never been shown to anyone.

    The pair is `consult_stage` + `consult_order_key`, written together by `_advance` and nowhere
    else. The key is qty-independent on purpose: "make it four" refines the sale in progress and is
    owed the price it asked for, not a restarted walkthrough.
    """

    LOCKS = {
        "6 SW": {"product_name": "6 SW", "base_price": 4200.0, "installation_fee": 400.0},
        "Smart Door Lock Base": {
            "product_name": "Smart Door Lock Base", "base_price": 18000.0, "installation_fee": 1500.0,
        },
        "Smart Door Lock Premium": {
            "product_name": "Smart Door Lock Premium", "base_price": 28000.0, "installation_fee": 1500.0,
        },
    }
    GAINS = (
        "so you can let a guest in from your phone while you're away; "
        "a log of who came in and when; one-time codes for guests"
    )

    def test_the_key_is_the_products_and_not_how_many_of_them(self):
        from src.graph.nodes.sales import _order_key
        one = build_order(items=(("6 SW", 1), ("Zigbee Hub", 1)), offer="NONE")
        four = build_order(items=(("Zigbee Hub", 4), ("6 SW", 9)), offer="NONE")
        assert _order_key(one) == _order_key(four) == "6 sw|zigbee hub"
        assert _order_key({}) == ""

    def test_only_a_sale_sharing_nothing_starts_the_beats_again(self):
        from src.graph.nodes.sales import _is_new_sale
        lock = build_order(items=(("6 SW", 1),), offer="NONE")
        grown = build_order(items=(("6 SW", 1), ("Zigbee Hub", 1)), offer="NONE")
        other = build_order(items=(("Indoor Smart Camera", 1),), offer="NONE")
        # The same sale, larger, or at a different quantity: keeps its place in the walkthrough.
        assert _is_new_sale(lock, "6 sw") is False
        assert _is_new_sale(grown, "6 sw") is False
        # Nothing walked yet, so there is nothing to restart.
        assert _is_new_sale(lock, None) is False
        assert _is_new_sale(lock, "") is False
        # A different product entirely — and it has never been shown its own step-up.
        assert _is_new_sale(other, "6 sw") is True

    def test_advance_writes_the_key_beside_the_stage(self):
        from src.graph.nodes import sales
        order = build_order(items=(("6 SW", 1),), offer="NONE")
        updates = sales._advance(order, 0)
        assert updates["consult_order_key"] == "6 sw"
        assert updates["consult_stage"] == sales._STAGE_AFTER["quote_ask"]

    def _patch(self, monkeypatch, response):
        from src.graph.nodes import sales

        class _Engine:
            def __init__(self, session):
                pass
            async def get_product_prices_batch(self, skus):
                return {
                    s: (SimpleNamespace(**TestABeatBelongsToTheOrderItIsWalking.LOCKS[s])
                        if s in TestABeatBelongsToTheOrderItIsWalking.LOCKS else None)
                    for s in skus
                }
            async def list_catalogue_names(self):
                return list(TestABeatBelongsToTheOrderItIsWalking.LOCKS)

        class _Session:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False

        async def _fake_llm_call(llm, prompt, schema, name):
            return response

        monkeypatch.setattr(sales, "PricingEngine", _Engine)
        monkeypatch.setattr(sales, "async_session_maker", lambda: _Session())
        monkeypatch.setattr(sales.LLMFactory, "get_llm", lambda *a, **kw: object())
        monkeypatch.setattr(sales, "execute_vendor_agnostic_node", _fake_llm_call)

    def _response(self, sku, qty=1, with_upgrade=True):
        from src.core.schemas import CheckoutItem, NodeExecutionSchema
        kwargs = {
            "conversational_text": "That one's a good fit for a front door.",
            "checkout_items": [CheckoutItem(sku=sku, qty=qty)],
        }
        if with_upgrade:
            kwargs.update(
                suggested_upgrade="Smart Door Lock Premium",
                upgrade_replaces=sku,
                upgrade_reason=self.GAINS,
            )
        return NodeExecutionSchema(**kwargs)

    async def test_a_different_product_gets_its_own_step_up_instead_of_a_price(self, monkeypatch):
        from src.graph.nodes import sales
        from src.logic.prompts import HIGH_INTENT_PROMPT

        self._patch(monkeypatch, self._response("Smart Door Lock Base"))
        out = await sales._execute_sales_node(
            {
                "messages": [
                    AIMessage(content="Which of these sounds closest to your place?"),
                    HumanMessage(content="the smart door lock"),
                ],
                # Last week's sale, walked all the way to the pairing beat.
                "consult_stage": 2,
                "consult_order_key": "6 sw",
            },
            HIGH_INTENT_PROMPT,
        )

        assert out["consult_stage"] == sales._STAGE_AFTER["upsell"]
        assert out["consult_order_key"] == "smart door lock base"
        body = "\n".join(str(m.content) for m in out["messages"])
        assert "*Let a guest in from your phone while you're away*" in body
        assert "The Smart Door Lock Premium _(+₹10,000)_ also gives you:" in body
        assert "*Total*" not in body, "a new sale must not open on a price"

    async def test_more_of_the_same_thing_keeps_its_place_in_the_walkthrough(self, monkeypatch):
        from src.graph.nodes import sales
        from src.logic.prompts import HIGH_INTENT_PROMPT

        # Same product, quantity changed — "make it two" mid-walkthrough. The key ignores qty, so the
        # stage stands and the customer is not sent back through beats they have already seen.
        self._patch(monkeypatch, self._response("Smart Door Lock Base", qty=2, with_upgrade=False))
        out = await sales._execute_sales_node(
            {
                "messages": [HumanMessage(content="make it two")],
                "consult_stage": 2,
                "consult_order_key": "smart door lock base",
            },
            HIGH_INTENT_PROMPT,
        )

        assert out["consult_stage"] == sales._STAGE_AFTER["quote_ask"]
        assert out["consult_order_key"] == "smart door lock base"
        body = "\n".join(str(m.content) for m in out["messages"])
        assert "*Shall I show you the price?*" in body
        assert "also gives you" not in body


# Wording the customer must never see again, each one a decision rather than a preference:
#   "grade"           — read as a quality tier, not a price tier, and nobody asked which grade
#   "quote"/"quoted"  — what one business sends another; this is a person buying a door lock
#   "either way" / "commitment" — hedging that answers an objection the customer hasn't raised
#   "also available"  — a heading that announces an upsell instead of naming what they gain
#   "goes well with" / "common pairing" — a label where the strongest benefit belongs
_BANNED_IN_CUSTOMER_TEXT = (
    r"\bgrade\b",      # \b so "upgrade" (internal, allowed) is not a false positive
    r"\bquot",         # quote, quoted, quotation
    r"either way",
    r"commitment",
    r"also available",
    r"goes well with",
    r"common pairing",
)


def _assert_clean(label: str, text: str):
    for pattern in _BANNED_IN_CUSTOMER_TEXT:
        assert not re.search(pattern, text or "", re.IGNORECASE), f"{pattern} in {label}: {text!r}"


class TestRetiredWordingCannotComeBack:
    """
    A cheap scan over what code actually renders, rather than over its source: internal comments,
    log lines and `internal_thought` all say "quote" and "upgrade" legitimately, and a source-level
    grep would either drown in those or have to be weakened until it caught nothing. Every string
    here is one a customer reads.
    """

    LOCKS = {
        "6 SW": {"product_name": "6 SW", "base_price": 4200.0, "installation_fee": 400.0},
        "Zigbee Hub": {"product_name": "Zigbee Hub", "base_price": 4500.0, "installation_fee": 400.0},
        "Smart Door Lock Base": {
            "product_name": "Smart Door Lock Base", "base_price": 18000.0, "installation_fee": 1500.0,
        },
        "Smart Door Lock Premium": {
            "product_name": "Smart Door Lock Premium", "base_price": 28000.0, "installation_fee": 1500.0,
        },
    }
    UPGRADE_REASON = (
        "so you can let a guest in from your phone while you're away; "
        "a log of who came in and when; one-time codes for guests"
    )
    COMPLEMENT_REASON = (
        "so you can run everything from one place; open the door for a guest while you're out"
    )

    def _orders(self):
        """Every shape a customer can be shown: bare, with a step-up, with a pairing, both offers."""
        from src.graph.nodes.sales import _upgrade_button_labels

        bare = build_order(items=(("6 SW", 1),), offer="NONE")
        several = build_order(items=(("6 SW", 3), ("Zigbee Hub", 1)), offer="BUNDLE8")

        # The step-up is rendered off a real registry pair rather than a stand-in, so the wording on
        # screen is the wording a customer gets: "Switch to Premium" over a Smart Door Lock Base.
        button, keep = _upgrade_button_labels("Smart Door Lock Base", "Smart Door Lock Premium")
        lines, _unresolved, _notes = discounts.price_line_items(
            [{"sku": "Smart Door Lock Base", "qty": 1}], self.LOCKS
        )
        stepped = discounts.apply_offer(lines, "FESTIVE5")
        stepped["suggested_upgrade"] = {
            "sku": "Smart Door Lock Premium", "display_name": "Smart Door Lock Premium",
            "replaces_sku": "Smart Door Lock Base", "replaces_display": "Smart Door Lock Base",
            "qty": 1, "unit_delta": 10000.0, "line_delta": 10000.0,
            "reason": self.UPGRADE_REASON, "button_label": button, "keep_label": keep,
        }
        paired = build_order(items=(("6 SW", 2),), offer="NONE")
        paired["suggested_complement"] = {
            "sku": "Zigbee Hub", "display_name": "Zigbee Hub",
            "reason": self.COMPLEMENT_REASON, "button_label": "Add Zigbee Hub",
        }
        paired["explore_hook"] = "Want to see what the lights left on all day are costing you?"
        return {"bare": bare, "several": several, "stepped": stepped, "paired": paired}

    def test_nothing_a_customer_reads_carries_a_retired_word(self):
        from src.graph.nodes.sales import _beat_message
        for name, order in self._orders().items():
            for show in (True, False):
                _assert_clean(f"quote[{name}, show={show}]",
                              discounts.format_quote_message(order, show_suggestions=show))
            _assert_clean(f"upgrade_pitch[{name}]", "\n".join(discounts.upgrade_pitch(order)))
            _assert_clean(f"complement_pitch[{name}]", "\n".join(discounts.complement_pitch(order)))
            for beat in ("upsell", "crosssell", "quote_ask", "quote"):
                if beat == "upsell" and not order.get("suggested_upgrade"):
                    continue
                if beat == "crosssell" and not order.get("suggested_complement"):
                    continue
                _assert_clean(f"beat[{name}/{beat}]", _beat_message(order, beat).content)

    def test_no_button_label_carries_one_either(self):
        from src.graph.nodes.sales import (
            _beat_message, _quote_options, _HOOK_LABELS, _HOOK_FALLBACK_LABEL, _HOOK_FALLBACK_TEXT,
            QUOTE_NOW_LABEL, CONSULT_NEXT_LABEL, CHECKOUT_NOT_YET_OPTION, CONFIRM_CHECKOUT_OPTION,
        )
        fixed = [label for _keys, label in _HOOK_LABELS] + [
            _HOOK_FALLBACK_LABEL, _HOOK_FALLBACK_TEXT, QUOTE_NOW_LABEL, CONSULT_NEXT_LABEL,
            CHECKOUT_NOT_YET_OPTION["label"], CONFIRM_CHECKOUT_OPTION["label"],
        ]
        for label in fixed:
            _assert_clean("label constant", label)

        for name, order in self._orders().items():
            for opt in _quote_options(order):
                _assert_clean(f"quote button[{name}]", opt["label"])
            for beat in ("upsell", "crosssell", "quote_ask"):
                if beat == "upsell" and not order.get("suggested_upgrade"):
                    continue
                if beat == "crosssell" and not order.get("suggested_complement"):
                    continue
                for opt in _beat_message(order, beat).response_metadata["options"]:
                    _assert_clean(f"beat button[{name}/{beat}]", opt["label"])

    async def test_the_taps_that_change_the_order_answer_in_the_same_words(self, monkeypatch):
        # The Apply tap is where the retired word actually lived ("here's the updated quote"), and
        # every one of these three writes its own lead-in in code — so they need the same scan.
        from src.graph.nodes import sales

        class _Engine:
            def __init__(self, session):
                pass
            async def get_product_prices_batch(self, skus):
                return {
                    s: (SimpleNamespace(**TestRetiredWordingCannotComeBack.LOCKS[s])
                        if s in TestRetiredWordingCannotComeBack.LOCKS else None)
                    for s in skus
                }

        class _Session:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False

        def _no_llm(*a, **kw):
            raise AssertionError("a tap that re-prices an existing order must not call an LLM")

        monkeypatch.setattr(sales, "PricingEngine", _Engine)
        monkeypatch.setattr(sales, "async_session_maker", lambda: _Session())
        monkeypatch.setattr(sales.LLMFactory, "get_llm", _no_llm)

        order = await sales._build_pending_order(
            [SimpleNamespace(sku="Smart Door Lock Base", qty=1), SimpleNamespace(sku="6 SW", qty=1)],
            "NONE",
            suggested_complement="Zigbee Hub",
            complement_reason=self.COMPLEMENT_REASON,
            suggested_upgrade="Smart Door Lock Premium",
            upgrade_replaces="Smart Door Lock Base",
            upgrade_reason=self.UPGRADE_REASON,
        )
        assert order and order["suggested_upgrade"] and order["suggested_complement"]

        state = {"pending_order": order, "consult_stage": 3}
        for tap, out in [
            ("apply", await sales._reprice_with_best_offer(dict(state))),
            ("add", await sales._add_complement_to_order(dict(state))),
            ("swap", await sales._swap_upgrade_in_order(dict(state))),
        ]:
            for msg in out["messages"]:
                _assert_clean(f"{tap} tap", str(msg.content))
                for opt in (msg.response_metadata.get("options") or []):
                    _assert_clean(f"{tap} tap button", opt["label"])


class TestTheTotalGoesOnScreenOnlyWhenItIsAskedFor:
    """
    "Shall I show you the price?" has exactly two honest answers: the `Yes, show the price` tap, or
    saying so in words. A live chat found the third: the customer tapped the OTHER button — the one
    documented as the way out of being priced — and got an itemised total, because that button is an
    ordinary LLM turn, the model re-proposed the same products, and `_next_beat` saw a stage already
    past the ask.

    The fix is a beat that renders nothing (`hold`) plus a guarantee that the price stays one tap
    away while it does: `_keep_price_reachable` puts the same button the customer already read into
    slot one of the model's own reply. So the order is priced, validated and stored on every one of
    those turns, and the total still waits to be asked for.
    """

    LOCKS = {
        "Smart Door Lock Base": {
            "product_name": "Smart Door Lock Base", "base_price": 18000.0, "installation_fee": 1500.0,
        },
        "Smart Door Lock Premium": {
            "product_name": "Smart Door Lock Premium", "base_price": 28000.0, "installation_fee": 1500.0,
        },
        "Video Door Phone": {
            "product_name": "Video Door Phone", "base_price": 12000.0, "installation_fee": 1000.0,
        },
    }

    def _order(self):
        order = build_order(items=(("6 SW", 1),), offer="NONE")
        order["explore_hook"] = "Want to see what's costing you most on the electricity bill?"
        return order

    def test_the_hold_beat_renders_nothing_while_keeping_the_order_priced(self):
        from src.graph.nodes.sales import _advance
        update = _advance(self._order(), 3, beat="hold")
        assert update["messages"] == []
        # Priced, validated and stored: the tap that shows it has to work the moment it comes.
        assert update["pending_order"]["amount"] > 0
        assert update["consult_order_key"] == "6 sw"
        ok, reasons = Guardrails.validate_payment_request(update["pending_order"], CATALOGUE)
        assert ok, reasons
        # Still an unauthorised order, and still re-mintable.
        assert update["checkout_confirmed"] is False
        assert update["payment_link_sent"] is False

    @pytest.mark.parametrize("stage", [0, 1, 2, 3, 4])
    def test_a_hold_leaves_the_stage_exactly_where_it_was(self, stage):
        # It earns 0 so `max(stage, _STAGE_AFTER[beat])` is a no-op. A hold that advanced the stage
        # would spend a beat the customer never saw.
        from src.graph.nodes.sales import _advance
        assert _advance(self._order(), stage, beat="hold")["consult_stage"] == stage

    def test_a_lead_in_is_never_sent_on_its_own(self):
        # The lead-in exists to introduce a code-built message. With nothing rendered there is
        # nothing to introduce, and a stray "Here you go." above silence reads as a dropped message.
        from src.graph.nodes.sales import _advance
        assert _advance(self._order(), 3, lead_in="Here you go.", beat="hold")["messages"] == []

    def test_the_price_takes_slot_one_and_the_models_own_options_follow(self):
        from src.graph.nodes.sales import _keep_price_reachable, QUOTE_NOW_LABEL, QUOTE_NOW_POSTBACK
        options = _keep_price_reachable([
            {"label": "Smart lighting", "postback_id": "INTENT_LIGHTING"},
            {"label": "Curtain motors", "postback_id": "INTENT_CURTAINS"},
        ])
        assert options[0] == {"label": QUOTE_NOW_LABEL, "postback_id": QUOTE_NOW_POSTBACK}
        assert [o["label"] for o in options[1:]] == ["Smart lighting", "Curtain motors"]

    def test_the_price_button_is_never_buried_behind_a_choose_one_tap(self):
        # One description anywhere flips whatsapp.py to the LIST layout, which is exactly what would
        # hide slot one on the turn whose whole job is to keep slot one reachable.
        from src.graph.nodes.sales import _keep_price_reachable
        options = _keep_price_reachable([
            {"label": "Smart lighting", "description": "Cut what empty rooms waste", "postback_id": "INTENT_LIGHTING"},
        ])
        assert not any(o.get("description") for o in options)

    def test_it_stays_inside_whatsapps_three_buttons(self):
        from src.graph.nodes.sales import _keep_price_reachable
        many = [{"label": f"Option {i}", "postback_id": f"INTENT_{i}"} for i in range(5)]
        assert len(_keep_price_reachable(many)) == 3

    def test_a_lone_price_button_is_backfilled_so_it_cannot_read_as_being_cornered(self):
        from src.graph.nodes.sales import _keep_price_reachable, CHECKOUT_NOT_YET_OPTION
        for empty in (None, [], [{"label": "", "postback_id": ""}]):
            options = _keep_price_reachable(empty)
            assert len(options) == 2
            assert options[1] == CHECKOUT_NOT_YET_OPTION

    def test_the_model_cannot_end_up_offering_the_price_twice(self):
        from src.graph.nodes.sales import _keep_price_reachable, QUOTE_NOW_LABEL
        options = _keep_price_reachable([{"label": QUOTE_NOW_LABEL, "postback_id": "SOMETHING_ELSE"}])
        assert [o["label"] for o in options].count(QUOTE_NOW_LABEL) == 1

    def test_a_second_price_button_goes_however_it_is_worded(self):
        # The live failure: `Yes, show the price [QUOTE_NOW]` shipped beside `Show me the price
        # [SHOW_PRICE]`. Only QUOTE_NOW is gated, so the second one answered with prose and no total —
        # a button that promises the price and cannot deliver it. Matching QUOTE_NOW_LABEL exactly
        # was what let every re-wording through.
        from src.graph.nodes.sales import _keep_price_reachable, QUOTE_NOW_POSTBACK
        for label in (
            "Show me the price", "See the total", "How much is it?", "Get a quote",
            "What's the cost?", "Pricing please", "Show the figure",
        ):
            options = _keep_price_reachable([{"label": label, "postback_id": "SHOW_PRICE"}])
            assert [o["label"] for o in options] != [label], label
            assert label not in [o["label"] for o in options], label
            assert options[0]["postback_id"] == QUOTE_NOW_POSTBACK
            # Dropping it must not leave the customer with one button and no way out.
            assert len(options) == 2

    def test_a_browse_option_with_an_invented_postback_id_is_kept(self):
        # An invented id on a browse option is harmless by design: processing.py appends it to the
        # tapped title and the next LLM turn reads it as text. Only a PRICE promise needs a real gate.
        from src.graph.nodes.sales import _keep_price_reachable
        options = _keep_price_reachable([{"label": "Indoor Smart Camera", "postback_id": "ADD_INDOOR_CAMERA"}])
        assert options[1] == {"label": "Indoor Smart Camera", "postback_id": "ADD_INDOOR_CAMERA"}

    def test_no_option_set_ever_carries_two_price_buttons(self):
        from src.graph.nodes.sales import _keep_price_reachable, _PRICE_WORDS
        options = _keep_price_reachable([
            {"label": "Show me the price", "postback_id": "SHOW_PRICE"},
            {"label": "Curtain motors", "postback_id": "INTENT_CURTAINS"},
            {"label": "What would it cost?", "postback_id": "COST"},
        ])
        pricey = [o for o in options if any(w in o["label"].lower() for w in _PRICE_WORDS)]
        assert len(pricey) == 1 and pricey[0]["label"] == "Yes, show the price"

    def _patch(self, monkeypatch, response):
        from src.graph.nodes import sales

        class _Engine:
            def __init__(self, session):
                pass
            async def get_product_prices_batch(self, skus):
                return {
                    s: (SimpleNamespace(**TestTheTotalGoesOnScreenOnlyWhenItIsAskedFor.LOCKS[s])
                        if s in TestTheTotalGoesOnScreenOnlyWhenItIsAskedFor.LOCKS else None)
                    for s in skus
                }
            async def list_catalogue_names(self):
                return list(TestTheTotalGoesOnScreenOnlyWhenItIsAskedFor.LOCKS)

        class _Session:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False

        async def _fake_llm_call(llm, prompt, schema, name):
            return response

        monkeypatch.setattr(settings, "AGENT_FULL_AUTONOMY", True)
        monkeypatch.setattr(sales, "PricingEngine", _Engine)
        monkeypatch.setattr(sales, "async_session_maker", lambda: _Session())
        monkeypatch.setattr(sales.LLMFactory, "get_llm", lambda *a, **kw: object())
        monkeypatch.setattr(sales, "execute_vendor_agnostic_node", _fake_llm_call)

    def _response(self, text="Here are a couple of things worth a look.", items=True, **kw):
        from src.core.schemas import CheckoutItem, NodeExecutionSchema
        fields = {"conversational_text": text, **kw}
        if items:
            fields["checkout_items"] = [CheckoutItem(sku="Smart Door Lock Base", qty=1)]
        return NodeExecutionSchema(**fields)

    async def _run(self, monkeypatch, response, text, stage=3, **state_extra):
        from src.graph.nodes import sales
        from src.logic.prompts import HIGH_INTENT_PROMPT

        self._patch(monkeypatch, response)
        state = {
            "messages": [
                AIMessage(content="*Shall I show you the price?*"),
                HumanMessage(content=text),
            ],
            "consult_stage": stage,
            "consult_order_key": "smart door lock base",
            **state_extra,
        }
        return await sales._execute_sales_node(state, HIGH_INTENT_PROMPT)

    async def test_tapping_the_way_out_of_being_priced_does_not_price_them(self, monkeypatch):
        """The live defect, end to end: the hook tap must not produce a total."""
        from src.graph.nodes.sales import QUOTE_NOW_POSTBACK, _STAGE_AFTER

        out = await self._run(
            monkeypatch,
            self._response("Lighting is where most homes are losing money."),
            "Yes, tell me more [CHECKOUT_NOT_YET]",
        )
        body = "\n".join(str(m.content) for m in out["messages"])
        assert "*Total*" not in body
        assert "₹" not in body, "nothing on this turn may carry a figure"
        # The model's own reply is the whole turn, and the way to the price rides on it.
        assert len(out["messages"]) == 1
        assert out["messages"][0].response_metadata["options"][0]["postback_id"] == QUOTE_NOW_POSTBACK
        # The order behind that button is priced, stored and unchanged in stage.
        assert out["pending_order"]["amount"] > 0
        assert out["consult_stage"] == _STAGE_AFTER["quote_ask"]

    async def test_the_price_button_survives_a_browse_turn_that_proposes_nothing(self, monkeypatch):
        # The other shape of the same turn: the model answers the browse request without re-setting
        # checkout_items, so no order is built and no beat runs. Without the second predicate the
        # only route left to the total would be typing.
        from src.graph.nodes.sales import QUOTE_NOW_POSTBACK

        out = await self._run(
            monkeypatch,
            self._response("Curtain motors are the other one people ask about.", items=False),
            "Yes, tell me more [CHECKOUT_NOT_YET]",
            pending_order=self._order(),
        )
        options = out["messages"][0].response_metadata["options"]
        assert options[0]["postback_id"] == QUOTE_NOW_POSTBACK
        assert "*Total*" not in "\n".join(str(m.content) for m in out["messages"])

    async def test_asking_for_the_figure_in_words_still_gets_it_immediately(self, monkeypatch):
        # The bypass has to keep working: a customer who says "how much?" must not be held.
        out = await self._run(
            monkeypatch,
            self._response("Here's where it lands.", quote_requested=True),
            "so how much is all that?",
        )
        assert "*Total*" in "\n".join(str(m.content) for m in out["messages"])

    async def test_a_typed_change_after_the_price_still_re_prices(self, monkeypatch):
        # The editable-order guarantee. The hold is narrowed to the ask itself for exactly this
        # reason — holding at stage 4 would silently swallow the edit the customer just asked for.
        from src.core.schemas import CheckoutItem
        from src.graph.nodes.sales import _STAGE_AFTER

        out = await self._run(
            monkeypatch,
            self._response("Updated.", items=False, checkout_items=[
                CheckoutItem(sku="Smart Door Lock Base", qty=2)
            ]),
            "make it two",
            stage=_STAGE_AFTER["quote"],
        )
        body = "\n".join(str(m.content) for m in out["messages"])
        assert "*Total*" in body
        assert "Just type it in the message box." in body
        assert out["checkout_confirmed"] is False


class TestOneProductIsNeverPitchedTwice:
    """
    The hook is written beside the pairing product, about the gap that product fills. So the moment
    the customer taps `Add ‹product›`, the hook is a question about something they now own — and a
    live chat did exactly that: the door phone arrived as the pairing card, went into the basket, and
    the very next message asked whether they'd like to know more about seeing who's at the door.

    The carry-across is therefore gone from the add tap and kept on the swap tap, and that asymmetry
    is the point: a swap changes which model is on the line, it does not put a new product in the
    basket, so nothing the hook was about has been answered.
    """

    CATALOGUE = {
        "Smart Door Lock Base": {
            "product_name": "Smart Door Lock Base", "base_price": 18000.0, "installation_fee": 1500.0,
        },
        "Smart Door Lock Premium": {
            "product_name": "Smart Door Lock Premium", "base_price": 28000.0, "installation_fee": 1500.0,
        },
        "Video Door Phone": {
            "product_name": "Video Door Phone", "base_price": 12000.0, "installation_fee": 1000.0,
        },
    }
    HOOK = "Ever wonder who rang the bell while you were out?"

    def _order(self):
        raw = [{"sku": "Smart Door Lock Base", "qty": 1}]
        line_items, _u, _n = discounts.price_line_items(raw, self.CATALOGUE)
        order = discounts.apply_offer(line_items, "NONE")
        order["suggested_complement"] = {
            "sku": "Video Door Phone", "display_name": "Video Door Phone",
            "reason": "so you can see who's at the door before you open it; "
                      "a photo of whoever rang while you were out",
            "button_label": "Add Video Door Phone",
        }
        order["suggested_upgrade"] = {
            "sku": "Smart Door Lock Premium", "display_name": "Smart Door Lock Premium",
            "replaces_sku": "Smart Door Lock Base", "replaces_display": "Smart Door Lock Base",
            "qty": 1, "unit_delta": 10000.0, "line_delta": 10000.0,
            "reason": "so you can lock it from your phone wherever you are; "
                      "a log of who came in and when",
            "button_label": "Switch to Premium", "keep_label": "Keep the Base",
        }
        order["explore_hook"] = self.HOOK
        return order

    def _patch(self, monkeypatch):
        from src.graph.nodes import sales

        class _Engine:
            def __init__(self, session):
                pass
            async def get_product_prices_batch(self, skus):
                return {s: TestOneProductIsNeverPitchedTwice.CATALOGUE.get(s) for s in skus}

        class _Session:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False

        def _no_llm(*a, **kw):
            raise AssertionError("a tap that re-prices an existing order must not call an LLM")

        monkeypatch.setattr(sales, "PricingEngine", _Engine)
        monkeypatch.setattr(sales, "async_session_maker", lambda: _Session())
        monkeypatch.setattr(sales.LLMFactory, "get_llm", _no_llm)

    async def _add(self, monkeypatch, stage=2):
        from src.graph.nodes import sales
        self._patch(monkeypatch)
        return await sales._add_complement_to_order({
            "messages": [HumanMessage(content="Add Video Door Phone [ADD_COMPLEMENT]")],
            "pending_order": self._order(),
            "consult_stage": stage,
        })

    async def test_adding_the_pairing_product_drops_the_hook_that_was_about_it(self, monkeypatch):
        out = await self._add(monkeypatch)
        grown = out["pending_order"]
        assert [li["sku"] for li in grown["line_items"]] == ["Smart Door Lock Base", "Video Door Phone"]
        assert "explore_hook" not in grown
        # The pairing is a line now, so its suggestion goes too; the step-up is still true.
        assert "suggested_complement" not in grown
        assert grown["suggested_upgrade"]["sku"] == "Smart Door Lock Premium"

    async def test_the_ask_beat_that_follows_is_about_nothing_they_now_own(self, monkeypatch):
        from src.graph.nodes.sales import _HOOK_FALLBACK_TEXT, _HOOK_FALLBACK_LABEL

        out = await self._add(monkeypatch)
        ask = out["messages"][-1]
        assert "*Shall I show you the price?*" in ask.content
        # The fallback is deliberately about nothing in particular, which is what makes it safe here.
        assert _HOOK_FALLBACK_TEXT in ask.content
        assert self.HOOK not in ask.content
        assert "Video Door Phone" not in ask.content
        assert ask.response_metadata["options"][1]["label"] == _HOOK_FALLBACK_LABEL

    async def test_a_swap_keeps_its_hook_because_it_put_nothing_new_in_the_basket(self, monkeypatch):
        from src.graph.nodes import sales

        self._patch(monkeypatch)
        out = await sales._swap_upgrade_in_order({
            "messages": [HumanMessage(content="Switch to Premium [SWAP_UPGRADE]")],
            "pending_order": self._order(),
            "consult_stage": 1,
        })
        swapped = out["pending_order"]
        assert [li["sku"] for li in swapped["line_items"]] == ["Smart Door Lock Premium"]
        assert swapped["explore_hook"] == self.HOOK
        # Spent: it IS the order now. The pairing has still never been shown.
        assert "suggested_upgrade" not in swapped
        assert swapped["suggested_complement"]["sku"] == "Video Door Phone"


class TestTheModelsOwnWordsAreCheckedForWhatThePromptAlreadyBans:
    """
    Two rules the prompt states plainly and a live transcript broke anyway: don't praise the choice,
    and don't put both models of one product on screen. The first is repaired in code because a
    compliment is a prefix and removing it is safe; the second is only logged, because rewriting a
    whole reply mid-turn is a far bigger risk than a chatty one.
    """

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Good choice! The lock fits a front door well.", "The lock fits a front door well."),
            ("Excellent — that suits a front door.", "That suits a front door."),
            ("Wonderful! Here's what fits.", "Here's what fits."),
            ("Amazing choice, that one's popular.", "That one's popular."),
        ],
    )
    def test_a_compliment_on_the_choice_is_removed(self, text, expected):
        from src.graph.nodes.sales import _strip_praise_opener
        assert _strip_praise_opener(text) == expected

    def test_a_two_word_compliment_is_consumed_whole(self):
        # The bug this pins: "amazing" was banned on its own, "amazing choice" was not, so the bare
        # word matched, its noun stayed behind, and the customer read "Choice, that one's popular."
        # Generated openers plus longest-first ordering is what makes that unrepresentable.
        from src.graph.nodes.sales import _strip_praise_opener, _PRAISE_QUALIFIERS, _PRAISE_NOUNS
        for word in _PRAISE_QUALIFIERS:
            for noun in _PRAISE_NOUNS:
                cleaned = _strip_praise_opener(f"{word.capitalize()} {noun}! It fits a wooden frame.")
                assert cleaned == "It fits a wooden frame.", f"{word} {noun} -> {cleaned!r}"

    def test_a_word_that_is_only_praise_when_paid_to_something_is_left_alone(self):
        # "Perfect choice" is a compliment; "Perfect for a rented flat" is a recommendation, and the
        # nouns are chosen so the difference holds.
        from src.graph.nodes.sales import _strip_praise_opener
        for kept in ("Perfect for a rented flat.", "Good for a wooden frame.", "Smart for a rented place."):
            assert _strip_praise_opener(kept) == kept

    def test_a_whole_sentence_of_praise_goes_rather_than_its_first_words(self):
        # Trimming the opener alone would leave "Way to secure the front door." — worse than either
        # the original or nothing at all.
        from src.graph.nodes.sales import _strip_praise_opener
        assert _strip_praise_opener(
            "That's a great way to secure the front door. The lock fits a wooden frame."
        ) == "The lock fits a wooden frame."

    def test_a_praise_word_only_counts_when_it_stands_alone(self):
        # "Excellently quiet" describes a product; it is not a compliment paid to the customer.
        from src.graph.nodes.sales import _strip_praise_opener
        for kept in ("Excellently quiet, and it fits a wooden frame.", "Brilliantly simple to fit."):
            assert _strip_praise_opener(kept) == kept

    def test_acknowledging_the_message_is_not_praising_the_choice(self):
        # CONVERSATION_STYLE allows a three-word acknowledgement, and stripping it would leave the
        # reply opening cold on a customer who has just told us something.
        from src.graph.nodes.sales import _strip_praise_opener
        for kept in ("Perfect. What's the door made of?", "Got it. Two panels, then.", "Right — a front door."):
            assert _strip_praise_opener(kept) == kept

    def test_it_never_raises_on_junk(self):
        from src.graph.nodes.sales import _strip_praise_opener
        for junk in (None, "", "   ", "Good choice!", "*Good choice*"):
            _strip_praise_opener(junk)

    def test_both_models_of_one_product_on_screen_together_are_spotted(self):
        from src.graph.nodes.sales import _menued_registry_pair
        found = _menued_registry_pair(
            "The Smart Door Lock Base covers the basics; the Smart Door Lock Premium adds more.", None
        )
        assert found == ("Smart Door Lock Base", "Smart Door Lock Premium")
        # Buttons count too — a menu is a menu whether it is prose or two options.
        assert _menued_registry_pair("Which suits you?", [
            SimpleNamespace(label="Base lock", description="Smart Door Lock Base", postback_id="A"),
            SimpleNamespace(label="Premium lock", description="Smart Door Lock Premium", postback_id="B"),
        ]) == ("Smart Door Lock Base", "Smart Door Lock Premium")

    def test_naming_one_of_them_is_exactly_what_the_agent_is_asked_to_do(self):
        from src.graph.nodes.sales import _menued_registry_pair
        assert _menued_registry_pair("The Smart Door Lock Base suits a wooden front door.", None) is None
        assert _menued_registry_pair("The Smart Door Lock Premium is the one for a rented flat.", None) is None
        assert _menued_registry_pair("", None) is None
        assert _menued_registry_pair(None, None) is None

    def test_the_plain_english_gloss_cannot_trip_it(self):
        # plain_product_name is identical for both halves of a pair, so matching on it would fire on
        # either one alone and the warning would mean nothing.
        from src.graph.nodes.sales import _menued_registry_pair
        for frm, spec in discounts.UPGRADES.items():
            gloss = discounts.plain_product_name(frm)
            if gloss and gloss.lower() != frm.lower():
                assert _menued_registry_pair(f"The {gloss} is a good fit.", None) is None

    def test_every_registry_pair_is_covered_not_just_the_lock(self):
        from src.graph.nodes.sales import _menued_registry_pair
        for frm, spec in discounts.UPGRADES.items():
            to = spec["to"]
            assert _menued_registry_pair(f"Either the {frm} or the {to} would work.", None) == (frm, to)
