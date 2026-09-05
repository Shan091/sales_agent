import logging
from typing import Any
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from src.core.llm_factory import LLMFactory, execute_vendor_agnostic_node
from src.core.schemas import TriageClassification
from src.logic.prompts import triage_prompt, OFFICE_HOURS
from src.core.business_hours import is_within_business_hours
from src.graph.state import ConversationState, last_user_text
from config.settings import settings

logger = logging.getLogger(__name__)


# Archetypes that represent an active, resumable sales/support thread. A bare "hi"
# or a low-effort "ok" mid-conversation should RESUME one of these (don't cold-restart).
# Crucially this excludes OUT_OF_DOMAIN, GENERAL_GREETING and HUMAN_ESCALATION: resuming
# OUT_OF_DOMAIN was making every later "hi" inherit a deflecting out-of-domain reply, so
# the conversation got stuck in a ditch it could never climb out of.
RESUMABLE_ARCHETYPES = {
    "SALES_HIGH_INTENT",
    "SALES_WINDOW_SHOPPER",
    "SALES_PROBLEM_SOLVER",
    "B2B_ENTERPRISE",
    "POST_SALE_SUPPORT",
}


import time

async def node_triage(state: ConversationState):
    """
    Analyzes intent, archetype, and language. Updates state.
    """
    current_time = time.time()
    last_timestamp = state.get("last_user_message_timestamp", current_time)

    # Safety net on a human-held thread. Handback is human-driven by design — a colleague
    # releases the thread with src/scripts/resolve_handoff.py once they're done. This only
    # covers the case where nobody remembers to, so a customer isn't left talking to an agent
    # that refuses to engage. Set HANDOFF_MAX_HOLD_HOURS=0 to require an explicit release always.
    handoff_release: dict = {}
    if state.get("handoff_active") and settings.HANDOFF_MAX_HOLD_HOURS > 0:
        started = state.get("handoff_started_at") or 0.0
        held_hours = (current_time - started) / 3600.0 if started else 0.0
        if held_hours >= settings.HANDOFF_MAX_HOLD_HOURS:
            logger.warning(
                f"Handoff held {held_hours:.1f}h without release (limit "
                f"{settings.HANDOFF_MAX_HOLD_HOURS}h). Auto-releasing to the agent."
            )
            handoff_release = {
                "handoff_active": False,
                "handoff_notified": False,
                "handoff_reason": None,
                "handoff_started_at": None,
                "requires_human_handoff": False,
                "handoff_notes": list(state.get("handoff_notes") or []) + [
                    "Auto-released after the maximum hold elapsed; no note was left by the team."
                ],
            }
            state = {**state, **handoff_release}

    # 1. Fast-path deterministic button taps before calling the LLM
    is_confirm = _is_confirm_checkout(state)
    if is_confirm:
        logger.info("Checkout confirmation detected in Triage. Bypassing LLM.")
        return {
            **handoff_release,
            "current_archetype": "SALES_HIGH_INTENT",
            "language_preference": state.get("language_preference", "English"),
            "requires_human_handoff": False,
            "human_request_count": 0,
            "data_routing_flag": "NONE",
            "checkout_confirmed": True,
            "last_user_message_timestamp": current_time,
            "primary_interest": state.get("primary_interest"),
        }
        
    is_connect_now = _is_connect_now(state)
    if is_connect_now:
        logger.info("Connect Now escape hatch tapped. Bypassing LLM and escalating.")
        return {
            **handoff_release,
            "current_archetype": "HUMAN_ESCALATION",
            "language_preference": state.get("language_preference", "English"),
            "requires_human_handoff": True,
            "handoff_reason": state.get("handoff_reason") or "requested",
            "human_request_count": 0,
            "data_routing_flag": "NONE",
            "checkout_confirmed": False,
            "last_user_message_timestamp": current_time,
            "primary_interest": state.get("primary_interest"),
        }

    # Applying an available offer is a re-pricing request, not an authorisation to charge —
    # so it deliberately leaves checkout_confirmed False. The customer still has to tap
    # "Confirm & pay" on the re-priced quote.
    if _is_apply_offer(state) and state.get("pending_order"):
        logger.info("Apply-offer tap detected in Triage. Bypassing LLM.")
        return {
            **handoff_release,
            "current_archetype": "SALES_HIGH_INTENT",
            "language_preference": state.get("language_preference", "English"),
            "requires_human_handoff": False,
            "human_request_count": 0,
            "data_routing_flag": "NONE",
            "checkout_confirmed": False,
            "apply_offer_requested": True,
            "last_user_message_timestamp": current_time,
            "primary_interest": state.get("primary_interest"),
        }

    # Adding the suggested complement is also a re-pricing request, not an authorisation to
    # charge. Same reasoning as the offer tap: the new total has to be confirmed on its own.
    if _is_add_complement(state) and state.get("pending_order"):
        logger.info("Add-complement tap detected in Triage. Bypassing LLM.")
        return {
            **handoff_release,
            "current_archetype": "SALES_HIGH_INTENT",
            "language_preference": state.get("language_preference", "English"),
            "requires_human_handoff": False,
            "human_request_count": 0,
            "data_routing_flag": "NONE",
            "checkout_confirmed": False,
            "add_complement_requested": True,
            "last_user_message_timestamp": current_time,
            "primary_interest": state.get("primary_interest"),
        }

    # ── The three walkthrough taps ───────────────────────────────────────────────────────────
    # Every one of them acts on an order already priced and sitting in state, so none needs the
    # model, and none of them authorises a charge. They are checked here, above the classifier,
    # for the same reason as the taps above: a button whose meaning code already knows must not
    # depend on a classification going the right way.
    walkthrough_flag = None
    if _is_swap_upgrade(state):
        walkthrough_flag = "swap_upgrade_requested"
    elif _is_consult_next(state):
        walkthrough_flag = "consult_next_requested"
    elif _is_quote_now(state):
        walkthrough_flag = "quote_now_requested"
    if walkthrough_flag and state.get("pending_order"):
        logger.info(f"Walkthrough tap detected in Triage ({walkthrough_flag}). Bypassing LLM.")
        return {
            **handoff_release,
            "current_archetype": "SALES_HIGH_INTENT",
            "language_preference": state.get("language_preference", "English"),
            "requires_human_handoff": False,
            "human_request_count": 0,
            "data_routing_flag": "NONE",
            "checkout_confirmed": False,
            walkthrough_flag: True,
            "last_user_message_timestamp": current_time,
            "primary_interest": state.get("primary_interest"),
        }

    llm = LLMFactory.get_llm(temperature=0.0, fast=True)

    formatted = triage_prompt.format_messages(chat_history=state["messages"])
    response = await execute_vendor_agnostic_node(llm, formatted, TriageClassification, "triage_node")

    if not response:
        logger.error("Triage execution failed. Escaping to Human.")
        return {"requires_human_handoff": True, "handoff_reason": "error", "current_archetype": "HUMAN_ESCALATION", "checkout_confirmed": False, "last_user_message_timestamp": current_time}

    # 1. 48-Hour Ghost Return Check (Stress-Test Fix)
    if (current_time - last_timestamp) > 172800 and state.get("current_archetype") and state.get("current_archetype") not in ["GENERAL_GREETING", "HUMAN_ESCALATION"]:
        logger.info("48h Ghost Return detected. Forcing Contextual Rewarm.")
        return {
            **handoff_release,
            "current_archetype": "CONTEXTUAL_REWARM",
            "language_preference": response.detected_language,
            "requires_human_handoff": False,
            "data_routing_flag": response.data_routing_flag,
            "checkout_confirmed": is_confirm,
            "last_user_message_timestamp": current_time
        }

    logger.info(f"Triage Result: {response.archetype} | Lang: {response.detected_language}")

    archetype = response.archetype
    active_archetype = state.get("current_archetype")

    # 2. State-Aware "Hi" (Re-Engagement Protection)
    # Only resume a genuine sales/support thread — never OUT_OF_DOMAIN (that caused a
    # stuck loop where every fresh "hi" got a deflecting out-of-domain reply).
    if archetype == "GENERAL_GREETING" and active_archetype in RESUMABLE_ARCHETYPES:
        logger.info(f"State-Aware Hi triggered. Ignoring GENERAL_GREETING and restoring {active_archetype}.")
        archetype = active_archetype

    # 3. "Ok" Affirmation Bounce
    if getattr(response, "is_affirmation", False):
        if active_archetype in RESUMABLE_ARCHETYPES:
            logger.info(f"Affirmation detected. Bouncing back to {active_archetype}.")
            archetype = active_archetype
        elif not active_archetype:
            logger.info("Naked affirmation detected (no prior state). Forcing GENERAL_GREETING.")
            archetype = "GENERAL_GREETING"

    # 4. Human handoff — gentle by default (do not forward on the first calm ask).
    #    - Angry/abusive (is_frustrated) or an explicit HUMAN_ESCALATION classification -> escalate now.
    #    - A calm "can I talk to someone" (wants_human) -> PROBE first (ask what it's about), and only
    #      escalate on a REPEAT ask or if they tap the "Connect me now" escape hatch.
    #    - "repeatedly" = consecutive: the counter resets whenever they move on, so a later single ask
    #      never gets ambushed by a stale count.
    wants_human = getattr(response, "wants_human", False)
    human_count = state.get("human_request_count", 0)

    # Belt-and-suspenders: tapping the "Connect me now" escape hatch must escalate immediately,
    # regardless of the counter, so the hatch never re-probes.
    # Only carry over the handoff reason if we are currently holding the thread.
    # Otherwise, an old "error" reason from a transient failure will pollute new escalations.
    handoff_reason = state.get("handoff_reason") if state.get("handoff_active") else None
    if _is_connect_now(state):
        handoff = True
        archetype = "HUMAN_ESCALATION"
        human_count = 0
        handoff_reason = handoff_reason or "requested"
    elif response.is_frustrated:
        handoff = True
        archetype = "HUMAN_ESCALATION"
        human_count = 0
        handoff_reason = "upset"
    elif archetype == "HUMAN_ESCALATION":
        handoff = True
        human_count = 0
        # The triage prompt reserves this classification for the critical set, so attribute it
        # to the reason the model can actually see. Anything more specific (a repeated payment
        # failure, a post-payment dispute) is set at its own source and preserved above.
        handoff_reason = handoff_reason or "safety"
    elif wants_human:
        human_count += 1
        if human_count >= 2:
            # They asked again after we offered to help -> honour it.
            logger.info("Repeat human request detected. Escalating.")
            handoff = True
            archetype = "HUMAN_ESCALATION"
            human_count = 0
            handoff_reason = "requested"
        else:
            # First calm ask -> warm probe, do NOT escalate yet.
            logger.info("Calm human request. Routing to HUMAN_PROBE before any escalation.")
            handoff = False
            archetype = "HUMAN_PROBE"
    else:
        handoff = False
        human_count = 0  # Reset: consecutive-request semantics for "repeatedly asking".

    return {
        **handoff_release,
        "current_archetype": archetype,
        "language_preference": response.detected_language,
        "requires_human_handoff": handoff,
        "handoff_reason": handoff_reason,
        "human_request_count": human_count,
        "data_routing_flag": response.data_routing_flag,
        "checkout_confirmed": is_confirm,
        "last_user_message_timestamp": current_time,
        "deferred_purchase_intent": getattr(response, "deferred_purchase_intent", None),
        # Sticky: keep the last known interest when this turn doesn't name a new one,
        # so a later "how much is it?" still has a product to price-check.
        "primary_interest": response.primary_interest or state.get("primary_interest"),
    }


def _is_connect_now(state: ConversationState) -> bool:
    """True if the latest user message is the 'Connect me now' escape-hatch tap/text."""
    text = last_user_text(state)
    return "connect_now" in text or "connect me now" in text


def _is_confirm_checkout(state: ConversationState) -> bool:
    """
    True if the latest user message is the explicit 'Confirm & pay' tap that authorises
    minting a payment link. The interactive button forwards its title text into the
    message, so we match that; the postback id 'CONFIRM_CHECKOUT' is accepted too,
    belt-and-braces, in case the raw id ever flows through.

    This is the gate. No tap -> checkout_confirmed stays False -> the worker never mints a
    link, no matter what the LLM says. Deliberately tight ('confirm & pay' is a distinctive
    phrase) so ordinary conversation can't trip it.
    """
    text = last_user_text(state)
    return "confirm_checkout" in text or "confirm & pay" in text


def _is_apply_offer(state: ConversationState) -> bool:
    """
    True if the latest user message is the 'Apply <offer>' tap on a quote that was priced at
    list price while an offer was in fact available. Re-prices the existing order in code —
    it is NOT an authorisation to charge, so it never sets checkout_confirmed.
    """
    text = last_user_text(state)
    return "apply_offer" in text or text.startswith("apply ")


def _is_add_complement(state: ConversationState) -> bool:
    """
    True if the latest user message is the 'Add <product>' tap on a quote.

    Matched on the postback id, not on the label: unlike the other buttons this label is built
    per-order from the suggested product, so there is no fixed phrase to match. The stored label
    is accepted as well, for the same belt-and-braces reason as the other gates.

    Deliberately NOT matched on a bare "add ..." prefix. A customer typing "add a curtain motor
    too" means a specific product, and this path adds whatever complement is stored on the order
    — which could be a different one. That message belongs to the LLM, which can read what they
    actually asked for.
    """
    text = last_user_text(state)
    if "add_complement" in text:
        return True
    label = ((state.get("pending_order") or {}).get("suggested_complement") or {}).get("button_label")
    return bool(label) and label.strip().lower() in text


def _is_swap_upgrade(state: ConversationState) -> bool:
    """
    True if the latest user message is the step-up tap ("Switch to Premium") shown during the
    consultative walkthrough.

    Matched on the postback id: like the add button, this label is built per-order from the pair
    the model proposed and code verified, so there is no fixed phrase. A re-pricing request, never
    an authorisation to charge — the swapped order still has to be quoted and confirmed.
    """
    text = last_user_text(state)
    if "swap_upgrade" in text:
        return True
    label = ((state.get("pending_order") or {}).get("suggested_upgrade") or {}).get("button_label")
    return bool(label) and label.strip().lower() in text


def _is_consult_next(state: ConversationState) -> bool:
    """
    True if the customer tapped past a walkthrough beat without taking it ("Keep the Base",
    "Just this for now"). Advances to the next beat and changes no money at all.

    Postback id only. The labels are written to avoid making anyone state a refusal, which means
    they are ordinary phrases ("keep the base") that must not be matched loosely in free text.
    """
    return "consult_next" in last_user_text(state)


def _is_quote_now(state: ConversationState) -> bool:
    """
    True if the customer tapped the quote button at the end of the walkthrough.

    Renders the itemised quote from the order already in state — code, no LLM call — and does NOT
    set checkout_confirmed: seeing a price is not agreeing to pay it. A customer who *types* a
    request for the price is handled by the model instead (NodeExecutionSchema.quote_requested),
    because recognising "what's the damage" as a price request needs to read the sentence.
    """
    return "quote_now" in last_user_text(state)


# What the customer reads when a person takes over, keyed by WHY. One canned sentence for
# every cause reads as an error page: a repeated card decline, an angry customer and a refund
# request are different situations and deserve different words. No emojis here — this is the
# moment someone is already unhappy, and cheerfulness reads as not listening.
_HANDOFF_LINES = {
    "payment": {
        "English": "I'm sorry that didn't work. I've asked a colleague to take this over and get your order completed — they have the full details, so there's nothing for you to repeat.",
        "Hindi": "क्षमा करें, भुगतान पूरा नहीं हो सका। मैंने एक सहयोगी को यह ज़िम्मा दिया है जो आपका ऑर्डर पूरा कराएँगे — उनके पास पूरी जानकारी है, आपको कुछ दोहराना नहीं पड़ेगा।",
        "Malayalam": "ക്ഷമിക്കണം, പേയ്‌മെന്റ് പൂർത്തിയായില്ല. നിങ്ങളുടെ ഓർഡർ പൂർത്തിയാക്കാൻ ഞാൻ ഒരു സഹപ്രവർത്തകനെ ഏർപ്പെടുത്തിയിട്ടുണ്ട് — എല്ലാ വിവരങ്ങളും അവരുടെ പക്കലുണ്ട്, നിങ്ങൾ ഒന്നും ആവർത്തിക്കേണ്ടതില്ല.",
    },
    "post_payment": {
        "English": "I've passed this to a colleague who handles completed orders — refunds and order changes are theirs to sort, not mine to guess at. They have your full order details.",
        "Hindi": "मैंने यह उस सहयोगी को भेज दिया है जो पूर्ण हो चुके ऑर्डर देखते हैं — रिफ़ंड और ऑर्डर में बदलाव उनका काम है। उनके पास आपके ऑर्डर का पूरा विवरण है।",
        "Malayalam": "പൂർത്തിയായ ഓർഡറുകൾ കൈകാര്യം ചെയ്യുന്ന സഹപ്രവർത്തകന് ഞാൻ ഇത് കൈമാറി — റീഫണ്ടും ഓർഡർ മാറ്റങ്ങളും അവരുടെ ചുമതലയാണ്. നിങ്ങളുടെ ഓർഡർ വിവരങ്ങൾ അവരുടെ പക്കലുണ്ട്.",
    },
    "upset": {
        "English": "That's a fair thing to be annoyed about, and I'd rather a person handled it than have me keep trying. A colleague is picking this up with your full conversation in front of them.",
        "Hindi": "आपकी नाराज़गी जायज़ है, और मैं चाहूँगा कि इसे कोई व्यक्ति संभाले। एक सहयोगी आपकी पूरी बातचीत के साथ यह देख रहे हैं।",
        "Malayalam": "നിങ്ങളുടെ അസ്വസ്ഥത ന്യായമാണ്; ഇത് ഒരു വ്യക്തി കൈകാര്യം ചെയ്യുന്നതാണ് നല്ലത്. നിങ്ങളുടെ മുഴുവൻ സംഭാഷണവുമായി ഒരു സഹപ്രവർത്തകൻ ഇത് ഏറ്റെടുക്കുന്നു.",
    },
    "safety": {
        "English": "I'm routing this to our team for you. A colleague is taking it on with everything you've told me. They'll be in touch within couple of hours.",
        "Hindi": "मैं इसे आपके लिए हमारी टीम को भेज रहा हूँ। एक सहयोगी आपकी दी गई सारी जानकारी के साथ यह देख रहे हैं। वे कुछ घंटों में आपसे संपर्क करेंगे।",
        "Malayalam": "കൂടുതൽ സഹായത്തിനായി ഞാൻ ഇത് ഞങ്ങളുടെ ടീമിന  ് കൈമാറുകയാണ്. നിങ്ങൾ പറഞ്ഞ എല്ലാ വിവരങ്ങളുമായി ഒരു സഹപ്രവർത്തകൻ ഇത് ഏറ്റെടുക്കുന്നു. അവർ ഏതാനും മണിക്കൂറിനുള്ളിൽ നിങ്ങളെ ബന്ധപ്പെടും.",
    },
    "error": {
        "English": "Something went wrong on my side just now. Rather than waste your time, I've handed this to a colleague along with the whole conversation.",
        "Hindi": "अभी मेरी तरफ़ कुछ गड़बड़ हो गई। आपका समय बरबाद करने के बजाय मैंने यह पूरी बातचीत के साथ एक सहयोगी को दे दिया है।",
        "Malayalam": "ഇപ്പോൾ എന്റെ ഭാഗത്ത് എന്തോ പ്രശ്നമുണ്ടായി. നിങ്ങളുടെ സമയം പാഴാക്കാതെ, മുഴുവൻ സംഭാഷണവും സഹിതം ഞാൻ ഇത് ഒരു സഹപ്രവർത്തകനെ ഏൽപ്പിച്ചു.",
    },
    "requested": {
        "English": "Of course. A colleague from the team is taking over from here, and they have the whole conversation, so you won't need to go over it again. They'll be in touch within couple of hours.",
        "Hindi": "बिलकुल। यहाँ से टीम का एक सहयोगी आगे संभालेंगे, और उनके पास पूरी बातचीत है, इसलिए आपको दोबारा कुछ बताने की ज़रूरत नहीं। वे कुछ घंटों में आपसे संपर्क करेंगे।",
        "Malayalam": "തീർച്ചയായും. ഇവിടെ നിന്ന് ടീമിലെ ഒരു സഹപ്രവർത്തകൻ തുടർന്ന് സഹായിക്കും; മുഴുവൻ സംഭാഷണവും അവരുടെ പക്കലുണ്ട്, അതിനാൽ വീണ്ടും പറയേണ്ടതില്ല. അവർ ഏതാനും മണിക്കൂറിനുള്ളിൽ നിങ്ങളെ ബന്ധപ്പെടും.",
    },
}

# Sent for every further message while a person still owns the thread. Short on purpose: the
# customer has already been told what's happening, and repeating the full notice each time is
# what makes an agent feel broken.
_HANDOFF_HOLDING_LINES = {
    "English": "You're still with the team on this one — I've passed your message along so nothing gets missed.",
    "Hindi": "यह मामला अभी टीम के पास है — मैंने आपका संदेश उन्हें भेज दिया है, कुछ छूटेगा नहीं।",
    "Malayalam": "ഇത് ഇപ്പോഴും ടീമിന്റെ പക്കലാണ് — നിങ്ങളുടെ സന്ദേശം ഞാൻ അവർക്ക് കൈമാറി, ഒന്നും വിട്ടുപോകില്ല.",
}


def _timing_line(language: str) -> str:
    """When to expect a reply — inside business hours vs outside. Kept separate from the
    reason lines so every reason gets an honest expectation without duplicating it six times."""
    if is_within_business_hours():
        return {
            "English": "They'll be in touch within about 2 working hours.",
            "Hindi": "वे लगभग 2 कार्य-घंटों के भीतर संपर्क करेंगे।",
            "Malayalam": "ഏകദേശം 2 പ്രവൃത്തി മണിക്കൂറിനുള്ളിൽ അവർ ബന്ധപ്പെടും.",
        }.get(language, "They'll be in touch within about 2 working hours.")
    return {
        "English": f"The team is offline right now ({OFFICE_HOURS}), so they'll reach out the next working day, before 11 AM.",
        "Hindi": f"टीम इस समय ऑफ़लाइन है ({OFFICE_HOURS}), इसलिए वे अगले कार्य-दिवस, सुबह 11 बजे से पहले संपर्क करेंगे।",
        "Malayalam": f"ടീം ഇപ്പോൾ ഓഫ്‌ലൈനാണ് ({OFFICE_HOURS}), അതിനാൽ അടുത്ത പ്രവൃത്തി ദിവസം രാവിലെ 11 മണിക്ക് മുമ്പ് അവർ ബന്ധപ്പെടും.",
    }.get(language, f"The team is offline right now ({OFFICE_HOURS}), so they'll reach out the next working day, before 11 AM.")


async def node_human_escalation(state: ConversationState):
    """
    Tell the customer a person is taking over, and mark the thread as theirs.

    Two distinct messages, because this node is reached twice for different reasons: the FIRST
    time to announce the handoff, and on every later message while the hold lasts. Sending the
    full announcement each time is what makes an agent look like it isn't reading.

    Sets handoff_active, which is sticky — a person now owns this thread on their own number, so
    the agent must not resume just because the next message looks ordinary. Only
    src/scripts/resolve_handoff.py (or the HANDOFF_MAX_HOLD_HOURS safety net) clears it.
    """
    language = state.get("language_preference", "English")
    already_notified = state.get("handoff_notified", False)

    if already_notified:
        text = _HANDOFF_HOLDING_LINES.get(language, _HANDOFF_HOLDING_LINES["English"])
        return {
            "messages": [AIMessage(
                content=text,
                response_metadata={"options": None, "internal_thought": "Thread still held by a human; short holding ack."},
            )],
            "requires_human_handoff": True,
            "handoff_active": True,
        }

    reason = state.get("handoff_reason") or "requested"
    lines = _HANDOFF_LINES.get(reason, _HANDOFF_LINES["requested"])
    text = lines.get(language, lines["English"]) + " " + _timing_line(language)

    # A TECHNICAL failure is not a customer needing a person, and it must not freeze the thread.
    # `error` means our own model call fell over — a credit limit, a timeout, a bad response — and
    # observed live, one of those locked a conversation behind the holding line for every message
    # afterwards: the customer said "Hi", got "I've handed this to a colleague", and the only way out
    # was a colleague noticing and releasing it. So the team is still told (the escalation row and
    # its email still go out) and the customer still gets the apology, but the hold is NOT made
    # sticky: `requires_human_handoff` is recomputed by triage on the next message, so the agent
    # retries by itself and a transient outage costs one turn instead of the conversation. Every
    # other reason — asked for a person, unresolved anger, payment trouble, safety — stays sticky,
    # because there a person genuinely owns the thread.
    sticky = reason != "error"
    if not sticky:
        logger.info("Technical-error escalation: telling the team, but not holding the thread.")

    return {
        "messages": [AIMessage(
            content=text,
            response_metadata={"options": None, "internal_thought": f"Human escalation ({reason}). Announcing handoff."},
        )],
        "requires_human_handoff": True,
        "handoff_active": sticky,
        "handoff_notified": sticky,
        "handoff_started_at": time.time() if sticky else None,
        "handoff_reason": reason,
    }


async def node_adversarial_block(state: ConversationState):
    """
    Handles MALICIOUS_ADVERSARIAL users: prompt injection, jailbreaks, trolls.
    Sends a polite but firm rejection. Does NOT hand off to humans.
    """
    ai_msg = AIMessage(
        content="I appreciate you reaching out! I'm Otohom's smart home assistant and I'm here to help with home automation, security, and lighting solutions. If you have any questions about our products, I'd love to help. 😊",
        response_metadata={"options": None, "internal_thought": "Adversarial input detected. Polite deflection without human escalation."}
    )

    return {
        "messages": [ai_msg],
        "requires_human_handoff": False
    }
