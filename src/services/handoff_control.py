"""
Giving a thread back to the agent after a person has dealt with it.

Handoff is human-driven and sticky: `node_human_escalation` marks a thread `handoff_active` and the
agent answers with a short holding line until a colleague releases it. That is deliberate — the
colleague works the customer on a DIFFERENT WhatsApp number, so nothing they say is visible here, and
an agent that resumed on its own would be answering blind, in parallel with a real person, on two
numbers at once.

Which is also the answer to "how does the agent get the context of the human conversation": through
the note released with it, and nowhere else. Whatever is written there is what the agent knows;
anything left out, it will not know. `sales.py::_build_handoff_block` injects it into every later
sales prompt, and `crm_handoff.build_digest` puts it on the sheet row.

This module is the single implementation, called by two channels that must not drift:
  * `src/scripts/resolve_handoff.py` — a terminal.
  * a `#done` / `#back` / `#status` WhatsApp message from an allowlisted colleague, handled in
    `tasks/processing.py`. A salesperson has a phone, not a shell, so in practice this is the one
    that gets used — before it existed, holds were released by the 24-hour safety net and the
    outcome was never recorded at all.
"""
import logging
import re
import time
from typing import Any, Awaitable, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# `#done` releases silently; `#back` also tells the customer the agent has picked things up again.
# Two prefixes rather than one plus a flag, because a flag is one more thing to mistype on a phone.
_COMMAND_RE = re.compile(
    r"^\s*#\s*(?P<command>done|back|status)\b[\s:]*(?P<target>\+?\d[\d\s\-()]{6,20})?(?P<note>[\s\S]*)$",
    re.IGNORECASE,
)

STAFF_COMMAND_HELP = (
    "Handoff commands:\n"
    "  #done <number> <what you did>   release the thread, agent carries on with your note\n"
    "  #back <number> <what you did>   same, and tell the customer the agent is back\n"
    "  #status <number>                show the hold without changing anything"
)


def looks_like_staff_command(text: str) -> bool:
    """Cheap check used before anything else: does this even start with a handoff command?"""
    return bool(_COMMAND_RE.match(text or ""))


def parse_staff_command(text: str) -> Optional[Tuple[str, str, str]]:
    """
    `(command, target_thread_id, note)` — or None when this is not a handoff command at all.

    The target is reduced to digits, because a colleague typing a number on a phone will include
    spaces, dashes or a leading +. A missing number is returned as an empty target so the caller can
    answer with the help text rather than silence; the note may legitimately be empty for `#status`.
    """
    match = _COMMAND_RE.match(text or "")
    if not match:
        return None
    command = match.group("command").lower()
    target = "".join(ch for ch in (match.group("target") or "") if ch.isdigit())
    note = " ".join((match.group("note") or "").split())
    return command, target, note


def thread_candidates(target: str, sender: Optional[str] = None) -> List[str]:
    """
    The thread ids a typed number could mean, best guess first.

    A colleague types the number the way they say it — `9812345678` — while a conversation is keyed
    by what Meta delivers, which is the full international form `919812345678`. Looking up the digits
    as typed found nothing and answered "No conversation found", **while the customer was still
    being held**, which is the most confusing possible failure: the command looked wrong when only
    the number was.

    The country prefix is taken from the colleague's OWN number (everything before its last ten
    digits) rather than hard-coded, so this works for whatever country the team is in. Both spellings
    are tried, so pasting the full number keeps working.
    """
    digits = "".join(ch for ch in (target or "") if ch.isdigit())
    if not digits:
        return []
    out = [digits]
    sender_digits = "".join(ch for ch in (sender or "") if ch.isdigit())
    prefix = sender_digits[:-10] if len(sender_digits) > 10 else ""
    if prefix and len(digits) == 10:
        out.insert(0, prefix + digits)          # the likelier reading goes first
    elif len(digits) > 10:
        out.append(digits[-10:])
    return list(dict.fromkeys(out))


async def resolve_thread(
    app: Any, target: str, sender: Optional[str] = None
) -> tuple[Optional[str], List[str]]:
    """
    `(the candidate that actually has a conversation, every candidate tried)`.

    Returning the tried list is not decoration: when nothing matches, the reply names what was looked
    for, so the next person does not have to guess whether the command or the number was wrong.
    """
    candidates = thread_candidates(target, sender)
    for candidate in candidates:
        try:
            snap = await app.aget_state({"configurable": {"thread_id": candidate}})
        except Exception as e:
            logger.warning(f"Could not read state for {candidate}: {e}")
            continue
        if snap and snap.values:
            return candidate, candidates
    return None, candidates


async def handoff_status(app: Any, thread_id: str) -> str:
    """A short, plain-text status for one thread. Reads only — changes nothing."""
    snap = await app.aget_state({"configurable": {"thread_id": thread_id}})
    state = snap.values if snap else {}
    if not state:
        return f"No conversation found for {thread_id}."

    lines = [f"{thread_id}"]
    if state.get("handoff_active"):
        started = state.get("handoff_started_at")
        held = f"{(time.time() - started) / 3600.0:.1f}h" if started else "unknown"
        lines.append(f"held by a person for {held} — reason: {state.get('handoff_reason') or 'unspecified'}")
    else:
        lines.append("not held — the agent is answering")
    if state.get("last_payment_status"):
        lines.append(f"payment: {state['last_payment_status']}")
    order = state.get("pending_order") or {}
    if order:
        lines.append(f"order: {order.get('product_summary')} — {order.get('amount')}")
    notes = [n for n in (state.get("handoff_notes") or []) if n]
    if notes:
        lines.append(f"last note: {notes[-1]}")
    return "\n".join(lines)


async def release_handoff(
    app: Any,
    thread_id: str,
    note: str,
    tell_customer: bool = False,
    notify: Optional[Callable[[str, str], Awaitable[bool]]] = None,
) -> str:
    """
    Clear the hold and record what the person did. Returns a one-line outcome for the caller to
    print or send back.

    `notify(thread_id, text)` sends the customer the "I'm back" line; it is injected so this module
    needs no WhatsApp or Redis import — the CLI builds its own client, the worker passes the one it
    already has. A failure to notify does NOT undo the release: the hold is the thing that matters,
    and leaving it on because a cosmetic message failed would strand the customer.

    Both delivery dedup flags are reset, so a thread that later needs a person again for a NEW
    reason is delivered to the team again rather than being silently swallowed as "already sent".
    """
    config = {"configurable": {"thread_id": thread_id}}
    snap = await app.aget_state(config)
    state = snap.values if snap else {}
    if not state:
        return f"No conversation found for {thread_id}. Nothing to release."
    if not state.get("handoff_active"):
        return f"{thread_id} is not currently held by a person — nothing to release."
    if not (note or "").strip():
        return (
            f"{thread_id} is held, but I need to know what happened before I hand it back — "
            "whatever you write is all the agent will know."
        )

    notes = list(state.get("handoff_notes") or [])
    notes.append(note.strip())

    await app.aupdate_state(config, {
        "handoff_active": False,
        "handoff_notified": False,
        "handoff_reason": None,
        "handoff_started_at": None,
        "requires_human_handoff": False,
        # A resolved decline shouldn't leave the customer one attempt from escalation again.
        "payment_failure_count": 0,
        "handoff_notes": notes,
        # Let the sinks fire again if this thread later needs a person, or becomes a lead, anew.
        "lead_sent": False,
        "escalation_sent": False,
    })
    logger.info(f"[{thread_id}] Handoff released. Note recorded: {note.strip()}")

    told = ""
    if tell_customer and notify is not None:
        try:
            await notify(
                thread_id,
                "Thanks for your patience — I'm back with you. If anything else comes up, "
                "just say the word and I'll pick it up from here.",
            )
            told = " Customer told the agent is back."
        except Exception as e:
            logger.warning(f"[{thread_id}] Handback message failed (hold still released): {e}")
            told = " (Could not send the customer the handback message.)"

    return f"Released {thread_id}. The agent will use your note on the next message.{told}"
