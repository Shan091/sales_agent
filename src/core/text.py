"""Small text helpers shared by the schema layer and the WhatsApp transport."""


def fit_label(text: str, limit: int) -> str:
    """
    Trim a label to `limit` characters at a WORD boundary.

    A blind slice is what produces "Smart switches & dim" and "Show me a typical" — the
    customer is asked to choose between fragments. Dropping the whole trailing word keeps
    every option readable, and an ellipsis marks the ones that were genuinely too long so the
    truncation reads as deliberate rather than broken.

    Used in two places on purpose: the schema shortens instead of REJECTING an over-long
    label (a cosmetic field must never invalidate a whole reply — including a valid order
    proposal — and force a retry the customer waits through), and the transport applies it
    again as the last guarantee before the payload reaches WhatsApp.
    """
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    clipped = text[:limit].rstrip()
    if " " in clipped:
        trimmed = clipped.rsplit(" ", 1)[0].rstrip(" ,-–—&/")
        # Only prefer the word boundary if it leaves something meaningful behind.
        if len(trimmed) >= limit // 2:
            clipped = trimmed
    if len(clipped) <= limit - 1:
        return clipped + "…"
    return clipped[: limit - 1].rstrip(" ,-–—&/") + "…"


def truncate_words(text: str, limit: int) -> str:
    """
    Shorten prose to roughly `limit` characters at a word boundary, collapsing whitespace.

    Separate from `fit_label` because the two have different jobs at different scales.
    `fit_label` fights for 20 characters on a button, where losing half a word is fatal and the
    ellipsis is a promise that the label was cut. This is for quoting a customer back to a
    salesperson in a spreadsheet cell: newlines have to go (they would break the row), the limit
    is generous, and an ellipsis is just punctuation.
    """
    flat = " ".join((text or "").split())
    if len(flat) <= limit:
        return flat
    clipped = flat[:limit].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0].rstrip(" ,.;:-–—&/")
    return clipped + "…"


def dedupe_keeping_first(values: list, limit: int = 8) -> list:
    """
    Drop repeats case-insensitively, keep the first spelling and the original order, cap the list.

    The lead sheet's pain-points cell read "front door safety; Front door safety; Front door safety"
    because every turn appended the model's extraction unconditionally, and the model varies the
    casing — so a set of exact strings never collapsed them. The cap is because a twenty-turn
    conversation otherwise writes a paragraph into one spreadsheet cell.

    Lives here rather than beside either caller so the node that appends and the digest that renders
    cannot disagree about what the list contains.
    """
    seen = set()
    out = []
    for value in values or []:
        text = (value or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out
