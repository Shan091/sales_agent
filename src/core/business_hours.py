# file: src/core/business_hours.py
"""
Otohom sales/support availability clock.

Single source of truth for "is the team online right now?" and the callback expectation
we set with the customer at a handoff moment. Facts come from the client's intake answers
(docs/client/otohom_client_answers.md):

- Hours: Monday–Saturday, 9 AM–5 PM IST (Sunday closed).
- Callback SLA: within business hours -> within ~2 working hours; outside -> next working
  day before 11 AM.

IST has no daylight saving, so a fixed UTC+5:30 offset is exact and needs no tzdata package
(keeps this working on a bare Windows checkout). Constants live here, mirroring the
OFFICE_HOURS constant in src/logic/prompts.py; both can move into config/settings.py later.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

# Fixed IST offset (no DST). Explicit offset avoids a hard dependency on the tzdata package.
IST = timezone(timedelta(hours=5, minutes=30))

# Mon–Sat, 09:00–17:00. Python weekday(): Monday=0 ... Sunday=6, so 0..5 are working days.
OPEN_HOUR = 9
CLOSE_HOUR = 17
WORKING_WEEKDAYS = range(0, 6)  # Monday..Saturday


def is_within_business_hours(now: Optional[datetime] = None) -> bool:
    """
    True if `now` falls inside Otohom's Mon–Sat 9 AM–5 PM IST window.

    `now` may be any timezone-aware or naive datetime; it is converted to IST first. Pass a
    fixed datetime in tests — never rely on the wall clock there.
    """
    if now is None:
        now = datetime.now(IST)
    elif now.tzinfo is None:
        # Assume a naive datetime is already IST (tests pass IST-local times).
        now = now.replace(tzinfo=IST)
    now = now.astimezone(IST)

    if now.weekday() not in WORKING_WEEKDAYS:
        return False
    return OPEN_HOUR <= now.hour < CLOSE_HOUR


def business_status_line(now: Optional[datetime] = None) -> str:
    """
    The guidance string injected into sales prompts as {business_status}. It tells the agent
    the correct callback expectation to set — in its OWN warm words — when it reaches a
    "the team will get back to you" moment. Not a canned customer-facing line.
    """
    if is_within_business_hours(now):
        return (
            "AVAILABILITY: The Otohom team is ONLINE right now (Mon–Sat, 9 AM–5 PM IST). "
            "When you tell the customer the team will follow up, set the expectation warmly in "
            "your own words: they'll typically hear back within about 2 working hours. Do not "
            "promise an exact minute."
        )
    return (
        "AVAILABILITY: The Otohom team is currently OFFLINE (outside Mon–Sat, 9 AM–5 PM IST). "
        "When you tell the customer the team will follow up, gently set the expectation in your "
        "own words: their enquiry is received and the team will reach out the next working day "
        "(before 11 AM). Stay warm and reassuring — never leave them feeling ignored."
    )
