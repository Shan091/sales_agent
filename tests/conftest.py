"""
Shared pytest fixtures and path setup for the Otohom test suite.

Imports across the project are absolute (`src.*`, `config.*`) with no packaging shim,
so we make sure the repo root is importable regardless of how pytest is invoked.
`pytest.ini` sets `asyncio_mode = auto`, so async tests need no decorator or event-loop fixture.
"""
import sys
from pathlib import Path

import pytest

# Ensure the repo root (parent of tests/) is on sys.path so `import src...` / `import config...`
# resolve even when pytest is launched from an unexpected working directory.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def make_state():
    """Factory for a minimal ConversationState-shaped dict.

    Graph routing and node helpers only ever `.get()` keys, so a plain dict is enough.
    Pass keyword overrides to set the fields a given test cares about.
    """
    def _make(**overrides):
        state = {
            "messages": [],
            "language_preference": "English",
            "current_archetype": "GENERAL_GREETING",
            "requires_human_handoff": False,
            "data_routing_flag": "NONE",
            "human_request_count": 0,
            "primary_interest": None,
            "context_chunks": [],
            "rag_query": None,
            "lead_ready_for_handoff": False,
            "lead_sent": False,
        }
        state.update(overrides)
        return state

    return _make
