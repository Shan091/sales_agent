"""
Local terminal chat harness — talk to the sales agent WITHOUT Meta/WhatsApp.

This drives the compiled LangGraph workflow directly (the exact same call the
TaskIQ worker makes in `src/tasks/processing.py`), but reads your messages from
stdin and prints the agent's replies to the terminal instead of dispatching them
to the Meta Cloud API.

What it needs:
  - OPENAI_API_KEY (OpenRouter) in your .env  ......... REQUIRED (the LLM brain)
  - Postgres + pgvector, with an ingested catalog ..... ONLY for technical/spec
    questions that route to TECHNICAL_RAG. Greetings, sales, B2B, etc. work with
    just the LLM key.
  - Redis / the webhook / the worker .................. NOT used (bypassed).

Multi-turn memory works in-process via the graph's MemorySaver checkpointer, so a
whole conversation is remembered for the life of this process (per --thread-id).

Usage (from the repo root):
    python -m src.scripts.local_chat
    python -m src.scripts.local_chat --thread-id alice

Type your message and press Enter. Type 'exit' or 'quit' (or Ctrl-C) to leave.
"""
import argparse
import asyncio
import logging

from config.settings import settings
from src.core.guardrails import Guardrails
from src.graph.workflow import compile_workflow

# Keep the console readable — surface warnings/errors from the nodes, not INFO noise.
logging.basicConfig(level=logging.WARNING)


def _print_options(options):
    """Render WhatsApp button/list options as a plain numbered list in the terminal."""
    if not options:
        return
    print("    [ options the agent would show as buttons/list: ]")
    for i, opt in enumerate(options, 1):
        label = opt.get("label", "")
        postback = opt.get("postback_id", "")
        print(f"      {i}. {label}  (id={postback})")


async def _run_turn(graph_app, thread_id: str, text: str) -> None:
    """Stream one user turn through the graph and print every AI message it emits."""
    # Mirror the worker: sanitize input before it reaches the graph.
    text = Guardrails.sanitize_input(text)
    config = {"configurable": {"thread_id": thread_id}}

    printed_any = False
    async for output in graph_app.astream(
        {"messages": [("user", text)]}, config=config, stream_mode="updates"
    ):
        for node_name, state_updates in output.items():
            for m in (state_updates or {}).get("messages", []):
                if getattr(m, "type", None) == "ai":
                    printed_any = True
                    print(f"\n  agent ({node_name}): {m.content}")
                    meta = getattr(m, "response_metadata", None) or {}
                    _print_options(meta.get("options"))

    if not printed_any:
        print("\n  (no AI message was emitted this turn — check WARNING logs above)")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Local terminal chat with the Otohom sales agent (no Meta).")
    parser.add_argument("--thread-id", default="local_test_user", help="Conversation thread id (state is per thread).")
    args = parser.parse_args()

    key = (settings.OPENAI_API_KEY or "").strip()
    if not key or key.startswith("your_"):
        print(
            "ERROR: OPENAI_API_KEY is not set (or is still a placeholder) in your .env.\n"
            "       The agent cannot generate replies without an OpenRouter key.\n"
            "       Set OPENAI_API_KEY=... in .env and try again."
        )
        return

    print("Compiling workflow (MemorySaver, no Meta, no worker)...")
    graph_app = compile_workflow()
    print(
        f"Ready. Chatting as thread '{args.thread_id}'. "
        "Type your message; 'exit' or Ctrl-C to quit.\n"
        "Note: technical/spec questions need Postgres+pgvector with an ingested catalog.\n"
    )

    while True:
        try:
            user_text = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("Bye.")
            return
        try:
            await _run_turn(graph_app, args.thread_id, user_text)
        except Exception as e:  # noqa: BLE001 — a bad turn shouldn't kill the REPL
            print(f"\n  [turn failed: {type(e).__name__}: {e}]")
            print("  (technical questions need Postgres+pgvector + an ingested catalog running.)")
        print()


if __name__ == "__main__":
    asyncio.run(main())
