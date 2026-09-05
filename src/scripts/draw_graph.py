# file: src/scripts/draw_graph.py
"""
Export the LangGraph conversation graph to disk.

Compiles WITHOUT a checkpointer (no DB/Redis needed — pure topology)
and writes two artifacts next to the repo root:
  - graph.mmd  : Mermaid source (always works, no extra deps)
  - graph.png  : PNG render (needs network for the mermaid.ink API,
                 or a local `pyppeteer`/`graphviz` backend)

Run from repo root:
    python -m src.scripts.draw_graph
    python -m src.scripts.draw_graph --out-dir ./docs
"""
import argparse
from pathlib import Path

from src.graph.workflow import create_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the LangGraph graph.")
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory to write graph.mmd / graph.png (default: repo root).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compile without a checkpointer — we only need the topology, not persistence.
    compiled = create_workflow().compile()
    graph = compiled.get_graph()

    # 1. Mermaid source — dependency-free, always succeeds.
    mmd_path = out_dir / "graph.mmd"
    mmd_path.write_text(graph.draw_mermaid(), encoding="utf-8")
    print(f"[ok] Mermaid source -> {mmd_path}")

    # 2. PNG — best effort (needs network or a local render backend).
    png_path = out_dir / "graph.png"
    try:
        png_path.write_bytes(graph.draw_mermaid_png())
        print(f"[ok] PNG          -> {png_path}")
    except Exception as exc:  # noqa: BLE001 — render backend is optional
        print(f"[skip] PNG render failed ({exc.__class__.__name__}: {exc}).")
        print(f"       Use the .mmd source at {mmd_path} instead.")


if __name__ == "__main__":
    main()