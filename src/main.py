"""
CLI entry point.

Usage:
    python -m src.main --inputs 2306.07691 path/to/paper.pdf --human-in-loop
    python -m src.main --inputs 2306.07691 --no-human-in-loop

If --human-in-loop is set and the graph pauses, this CLI prompts you right
in the terminal to approve/reject - useful for testing without the
Streamlit dashboard. In practice you'd more likely drive the paused state
from the dashboard (see dashboard/app.py), which is why review_node's
`interrupt()` payload is plain JSON-serializable data, not tied to any one
frontend.
"""
import argparse
import json
import uuid

from .graph import build_graph
from langgraph.types import Command


def run(input_sources: list[str], human_in_loop: bool, audio_path: str = None):
    app = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = app.invoke(
            {"input_sources": input_sources, "human_in_loop": human_in_loop}, config=config
        )
    except ValueError as e:
        print(f"\n❌ {e}")
        print("If this was an arXiv rate limit (HTTP 429), wait a minute or two and try again - "
              "arXiv throttles repeated requests from the same IP.")
        return None

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\n=== HUMAN REVIEW REQUIRED ===")
        for line in payload["draft_script"]:
            print(f"[{line['speaker']}] {line['text']}")
        print("\nGrounding report:", json.dumps(payload["grounding_report"], indent=2))
        choice = input("\nApprove as-is? [y]/n: ").strip().lower()
        resume_value = {"approved": choice != "n"}
        result = app.invoke(Command(resume=resume_value), config=config)

    print("\n=== FINAL SCRIPT ===")
    for line in result["final_script"].lines:
        refs = f" (refs: {', '.join(line.cites_paper_ids)})" if line.cites_paper_ids else ""
        print(f"[{line.speaker}] {line.text}{refs}")

    total_cost = sum(r.estimated_cost_usd for r in result["run_log"])
    total_time = sum(r.duration_s for r in result["run_log"])
    print(f"\n=== COST/LATENCY ===\nTotal estimated cost: ${total_cost:.4f}  |  Total time: {total_time:.2f}s")
    print(f"Grounding rate: {result['grounding_report']['grounding_rate']:.0%}")

    if audio_path:
        print(f"\n=== GENERATING AUDIO ===\nRendering to {audio_path} (this can take a while on first run "
              "while Kokoro downloads its model)...")
        from .tts import render_script_to_audio
        try:
            render_script_to_audio(result["final_script"], audio_path)
            print(f"Done: {audio_path}")
        except Exception as e:
            print(f"Audio generation failed: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="ArXiv Podcast Debate Generator")
    parser.add_argument("--inputs", nargs="+", required=True, help="1-3 arXiv IDs or local PDF file paths")
    parser.add_argument("--human-in-loop", dest="hil", action="store_true", default=False)
    parser.add_argument("--no-human-in-loop", dest="hil", action="store_false")
    parser.add_argument("--audio", dest="audio_path", default=None,
                         help="If set, render the final script to this .wav path using Kokoro TTS "
                              "(e.g. --audio outputs/episode.wav)")
    args = parser.parse_args()
    run(args.inputs, args.hil, args.audio_path)


if __name__ == "__main__":
    main()
