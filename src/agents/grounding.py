"""
Grounding node: not an LLM call at all - just verification logic run against
every citation the Academic agent produced across all papers. This sits
between the Academic and Skeptic agents so the Skeptic can be told which
claims are actually backed by the source text and which are not.
"""
from ..citation_grounding import verify_citations, grounding_summary
from ..cost_tracker import track_node


def grounding_node(state: dict) -> dict:
    run_log = state.setdefault("run_log", [])
    papers_by_id = {p.paper_id: p for p in state["papers"]}
    all_citations = [c for s in state["summaries"] for c in s.citations]

    with track_node("citation_grounding", run_log) as ctx:
        # No LLM call here, so cost is $0, but we still record latency -
        # useful to show this step isn't free (fuzzy matching does scale
        # with paper length) even though it costs no tokens.
        verified = verify_citations(all_citations, papers_by_id)
        ctx["input_tokens"] = 0
        ctx["output_tokens"] = 0

    summary = grounding_summary(verified)
    return {"verified_citations": verified, "run_log": run_log, "grounding_report": summary}
