"""
RAG node: for each paper, finds related prior work from both the local
library (built up across past runs) and live arXiv search, so the Skeptic
can check whether claims are corroborated or contradicted by literature
outside the papers given in this run - a step beyond cross_paper_conflicts,
which only compares papers within the current run against each other.

Also persists each processed paper's summary into the library, so future
runs benefit from what this run found.
"""
from ..cost_tracker import track_node
from ..rag import add_paper_to_library, query_library, search_related_arxiv


def rag_node(state: dict) -> dict:
    run_log = state.setdefault("run_log", [])
    papers = state["papers"]
    summaries = state["summaries"]
    summaries_by_id = {s.paper_id: s for s in summaries}
    all_ids = [p.paper_id for p in papers]

    rag_context = []
    with track_node("rag_lookup", run_log) as ctx:
        # No LLM call - embeddings run locally and arxiv search is a plain
        # HTTP call, so cost is $0, but latency is still worth tracking
        # (embedding model load + arxiv network round trips are not free
        # time-wise even though they're free dollar-wise).
        ctx["input_tokens"] = 0
        ctx["output_tokens"] = 0

        for paper in papers:
            summary = summaries_by_id.get(paper.paper_id)
            query_text = summary.summary if summary else paper.abstract

            rag_context.extend(query_library(query_text, for_paper_id=paper.paper_id, exclude_ids=all_ids))
            rag_context.extend(search_related_arxiv(paper))

            # Persist this run's paper into the library regardless of
            # whether we found matches for it - it becomes future runs'
            # "prior work" from here on.
            if summary:
                add_paper_to_library(paper, summary)

    return {"rag_context": rag_context, "run_log": run_log}
