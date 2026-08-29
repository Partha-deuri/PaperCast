"""
The Skeptic agent: reads all paper summaries at once (not one at a time,
unlike the Academic) specifically so it can compare across papers when 2-3
were provided - looking for contradictions, not just per-paper flaws.

It's also told:
  - which citations failed grounding, so a hallucinated quote becomes a
    critique point in its own right ("the summary cites a result that
    doesn't appear in the paper").
  - related prior work found via RAG (local library + live arXiv search),
    so it can flag when a paper's claims are contradicted or already
    well-supported by literature outside the papers given in this run -
    not just checking the input papers against each other.
"""
from ..models import PaperSummary, SkepticCritique
from ..llm_client import call_structured
from ..cost_tracker import track_node

def _get_system_prompt(domains: set) -> str:
    domain_instruction = ""
    if domains:
        domains_str = ", ".join(domains)
        domain_instruction = f"The papers are from the following domains: {domains_str}. Tailor your critique to the specific methodological pitfalls common in these fields (e.g., sample sizes and p-hacking for biology/medicine, compute constraints and unproven assumptions for CS/AI, rigorous derivations for physics/math)."
        
    return f"""You are "The Skeptic" - a rigorous peer reviewer. You are given summaries of \
1-3 papers (with their claims and citations), a grounding report showing which citations could \
NOT be verified against the source text, and a list of related prior work found via retrieval \
(RAG) - some from a library of papers this tool has processed before, some from a live arXiv \
search for related work.

{domain_instruction}

For each paper, list 1-3 concrete methodological flaws (small sample sizes, missing baselines, \
unproven assumptions, weak evaluation, etc). If a paper has ungrounded/hallucinated citations \
per the grounding report, call that out explicitly as a flaw for that paper.

If more than one input paper was provided, also list any contradictions or tensions between the \
papers' claims in cross_paper_conflicts. If only one input paper was provided, leave that list empty.

Separately, using the related prior work list, write 1-3 notes in external_literature_notes on \
how each paper's claims relate to that outside literature - e.g. "Paper X's claim of Y is \
consistent with [related paper title]" or "no related prior work was found addressing Z, which \
itself may be worth noting". If no related work was found at all, leave this list empty rather \
than inventing connections.
"""


def _mock_critique(summaries: list, rag_context: list) -> SkepticCritique:
    flaws = {s.paper_id: [f"[MOCK] Potential flaw in {s.paper_id}'s methodology"] for s in summaries}
    conflicts = (
        [f"[MOCK] Tension between {summaries[0].paper_id} and {summaries[1].paper_id}"]
        if len(summaries) > 1
        else []
    )
    lit_notes = (
        [f"[MOCK] Related work found: {rag_context[0].title}"] if rag_context else []
    )
    return SkepticCritique(per_paper_flaws=flaws, cross_paper_conflicts=conflicts, external_literature_notes=lit_notes)


def skeptic_node(state: dict) -> dict:
    run_log = state.setdefault("run_log", [])
    summaries: list[PaperSummary] = state["summaries"]
    papers = state.get("papers", [])
    grounding_report = state.get("grounding_report", {})
    rag_context = state.get("rag_context", [])
    
    domains = {p.domain for p in papers if getattr(p, "domain", None) and p.domain != "Unknown"}

    summaries_text = "\n\n".join(
        f"Paper {s.paper_id}:\nSummary: {s.summary}\nClaims: {s.key_claims}\n"
        f"Citations: {[c.quote for c in s.citations]}"
        for s in summaries
    )
    flagged = grounding_report.get("flagged", [])
    flagged_text = (
        "\n".join(f"- {f['paper_id']}: \"{f['quote']}\" (match_score={f['match_score']})" for f in flagged)
        or "None - all citations verified."
    )
    rag_text = (
        "\n".join(
            f"- [{m.source}] related to {m.for_paper_id}: \"{m.title}\" - {m.snippet}"
            for m in rag_context
        )
        or "None found."
    )

    with track_node("skeptic_agent", run_log) as ctx:
        user_prompt = (
            f"{summaries_text}\n\n"
            f"--- Ungrounded/unverified citations flagged by the grounding check ---\n{flagged_text}\n\n"
            f"--- Related prior work found via RAG ---\n{rag_text}"
        )
        response = call_structured(
            _get_system_prompt(domains),
            user_prompt,
            SkepticCritique,
            mock_factory=lambda: _mock_critique(summaries, rag_context),
        )
        ctx["input_tokens"] = response.input_tokens
        ctx["output_tokens"] = response.output_tokens
        critique = response.parsed

    return {"critique": critique, "run_log": run_log}
