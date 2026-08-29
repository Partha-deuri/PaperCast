"""
Citation grounding: verify that quotes the Academic agent attributes to a
paper actually appear (or closely paraphrase something that appears) in that
paper's extracted text.

This is the anti-hallucination layer. LLMs are very good at generating
plausible-sounding quotes that were never actually said. We don't want to
just trust the model - we check.

Approach: exact substring match first (cheap, catches verbatim quotes), then
fuzzy sequence matching as a fallback for near-verbatim quotes (whitespace/
punctuation differences), reporting a match_score so downstream code can
decide a threshold.
"""
import re
import difflib
from typing import List, Dict
from .models import Citation, VerifiedCitation, PaperData

FUZZY_MATCH_THRESHOLD = 0.75


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _best_fuzzy_match_score(quote: str, source: str, window_slack: int = 20) -> float:
    """Slide a window roughly the size of the quote across the source text
    and return the best similarity ratio found. This is O(n) windows but
    papers are capped at MAX_FULLTEXT_CHARS_PER_PAPER so it stays fast."""
    norm_quote = _normalize(quote)
    norm_source = _normalize(source)

    if not norm_quote:
        return 0.0
    if norm_quote in norm_source:
        return 1.0

    q_len = len(norm_quote)
    best = 0.0
    step = max(q_len // 4, 10)
    for start in range(0, max(len(norm_source) - q_len, 1) + 1, step):
        window = norm_source[start:start + q_len + window_slack]
        ratio = difflib.SequenceMatcher(None, norm_quote, window).ratio()
        if ratio > best:
            best = ratio
    return best


def verify_citations(
    citations: List[Citation], papers_by_id: Dict[str, PaperData]
) -> List[VerifiedCitation]:
    """Check each citation against its claimed source paper's full_text."""
    verified = []
    for c in citations:
        source_paper = papers_by_id.get(c.paper_id)
        if source_paper is None:
            # Agent cited a paper_id that doesn't exist in our set at all -
            # definitely flag this, it's a worse failure than a bad quote.
            verified.append(
                VerifiedCitation(**c.model_dump(), verified=False, match_score=0.0)
            )
            continue

        score = _best_fuzzy_match_score(c.quote, source_paper.full_text)
        verified.append(
            VerifiedCitation(
                **c.model_dump(),
                verified=score >= FUZZY_MATCH_THRESHOLD,
                match_score=round(score, 3),
            )
        )
    return verified


def grounding_summary(verified: List[VerifiedCitation]) -> dict:
    """Small report used by the dashboard and by the Skeptic agent (an
    ungrounded citation is itself a flaw worth surfacing in the debate)."""
    total = len(verified)
    passed = sum(1 for v in verified if v.verified)
    return {
        "total_citations": total,
        "verified_citations": passed,
        "hallucinated_citations": total - passed,
        "grounding_rate": round(passed / total, 3) if total else 1.0,
        "flagged": [
            {"paper_id": v.paper_id, "quote": v.quote, "match_score": v.match_score}
            for v in verified
            if not v.verified
        ],
    }
