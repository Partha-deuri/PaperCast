"""
RAG layer: cross-references each input paper's claims against prior work
from two sources:

  1. A persistent local library (Chroma, on disk at config.CHROMA_DIR) of
     every paper this tool has ever processed - grows across runs. This is
     what lets the Skeptic eventually say "you claimed X, but a paper we
     processed three runs ago found the opposite."
  2. Live arXiv search for papers related to each input paper's title -
     catches prior work this tool has never seen before, not just its own
     accumulated library.

Embeddings run fully locally via sentence-transformers (free, no API key,
no rate limit) - only the arXiv *search* calls need network, and those
share ingestion.py's rate-limited client.

Every function here is best-effort: Chroma, sentence-transformers, or arXiv
search failing for any reason (missing deps, no internet on first run before
the embedding model is cached, arXiv rate limit) logs a warning and returns
an empty result rather than raising - RAG is a pure enhancement layer, never
a hard dependency for getting a script out of the pipeline.
"""
import logging
from typing import List, Optional

from .models import PaperData, PaperSummary, RAGMatch
from . import config

logger = logging.getLogger(__name__)

_collection = None  # lazily-initialized Chroma collection, cached at module level


def _get_collection():
    """Returns the persistent Chroma collection, creating it (and loading
    the embedding model) on first call. Cached afterward so repeated calls
    within one run don't reload the model."""
    global _collection
    if _collection is not None:
        return _collection

    import chromadb
    from chromadb.utils import embedding_functions

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    _collection = client.get_or_create_collection(
        name="paper_library", embedding_function=embed_fn
    )
    return _collection


def add_paper_to_library(paper: PaperData, summary: PaperSummary) -> None:
    """Persist a processed paper into the local library so future runs can
    find it as related work. Safe to call repeatedly - upsert overwrites."""
    try:
        collection = _get_collection()
        doc_text = (
            f"{paper.title}\n\n{summary.summary}\n\n"
            f"Key claims: {'; '.join(summary.key_claims)}"
        )
        collection.upsert(
            ids=[paper.paper_id],
            documents=[doc_text],
            metadatas=[{"title": paper.title, "paper_id": paper.paper_id}],
        )
    except Exception as e:
        logger.warning(f"Failed to add {paper.paper_id} to RAG library: {e}")

def list_library() -> List[dict]:
    """Returns a list of all papers currently in the Chroma library."""
    try:
        collection = _get_collection()
        results = collection.get(include=["metadatas"])
        if not results or not results.get("metadatas"):
            return []
        
        papers = []
        for meta, doc_id in zip(results["metadatas"], results["ids"]):
            papers.append({
                "id": doc_id,
                "title": meta.get("title", doc_id)
            })
        return papers
    except Exception as e:
        logger.warning(f"Failed to list RAG library: {e}")
        return []

def delete_from_library(paper_id: str) -> bool:
    """Deletes a specific paper from the local library."""
    try:
        collection = _get_collection()
        collection.delete(ids=[paper_id])
        return True
    except Exception as e:
        logger.warning(f"Failed to delete {paper_id} from RAG library: {e}")
        return False

def clear_library() -> bool:
    """Deletes all papers from the local library."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        client.delete_collection("paper_library")
        global _collection
        _collection = None
        return True
    except Exception as e:
        logger.warning(f"Failed to clear RAG library: {e}")
        return False


def query_library(query_text: str, for_paper_id: str, exclude_ids: List[str], top_k: int = None) -> List[RAGMatch]:
    """Semantic search over the local library, excluding the papers in the
    current run (we want *prior* work, not the input set matching itself)."""
    top_k = top_k or config.RAG_TOP_K
    try:
        collection = _get_collection()
        # over-fetch a bit since we filter out excluded ids after the fact
        results = collection.query(query_texts=[query_text], n_results=top_k + len(exclude_ids))
    except Exception as e:
        logger.warning(f"RAG library query failed: {e}")
        return []

    matches = []
    ids = results.get("ids", [[]])[0]
    if not ids:
        return matches
    metadatas = results["metadatas"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]

    for doc_id, meta, doc, dist in zip(ids, metadatas, documents, distances):
        if doc_id in exclude_ids:
            continue
        matches.append(
            RAGMatch(
                paper_id=doc_id,
                title=meta.get("title", doc_id),
                snippet=doc[:300],
                source="library",
                for_paper_id=for_paper_id,
                similarity=round(max(0.0, 1 - dist), 3),  # Chroma default = cosine distance
            )
        )
        if len(matches) >= top_k:
            break
    return matches


def search_related_arxiv(paper: PaperData, top_k: int = None) -> List[RAGMatch]:
    """Live arXiv search for papers related to this one. Best-effort: any
    failure (network, rate limit) returns [] rather than raising, so a
    flaky arXiv search never breaks the pipeline."""
    top_k = top_k or config.RAG_TOP_K
    try:
        import arxiv
        from .ingestion import ARXIV_CLIENT

        search = arxiv.Search(
            query=paper.title, max_results=top_k + 1, sort_by=arxiv.SortCriterion.Relevance
        )
        matches = []
        this_id = paper.paper_id.split("v")[0] if "v" in paper.paper_id else paper.paper_id
        for result in ARXIV_CLIENT.results(search):
            if result.get_short_id().split("v")[0] == this_id:
                continue  # skip the paper itself
            matches.append(
                RAGMatch(
                    paper_id=result.get_short_id(),
                    title=result.title,
                    snippet=result.summary[:300].replace("\n", " "),
                    source="arxiv_search",
                    for_paper_id=paper.paper_id,
                    similarity=None,
                )
            )
            if len(matches) >= top_k:
                break
        return matches
    except Exception as e:
        logger.warning(f"Related-paper arXiv search failed for {paper.paper_id}: {e}")
        return []
