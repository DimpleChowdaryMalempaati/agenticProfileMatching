"""RAG search tool wrapper for the matching agent."""

from __future__ import annotations

from typing import Any

from config import TOP_K
from core.exceptions import error_response, success_response
from core.resume_rag import ResumeRAG

_rag: ResumeRAG | None = None


def get_rag() -> ResumeRAG:
    """Lazy singleton RAG instance."""
    global _rag
    if _rag is None:
        _rag = ResumeRAG()
    return _rag


def rag_search(query: str, top_k: int = TOP_K) -> dict[str, Any]:
    """Semantic search over indexed resumes.

    Args:
        query: Natural language search query (skills, role, constraints).
        top_k: Max chunks to retrieve.

    Returns:
        Ranked resume chunk hits with similarity and candidate metadata.
    """
    tool = "rag_search"
    try:
        if not query or not str(query).strip():
            return error_response("Query is empty", "EMPTY_QUERY", tool=tool)

        rag = get_rag()
        stats = rag.get_stats()
        if stats.get("chunk_count", 0) == 0:
            return error_response(
                "Resume index is empty. Run: python scripts/setup_index.py",
                "INDEX_NOT_READY",
                tool=tool,
            )

        result = rag.query(str(query).strip(), top_k=int(top_k))
        if not result.get("success"):
            err = result.get("error", {})
            return error_response(
                err.get("message", "RAG query failed"),
                err.get("code", "QUERY_FAILED"),
                tool=tool,
            )

        # Aggregate to candidate-level shortlist for agent convenience
        by_candidate: dict[str, dict[str, Any]] = {}
        for hit in result.get("results", []):
            meta = hit.get("metadata", {})
            name = meta.get("candidate_name", "Unknown")
            path = meta.get("resume_path", "")
            key = path or name
            if key not in by_candidate:
                by_candidate[key] = {
                    "candidate_name": name,
                    "resume_path": path,
                    "best_similarity": hit.get("similarity", 0),
                    "sections": [],
                    "excerpts": [],
                }
            entry = by_candidate[key]
            entry["best_similarity"] = max(
                entry["best_similarity"], hit.get("similarity", 0)
            )
            section = meta.get("section")
            if section:
                entry["sections"].append(section)
            if hit.get("text"):
                entry["excerpts"].append(hit["text"][:250])

        candidates = sorted(
            by_candidate.values(),
            key=lambda c: c["best_similarity"],
            reverse=True,
        )

        return success_response(
            tool,
            query=query,
            chunk_hits=result.get("results", []),
            candidates=candidates,
            count=len(candidates),
            latency_seconds=result.get("latency_seconds"),
        )
    except Exception as exc:  # noqa: BLE001
        return error_response(f"rag_search failed: {exc}", "UNEXPECTED", tool=tool)
