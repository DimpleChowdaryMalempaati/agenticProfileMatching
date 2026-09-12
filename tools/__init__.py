"""Agent-callable tools: filesystem, RAG, requirements, compare, interview.

Import submodules directly (e.g. ``from tools.rag_search import rag_search``)
to avoid circular imports with ``core``.
"""

from __future__ import annotations

from typing import Any, Callable


def get_tool_dispatch() -> dict[str, Callable[..., Any]]:
    """Lazy tool registry (avoids circular imports at package import time)."""
    from tools.compare_candidates import compare_candidates
    from tools.extract_requirements import extract_requirements
    from tools.fs_tools import TOOL_DISPATCH as FS_TOOL_DISPATCH
    from tools.interview_questions import generate_interview_questions
    from tools.rag_search import rag_search
    from tools.resume_loader import list_resumes, load_resume

    return {
        **FS_TOOL_DISPATCH,
        "rag_search": rag_search,
        "extract_requirements": extract_requirements,
        "compare_candidates": compare_candidates,
        "generate_interview_questions": generate_interview_questions,
        "list_resumes": list_resumes,
        "load_resume": load_resume,
    }


def execute_agent_tool(name: str, arguments: dict) -> object:
    """Dispatch any agent tool by name."""
    dispatch = get_tool_dispatch()
    if name not in dispatch:
        return {"success": False, "error": f"Unknown tool: {name}"}
    return dispatch[name](**arguments)


__all__ = [
    "execute_agent_tool",
    "get_tool_dispatch",
]
