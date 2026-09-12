"""Shared exception hierarchy with stable error codes."""

from __future__ import annotations

from typing import Any


class AgentError(Exception):
    """Base error for the matching agent stack."""

    def __init__(self, message: str, code: str = "AGENT_ERROR", details: Any = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "success": False,
            "error": {"message": self.message, "code": self.code},
        }
        if self.details is not None:
            payload["error"]["details"] = self.details
        return payload


class ValidationError(AgentError):
    def __init__(self, message: str, details: Any = None):
        super().__init__(message, code="VALIDATION_ERROR", details=details)


class ToolExecutionError(AgentError):
    def __init__(self, message: str, tool: str | None = None, details: Any = None):
        super().__init__(message, code="TOOL_ERROR", details=details)
        self.tool = tool


class IndexNotReadyError(AgentError):
    def __init__(self, message: str = "Resume index is empty. Run scripts/setup_index.py first."):
        super().__init__(message, code="INDEX_NOT_READY")


class CandidateNotFoundError(AgentError):
    def __init__(self, candidate_id: str):
        super().__init__(
            f"Candidate not found: {candidate_id}",
            code="CANDIDATE_NOT_FOUND",
            details={"candidate_id": candidate_id},
        )


def error_response(
    message: str,
    code: str = "ERROR",
    *,
    tool: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Standard failure envelope used by tools and nodes."""
    payload: dict[str, Any] = {
        "success": False,
        "error": {"message": message, "code": code},
        **extra,
    }
    if tool:
        payload["tool"] = tool
    return payload


def success_response(tool: str | None = None, **payload: Any) -> dict[str, Any]:
    """Standard success envelope used by tools and nodes."""
    result: dict[str, Any] = {"success": True, **payload}
    if tool:
        result["tool"] = tool
    return result
