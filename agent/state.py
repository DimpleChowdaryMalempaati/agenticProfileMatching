"""LangGraph agent state for conversational profile matching."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict


def _merge_lists(left: list, right: list) -> list:
    """Reducer that appends new items (used for reasoning trail)."""
    return list(left or []) + list(right or [])


class AgentState(TypedDict, total=False):
    """Full matching-agent state tracked across the graph.

    Tracks:
    - conversation history
    - job requirements understanding
    - candidate shortlist and reasoning
    - multi-round screening artifacts
    - human feedback / refinement loop
    """

    # Conversation
    messages: Annotated[list[dict[str, str]], operator.add]
    user_query: str
    intent: str

    # Job understanding
    jd_text: str
    jd_title: str
    requirements: dict[str, Any]
    refined_criteria: dict[str, Any]

    # Search / ranking
    candidates: list[dict[str, Any]]
    shortlist: list[dict[str, Any]]
    previous_shortlist: list[dict[str, Any]]
    ranking_changes: list[dict[str, Any]]
    reasoning: Annotated[list[str], _merge_lists]

    # Multi-round
    screening_round: int
    round1_shortlist: list[dict[str, Any]]
    round2_deep_analysis: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]

    # Outputs
    report: str
    comparison: dict[str, Any]
    interview_pack: dict[str, Any]
    assistant_response: str

    # Human feedback loop
    human_feedback: str
    awaiting_feedback: bool
    feedback_action: Literal["continue", "refine", "end", ""]

    # Control / errors
    error: str | None
    step: str


def initial_state(user_query: str = "") -> AgentState:
    """Create a clean agent state for a new conversation turn / run."""
    return AgentState(
        messages=[{"role": "user", "content": user_query}] if user_query else [],
        user_query=user_query,
        intent="",
        jd_text="",
        jd_title="",
        requirements={},
        refined_criteria={},
        candidates=[],
        shortlist=[],
        previous_shortlist=[],
        ranking_changes=[],
        reasoning=[],
        screening_round=0,
        round1_shortlist=[],
        round2_deep_analysis=[],
        recommendations=[],
        report="",
        comparison={},
        interview_pack={},
        assistant_response="",
        human_feedback="",
        awaiting_feedback=False,
        feedback_action="",
        error=None,
        step="start",
    )
