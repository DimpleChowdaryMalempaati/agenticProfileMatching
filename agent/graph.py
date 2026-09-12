"""LangGraph workflow: parse → extract → search → rank → report → feedback."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    compare_node,
    error_node,
    explain_rank_node,
    extract_requirements_node,
    generate_report,
    human_feedback,
    interview_node,
    multi_round_node,
    parse_jd,
    rank_candidates,
    route_intent,
    search_resumes,
)
from agent.state import AgentState


def _after_route(state: AgentState) -> str:
    if state.get("error"):
        return "error"
    intent = state.get("intent") or "search"
    mapping = {
        "compare": "compare",
        "explain_rank": "explain_rank",
        "interview": "interview",
        "multi_round": "multi_round",
        "report": "generate_report",
        "feedback": "human_feedback",
        "refine": "parse_jd",
        "full_pipeline": "parse_jd",
        "search": "parse_jd",
        "general": "parse_jd",
    }
    return mapping.get(intent, "parse_jd")


def _after_parse(state: AgentState) -> str:
    return "error" if state.get("error") else "extract_requirements"


def _after_extract(state: AgentState) -> str:
    return "error" if state.get("error") else "search_resumes"


def _after_search(state: AgentState) -> str:
    return "error" if state.get("error") else "rank_candidates"


def _after_rank(state: AgentState) -> str:
    return "error" if state.get("error") else "generate_report"


def _after_feedback(state: AgentState) -> Literal["parse_jd", "compare", "explain_rank", "interview", "multi_round", "__end__"]:
    action = state.get("feedback_action") or "end"
    if action == "refine":
        intent = state.get("intent") or "refine"
        if intent == "compare":
            return "compare"
        if intent == "explain_rank":
            return "explain_rank"
        if intent == "interview":
            return "interview"
        if intent == "multi_round":
            return "multi_round"
        return "parse_jd"
    return END


def build_matching_graph() -> Any:
    """Compile the LangGraph matching agent.

    Workflow:
        START → route_intent → (branch)
            → parse_jd → extract_requirements → search_resumes → rank_candidates
              → generate_report → human_feedback → (refine loop | END)
            → compare | explain_rank | interview | multi_round | error → END-ish
    """
    graph = StateGraph(AgentState)

    graph.add_node("route_intent", route_intent)
    graph.add_node("parse_jd", parse_jd)
    graph.add_node("extract_requirements", extract_requirements_node)
    graph.add_node("search_resumes", search_resumes)
    graph.add_node("rank_candidates", rank_candidates)
    graph.add_node("generate_report", generate_report)
    graph.add_node("human_feedback", human_feedback)
    graph.add_node("compare", compare_node)
    graph.add_node("explain_rank", explain_rank_node)
    graph.add_node("interview", interview_node)
    graph.add_node("multi_round", multi_round_node)
    graph.add_node("error", error_node)

    graph.add_edge(START, "route_intent")
    graph.add_conditional_edges(
        "route_intent",
        _after_route,
        {
            "parse_jd": "parse_jd",
            "compare": "compare",
            "explain_rank": "explain_rank",
            "interview": "interview",
            "multi_round": "multi_round",
            "generate_report": "generate_report",
            "human_feedback": "human_feedback",
            "error": "error",
        },
    )
    graph.add_conditional_edges(
        "parse_jd",
        _after_parse,
        {"extract_requirements": "extract_requirements", "error": "error"},
    )
    graph.add_conditional_edges(
        "extract_requirements",
        _after_extract,
        {"search_resumes": "search_resumes", "error": "error"},
    )
    graph.add_conditional_edges(
        "search_resumes",
        _after_search,
        {"rank_candidates": "rank_candidates", "error": "error"},
    )
    graph.add_conditional_edges(
        "rank_candidates",
        _after_rank,
        {"generate_report": "generate_report", "error": "error"},
    )
    graph.add_edge("generate_report", END)
    graph.add_conditional_edges(
        "human_feedback",
        _after_feedback,
        {
            "parse_jd": "parse_jd",
            "compare": "compare",
            "explain_rank": "explain_rank",
            "interview": "interview",
            "multi_round": "multi_round",
            END: END,
        },
    )
    graph.add_edge("compare", END)
    graph.add_edge("explain_rank", END)
    graph.add_edge("interview", END)
    graph.add_edge("multi_round", END)
    graph.add_edge("error", END)

    return graph.compile()


# Module-level compiled graph for convenience
matching_graph = None


def get_graph():
    global matching_graph
    if matching_graph is None:
        matching_graph = build_matching_graph()
    return matching_graph
