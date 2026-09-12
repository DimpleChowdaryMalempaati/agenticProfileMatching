"""
matching_agent.py — LangGraph-based Agentic Profile Matching System

Assignment workflow:
    START → Parse JD → Extract Requirements → Search Resumes →
    Rank Candidates → Generate Report → Human Feedback Loop → END

Tools available to the agent:
    - Filesystem tools (Milestone 1): read_file, list_files, write_file, search_in_file
    - RAG search tool (Milestone 2): rag_search
    - extract_requirements(jd)
    - compare_candidates(candidate_ids)
    - generate_interview_questions(candidate_id)
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from agent.graph import build_matching_graph, get_graph
from agent.state import AgentState, initial_state
from screening.multi_round import run_multi_round_screening
from tools import execute_agent_tool, get_tool_dispatch
from tools.compare_candidates import compare_candidates
from tools.extract_requirements import extract_requirements
from tools.interview_questions import generate_interview_questions
from tools.rag_search import rag_search

AGENT_TOOL_DISPATCH = get_tool_dispatch()


class MatchingAgent:
    """Conversational matching agent backed by a LangGraph state machine."""

    def __init__(self):
        self.graph = build_matching_graph()
        self.state: AgentState = initial_state()

    def reset(self) -> None:
        self.state = initial_state()

    def run(self, user_query: str, *, human_feedback: str | None = None) -> AgentState:
        """Execute one agent turn.

        Args:
            user_query: Natural language request from the recruiter.
            human_feedback: Optional feedback for the refinement loop.
        """
        # Carry forward durable session fields
        carried = {
            "shortlist": self.state.get("shortlist") or [],
            "previous_shortlist": self.state.get("shortlist") or [],
            "requirements": self.state.get("requirements") or {},
            "jd_text": self.state.get("jd_text") or "",
            "jd_title": self.state.get("jd_title") or "",
            "refined_criteria": self.state.get("refined_criteria") or {},
            "recommendations": self.state.get("recommendations") or [],
            "report": self.state.get("report") or "",
            "awaiting_feedback": self.state.get("awaiting_feedback", False),
            "reasoning": [],  # fresh trail per turn; prior kept in messages
            "messages": list(self.state.get("messages") or []),
        }

        turn = initial_state(user_query)
        turn.update(carried)
        turn["user_query"] = user_query
        turn["messages"] = carried["messages"] + [{"role": "user", "content": user_query}]
        if human_feedback is not None:
            turn["human_feedback"] = human_feedback
            turn["awaiting_feedback"] = True

        result = self.graph.invoke(turn)
        self.state = result  # type: ignore[assignment]
        return self.state

    def apply_feedback(self, feedback: str) -> AgentState:
        """Continue the human-feedback loop with refinement or side intents."""
        return self.run(feedback, human_feedback=feedback)

    def multi_round(self, jd: str | None = None) -> dict[str, Any]:
        """Run multi-round screening using current or provided JD."""
        source = jd or self.state.get("jd_text") or self.state.get("user_query") or ""
        result = run_multi_round_screening(source)
        if result.get("success"):
            self.state["round1_shortlist"] = result.get("round1_shortlist", [])
            self.state["round2_deep_analysis"] = result.get("round2_deep_analysis", [])
            self.state["recommendations"] = result.get("round3_recommendations", [])
            self.state["shortlist"] = result.get("round1_shortlist", [])
            self.state["report"] = result.get("report_markdown", "")
            self.state["requirements"] = result.get("requirements", {})
            self.state["assistant_response"] = result.get("summary", "")
        return result

    def last_response(self) -> str:
        return self.state.get("assistant_response") or ""

    def reasoning_trail(self) -> list[str]:
        return list(self.state.get("reasoning") or [])


def run_query(query: str) -> dict[str, Any]:
    """One-shot helper used by CLI demos and tests."""
    agent = MatchingAgent()
    state = agent.run(query)
    return {
        "success": not bool(state.get("error")),
        "response": state.get("assistant_response"),
        "intent": state.get("intent"),
        "shortlist": state.get("shortlist"),
        "requirements": state.get("requirements"),
        "reasoning": state.get("reasoning"),
        "error": state.get("error"),
        "report": state.get("report"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph Matching Agent")
    parser.add_argument("query", nargs="?", help="Natural language hiring query")
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Start interactive CLI chat loop",
    )
    parser.add_argument(
        "--multi-round",
        action="store_true",
        help="Run multi-round screening on the query/JD",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON result",
    )
    args = parser.parse_args()

    if args.interactive or not args.query:
        from app.cli import interactive_chat

        interactive_chat()
        return

    if args.multi_round:
        result = run_multi_round_screening(args.query)
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(result.get("summary", result))
            for rec in result.get("round3_recommendations", [])[:10]:
                print(
                    f"  [{rec['decision']}] {rec['candidate_name']} "
                    f"(score {rec['match_score']}) — {rec['rationale']}"
                )
        return

    result = run_query(args.query)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(result.get("response") or result.get("error") or "No response")
        if result.get("reasoning"):
            print("\n--- Agent reasoning ---")
            for line in result["reasoning"]:
                print(f"  • {line}")


# Re-export tools for assignment visibility / notebook use
__all__ = [
    "MatchingAgent",
    "AGENT_TOOL_DISPATCH",
    "build_matching_graph",
    "compare_candidates",
    "execute_agent_tool",
    "extract_requirements",
    "generate_interview_questions",
    "get_graph",
    "rag_search",
    "run_query",
]


if __name__ == "__main__":
    main()
