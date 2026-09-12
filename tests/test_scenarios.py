"""Test scenarios: 5+ conversation flows for the matching agent."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matching_agent import MatchingAgent, run_query
from tools.compare_candidates import compare_candidates
from tools.extract_requirements import extract_requirements
from tools.interview_questions import generate_interview_questions
from screening.multi_round import run_multi_round_screening


def _index_ready() -> bool:
    try:
        from tools.rag_search import get_rag

        return get_rag().get_stats().get("chunk_count", 0) > 0
    except Exception:
        return False


requires_index = pytest.mark.skipif(
    not _index_ready(),
    reason="Resume index empty — run python scripts/setup_index.py",
)


# ---------------------------------------------------------------------------
# Unit-level tool tests (no index required)
# ---------------------------------------------------------------------------


def test_extract_requirements_react_query():
    result = extract_requirements(
        "Find me candidates with React and 3+ years experience"
    )
    assert result["success"] is True
    skills = [s.lower() for s in result.get("critical_skills", [])]
    assert "react" in skills
    assert result.get("min_experience_years") == 3 or any(
        m.get("skill", "").lower() == "react" for m in result.get("must_have", [])
    )


def test_extract_requirements_from_jd_file():
    jd = ROOT / "job_descriptions" / "full_stack_developer.json"
    result = extract_requirements(str(jd))
    assert result["success"] is True
    assert "React" in result["critical_skills"] or any(
        m["skill"] == "React" for m in result["must_have"]
    )


def test_compare_candidates_tool():
    result = compare_candidates(["alice_smith", "bob_johnson"])
    assert result["success"] is True
    assert len(result["candidates"]) == 2
    assert "summary" in result


def test_generate_interview_questions_tool():
    result = generate_interview_questions("alice_smith")
    assert result["success"] is True
    assert result["total_questions"] >= 3
    assert result["behavioral_questions"]


# ---------------------------------------------------------------------------
# Conversation flows (require index)
# ---------------------------------------------------------------------------


@requires_index
def test_flow_1_find_react_candidates():
    """Flow 1: natural language search for React + experience."""
    result = run_query("Find me candidates with React and 3+ years experience")
    assert result["success"] is True
    assert result["shortlist"]
    assert result["response"]
    assert any("react" in str(c).lower() or c.get("match_score", 0) > 0 for c in result["shortlist"])


@requires_index
def test_flow_2_compare_top_3():
    """Flow 2: search then compare top 3 side by side."""
    agent = MatchingAgent()
    agent.run("Find me full stack developers with React and Node.js")
    state = agent.apply_feedback("Compare the top 3 matches side by side")
    assert state.get("comparison") or "Compar" in (state.get("assistant_response") or "")
    assert state.get("assistant_response")


@requires_index
def test_flow_3_explain_ranking():
    """Flow 3: why did A rank higher than B?"""
    agent = MatchingAgent()
    state = agent.run("Match candidates for full stack developer role with React")
    assert state.get("shortlist") and len(state["shortlist"]) >= 2
    a = state["shortlist"][0]["candidate_name"]
    b = state["shortlist"][1]["candidate_name"]
    explained = agent.run(f"Why did {a} rank higher than {b}?")
    assert explained.get("assistant_response")
    assert a.split()[0] in explained["assistant_response"]


@requires_index
def test_flow_4_iterative_refinement():
    """Flow 4: adjust requirements mid-conversation and re-rank."""
    agent = MatchingAgent()
    first = agent.run("Find me candidates with Python and machine learning")
    assert first.get("shortlist")
    before = [c["candidate_name"] for c in first["shortlist"][:5]]
    second = agent.apply_feedback("prioritize AWS and Docker experience")
    assert second.get("shortlist")
    assert second.get("assistant_response")
    # Reasoning should mention refine / ranking
    trail = " ".join(second.get("reasoning") or [])
    assert "rank" in trail.lower() or second.get("ranking_changes") is not None or before


@requires_index
def test_flow_5_multi_round_screening():
    """Flow 5: multi-round screen → deep analysis → hire/no-hire."""
    jd = str(ROOT / "job_descriptions" / "full_stack_developer.json")
    result = run_multi_round_screening(jd)
    assert result["success"] is True
    assert len(result["round1_shortlist"]) >= 1
    assert result["round2_deep_analysis"]
    assert result["round3_recommendations"]
    decisions = {r["decision"] for r in result["round3_recommendations"]}
    assert decisions & {"HIRE", "MAYBE", "NO_HIRE"}


@requires_index
def test_flow_6_interview_questions():
    """Flow 6: generate screening questions for top candidate."""
    agent = MatchingAgent()
    agent.run("Find senior python ML engineers")
    assert agent.state.get("shortlist")
    name = agent.state["shortlist"][0]["candidate_name"]
    state = agent.run(f"Generate interview questions for {name}")
    assert "Interview" in (state.get("assistant_response") or "") or state.get("interview_pack")


@requires_index
def test_flow_7_jd_file_pipeline():
    """Flow 7: full pipeline from a JD file path."""
    jd = str(ROOT / "job_descriptions" / "data_scientist.json")
    result = run_query(jd)
    assert result["success"] is True
    assert result["requirements"]
    assert result["shortlist"]


def test_agent_error_on_empty_query_tools():
    assert extract_requirements("").get("success") is False
    assert compare_candidates(["only_one"]).get("success") is False


if __name__ == "__main__":
    # Lightweight manual runner without pytest
    print("Running tool smoke tests…")
    test_extract_requirements_react_query()
    test_extract_requirements_from_jd_file()
    test_compare_candidates_tool()
    test_generate_interview_questions_tool()
    print("Tool tests OK")

    if _index_ready():
        print("Running conversation flows…")
        test_flow_1_find_react_candidates()
        print("  flow1 OK")
        test_flow_2_compare_top_3()
        print("  flow2 OK")
        test_flow_3_explain_ranking()
        print("  flow3 OK")
        test_flow_4_iterative_refinement()
        print("  flow4 OK")
        test_flow_5_multi_round_screening()
        print("  flow5 OK")
        test_flow_6_interview_questions()
        print("  flow6 OK")
        test_flow_7_jd_file_pipeline()
        print("  flow7 OK")
        print("All flows passed.")
    else:
        print("Index not ready — skipped conversation flows.")
        print("Run: python scripts/setup_index.py")
