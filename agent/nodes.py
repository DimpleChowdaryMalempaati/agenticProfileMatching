"""LangGraph node implementations for the matching agent workflow."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from config import JOB_DESCRIPTIONS_DIR, REPORTS_DIR, ROUND1_TOP_K
from agent.router import detect_intent
from agent.state import AgentState
from core.job_matcher import JobDescription, JobMatcher, MustHaveRequirement
from screening.explainability import build_match_report
from screening.multi_round import explain_ranking_change, run_multi_round_screening
from tools.compare_candidates import compare_candidates
from tools.extract_requirements import extract_requirements
from tools.interview_questions import generate_interview_questions
from tools.rag_search import rag_search


def _note(state: AgentState, message: str) -> list[str]:
    return [f"[{state.get('step', 'agent')}] {message}"]


def _req_to_jd(requirements: dict[str, Any]) -> JobDescription:
    must = [
        MustHaveRequirement(
            skill=m.get("skill", ""),
            min_years=m.get("min_years"),
            description=m.get("description", ""),
        )
        for m in requirements.get("must_have", [])
    ]
    return JobDescription(
        title=requirements.get("title", "Custom Role"),
        text=requirements.get("jd_text", ""),
        critical_skills=requirements.get("critical_skills", []),
        must_have=must,
    )


def _attach_ids(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, c in enumerate(matches, start=1):
        item = dict(c)
        path = item.get("resume_path", "")
        stem = Path(path).stem if path else ""
        item["candidate_id"] = stem or item.get("candidate_name", f"cand_{i}")
        item["rank"] = i
        out.append(item)
    return out


def route_intent(state: AgentState) -> dict[str, Any]:
    """START: classify the user query and set routing intent."""
    query = state.get("user_query") or ""
    if state.get("messages"):
        # Prefer latest user message
        for msg in reversed(state["messages"]):
            if msg.get("role") == "user" and msg.get("content"):
                query = msg["content"]
                break

    intent = detect_intent(query, has_shortlist=bool(state.get("shortlist")))
    # Feedback continuation
    if state.get("human_feedback") and state.get("awaiting_feedback"):
        fb = state["human_feedback"].strip().lower()
        if fb in {"done", "end", "quit", "exit", "no", "n"}:
            intent = "feedback"
        elif any(k in fb for k in ("compare", "why", "interview", "report")):
            intent = detect_intent(state["human_feedback"], has_shortlist=True)
        else:
            intent = "refine"

    return {
        "user_query": query,
        "intent": intent,
        "step": "route_intent",
        "error": None,
        "reasoning": _note({**state, "step": "route_intent"}, f"Detected intent: {intent}"),
    }


def parse_jd(state: AgentState) -> dict[str, Any]:
    """Parse JD from query text, known JD filename, or free-form hiring request."""
    query = state.get("user_query", "")
    jd_text = state.get("jd_text") or ""
    jd_title = state.get("jd_title") or ""

    # Explicit JD file reference
    maybe_path = Path(query.strip().strip('"'))
    candidates = [
        maybe_path,
        JOB_DESCRIPTIONS_DIR / query.strip(),
        JOB_DESCRIPTIONS_DIR / f"{query.strip()}.json",
    ]
    # Fuzzy match job title keywords to a JD file
    lower_q = query.lower()
    if not any(p.exists() and p.is_file() for p in candidates):
        for job_file in JOB_DESCRIPTIONS_DIR.glob("*.json"):
            stem = job_file.stem.replace("_", " ")
            if stem in lower_q or any(
                token in lower_q for token in stem.split() if len(token) > 4
            ):
                # Prefer stronger overlap
                overlap = sum(1 for t in stem.split() if t in lower_q)
                if overlap >= 2 or stem in lower_q:
                    candidates.insert(0, job_file)
                    break

    for path in candidates:
        if path.exists() and path.is_file():
            try:
                jd = JobDescription.from_file(path)
                jd_text = jd.text
                jd_title = jd.title
                break
            except Exception as exc:  # noqa: BLE001
                return {
                    "step": "parse_jd",
                    "error": f"Failed to parse JD file: {exc}",
                    "reasoning": _note({**state, "step": "parse_jd"}, str(exc)),
                }

    if not jd_text:
        # Treat the natural language query itself as the JD / search brief
        jd_text = query
        jd_title = jd_title or "Hiring Request"

    return {
        "jd_text": jd_text,
        "jd_title": jd_title,
        "step": "parse_jd",
        "reasoning": _note(
            {**state, "step": "parse_jd"},
            f"Parsed JD '{jd_title}' ({len(jd_text)} chars).",
        ),
    }


def extract_requirements_node(state: AgentState) -> dict[str, Any]:
    """Extract must-have vs nice-to-have requirements."""
    source = state.get("jd_text") or state.get("user_query", "")
    # Merge refined criteria into the source text when present
    refined = state.get("refined_criteria") or {}
    if refined.get("extra_text"):
        source = f"{source}\n\nAdditional criteria: {refined['extra_text']}"

    result = extract_requirements(source)
    if not result.get("success"):
        err = result.get("error", {}).get("message", "Requirement extraction failed")
        return {
            "step": "extract_requirements",
            "error": err,
            "reasoning": _note({**state, "step": "extract_requirements"}, err),
        }

    # Apply refined skill boosts
    if refined.get("must_add"):
        existing = {m["skill"].lower() for m in result.get("must_have", [])}
        for skill in refined["must_add"]:
            if skill.lower() not in existing:
                result.setdefault("must_have", []).append(
                    {"skill": skill, "min_years": refined.get("min_years"), "category": "must_have"}
                )
                result.setdefault("critical_skills", []).append(skill)

    if refined.get("min_years") is not None:
        result["min_experience_years"] = refined["min_years"]

    return {
        "requirements": result,
        "jd_title": result.get("title") or state.get("jd_title") or "Role",
        "jd_text": result.get("jd_text") or state.get("jd_text", ""),
        "step": "extract_requirements",
        "reasoning": _note(
            {**state, "step": "extract_requirements"},
            result.get("summary", "Requirements extracted."),
        ),
    }


def search_resumes(state: AgentState) -> dict[str, Any]:
    """Search resume corpus via RAG + hybrid matcher."""
    requirements = state.get("requirements") or {}
    jd = _req_to_jd(requirements) if requirements.get("jd_text") else None
    query = requirements.get("jd_text") or state.get("jd_text") or state.get("user_query", "")

    # RAG retrieval for transparency
    rag = rag_search(query, top_k=20)
    if not rag.get("success"):
        err = rag.get("error", {}).get("message", "RAG search failed")
        # Continue with matcher if possible; still surface error if index empty
        if rag.get("error", {}).get("code") == "INDEX_NOT_READY":
            return {
                "step": "search_resumes",
                "error": err,
                "reasoning": _note({**state, "step": "search_resumes"}, err),
            }

    matcher = JobMatcher()
    if jd is not None:
        match_result = matcher.match_job(jd, top_k=ROUND1_TOP_K, apply_must_have_filter=False)
    else:
        match_result = matcher.match_job(query, top_k=ROUND1_TOP_K, apply_must_have_filter=False)

    if not match_result.get("success"):
        err = match_result.get("error", {})
        msg = err.get("message") if isinstance(err, dict) else str(err)
        return {
            "step": "search_resumes",
            "error": msg or "Matching failed",
            "reasoning": _note({**state, "step": "search_resumes"}, msg or "Matching failed"),
        }

    candidates = _attach_ids(match_result.get("top_matches", []))
    return {
        "candidates": candidates,
        "step": "search_resumes",
        "reasoning": _note(
            {**state, "step": "search_resumes"},
            f"Evaluated {match_result.get('total_candidates_evaluated', '?')} candidates; "
            f"retrieved {len(candidates)} for ranking. "
            f"RAG hits: {rag.get('count', 0) if rag.get('success') else 0}.",
        ),
    }


def rank_candidates(state: AgentState) -> dict[str, Any]:
    """Rank / re-rank shortlist; explain changes when refining."""
    candidates = list(state.get("candidates") or [])
    requirements = state.get("requirements") or {}
    min_years = requirements.get("min_experience_years")
    must_skills = [m.get("skill", "").lower() for m in requirements.get("must_have", [])]

    def _boost(c: dict[str, Any]) -> float:
        score = float(c.get("match_score", 0))
        matched = [s.lower() for s in c.get("matched_skills", [])]
        for skill in must_skills:
            if skill and skill in matched:
                score += 3
            elif skill and skill in (c.get("reasoning") or "").lower():
                score += 1
        # Soft experience preference
        # experience not always on match object — parse from reasoning if needed
        if min_years is not None:
            # slight preference already baked into matcher; keep stable
            pass
        return score

    ranked = sorted(candidates, key=_boost, reverse=True)
    for i, c in enumerate(ranked, start=1):
        c["rank"] = i
        c["match_score"] = int(min(100, round(_boost(c))))

    previous = state.get("shortlist") or state.get("previous_shortlist") or []
    changes_payload: list[dict[str, Any]] = []
    change_explanation = ""
    if previous and state.get("intent") == "refine":
        expl = explain_ranking_change(
            previous,
            ranked,
            criteria_note=(state.get("refined_criteria") or {}).get("extra_text", ""),
        )
        changes_payload = expl.get("changes", [])
        change_explanation = expl.get("explanation", "")

    return {
        "previous_shortlist": previous,
        "shortlist": ranked,
        "ranking_changes": changes_payload,
        "step": "rank_candidates",
        "reasoning": _note(
            {**state, "step": "rank_candidates"},
            f"Ranked {len(ranked)} candidates. "
            + (change_explanation or f"Top: {ranked[0]['candidate_name']}." if ranked else "Empty shortlist."),
        ),
    }


def generate_report(state: AgentState) -> dict[str, Any]:
    """Generate explainability match report and optional file output."""
    shortlist = state.get("shortlist") or []
    requirements = state.get("requirements") or {}
    title = state.get("jd_title") or requirements.get("title") or "Role"

    if not shortlist:
        return {
            "step": "generate_report",
            "report": "No candidates to report.",
            "assistant_response": "I don't have a shortlist yet. Ask me to find candidates first.",
            "awaiting_feedback": True,
            "reasoning": _note({**state, "step": "generate_report"}, "Empty shortlist."),
        }

    built = build_match_report(shortlist, requirements, job_title=title)
    markdown = built.get("markdown", "")

    # Persist report
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^\w\-]+", "_", title).strip("_") or "report"
        out = REPORTS_DIR / f"match_report_{safe}.md"
        out.write_text(markdown, encoding="utf-8")
        saved = str(out)
    except OSError:
        saved = ""

    top_lines = []
    for c in shortlist[:5]:
        top_lines.append(
            f"{c.get('rank', '?')}. {c.get('candidate_name')} "
            f"(score {c.get('match_score')}) — {c.get('reasoning', '')[:120]}"
        )

    response = (
        f"## Shortlist for {title}\n\n"
        + "\n".join(top_lines)
        + ("\n\n" + (state.get("reasoning") or [""])[-1] if state.get("ranking_changes") else "")
        + (f"\n\nFull report saved to `{saved}`." if saved else "")
        + "\n\nReply with feedback to refine (e.g. 'prioritize AWS'), "
        "'compare top 3', 'multi-round', or 'done'."
    )

    return {
        "report": markdown,
        "assistant_response": response,
        "awaiting_feedback": True,
        "step": "generate_report",
        "messages": [{"role": "assistant", "content": response}],
        "reasoning": _note(
            {**state, "step": "generate_report"},
            f"Generated report for {len(shortlist)} candidates.",
        ),
    }


def human_feedback(state: AgentState) -> dict[str, Any]:
    """Process human feedback: end, refine, or branch to other intents."""
    feedback = (state.get("human_feedback") or "").strip()
    if not feedback:
        return {
            "step": "human_feedback",
            "feedback_action": "end",
            "awaiting_feedback": False,
            "reasoning": _note({**state, "step": "human_feedback"}, "No feedback — ending."),
        }

    lower = feedback.lower()
    if lower in {"done", "end", "quit", "exit", "no", "n", "thanks", "thank you"}:
        return {
            "step": "human_feedback",
            "feedback_action": "end",
            "awaiting_feedback": False,
            "assistant_response": "Closing this matching session. Good luck with hiring!",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Closing this matching session. Good luck with hiring!",
                }
            ],
            "reasoning": _note({**state, "step": "human_feedback"}, "User ended session."),
        }

    # Parse refinement hints
    refined = dict(state.get("refined_criteria") or {})
    refined["extra_text"] = feedback
    years = re.search(r"(\d+)\+?\s*years?", feedback, re.I)
    if years:
        refined["min_years"] = float(years.group(1))

    # Skills to prioritize
    from core.job_matcher import _extract_critical_skills_from_jd

    skills = _extract_critical_skills_from_jd(feedback)
    if skills:
        refined["must_add"] = list(
            dict.fromkeys((refined.get("must_add") or []) + skills)
        )

    return {
        "step": "human_feedback",
        "feedback_action": "refine",
        "awaiting_feedback": False,
        "user_query": feedback,
        "intent": "refine",
        "refined_criteria": refined,
        "human_feedback": feedback,
        "reasoning": _note(
            {**state, "step": "human_feedback"},
            f"Refining with feedback: {feedback[:120]}",
        ),
    }


def compare_node(state: AgentState) -> dict[str, Any]:
    """Compare candidates mentioned in the query or top N from shortlist."""
    query = state.get("user_query", "")
    shortlist = state.get("shortlist") or []

    ids: list[str] = []
    # "top 3"
    top_n = re.search(r"top\s+(\d+)", query, re.I)
    if top_n and shortlist:
        n = min(int(top_n.group(1)), len(shortlist))
        ids = [c["candidate_id"] for c in shortlist[:n]]
    else:
        # Extract names that appear in shortlist
        for c in shortlist:
            name = c.get("candidate_name", "")
            if name and name.lower() in query.lower():
                ids.append(c["candidate_id"])
        # Fallback: raw tokens that look like names / stems
        if len(ids) < 2:
            tokens = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b", query)
            ids.extend(tokens)

    if len(ids) < 2 and shortlist:
        ids = [c["candidate_id"] for c in shortlist[: min(3, len(shortlist))]]

    result = compare_candidates(ids)
    if not result.get("success"):
        msg = result.get("error", {}).get("message", "Comparison failed")
        return {
            "step": "compare",
            "error": msg,
            "assistant_response": msg,
            "messages": [{"role": "assistant", "content": msg}],
            "awaiting_feedback": True,
            "reasoning": _note({**state, "step": "compare"}, msg),
        }

    response = result.get("summary", "Comparison complete.")
    return {
        "comparison": result,
        "assistant_response": response,
        "messages": [{"role": "assistant", "content": response}],
        "awaiting_feedback": True,
        "step": "compare",
        "reasoning": _note({**state, "step": "compare"}, response),
    }


def explain_rank_node(state: AgentState) -> dict[str, Any]:
    """Explain why candidate A ranked higher than B."""
    query = state.get("user_query", "")
    shortlist = state.get("shortlist") or []
    by_name = {c.get("candidate_name", "").lower(): c for c in shortlist}

    names = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b", query)
    # Also try stems from shortlist mentioned in query
    mentioned = []
    for c in shortlist:
        n = c.get("candidate_name", "")
        if n and n.lower() in query.lower():
            mentioned.append(c)

    if len(mentioned) >= 2:
        a, b = mentioned[0], mentioned[1]
    elif len(names) >= 2:
        a = by_name.get(names[0].lower()) or {"candidate_name": names[0], "match_score": "?"}
        b = by_name.get(names[1].lower()) or {"candidate_name": names[1], "match_score": "?"}
    elif len(shortlist) >= 2:
        a, b = shortlist[0], shortlist[1]
    else:
        msg = "Need at least two candidates on the shortlist to explain rankings."
        return {
            "assistant_response": msg,
            "messages": [{"role": "assistant", "content": msg}],
            "awaiting_feedback": True,
            "step": "explain_rank",
        }

    higher, lower = (a, b)
    try:
        if int(a.get("match_score", 0)) < int(b.get("match_score", 0)):
            higher, lower = b, a
    except (TypeError, ValueError):
        pass

    ha = set(s.lower() for s in higher.get("matched_skills", []))
    hb = set(s.lower() for s in lower.get("matched_skills", []))
    only_higher = sorted(ha - hb)
    only_lower = sorted(hb - ha)

    response = (
        f"{higher.get('candidate_name')} ranks higher than {lower.get('candidate_name')} "
        f"({higher.get('match_score')} vs {lower.get('match_score')}).\n"
        f"- {higher.get('candidate_name')}: {higher.get('reasoning', 'n/a')}\n"
        f"- {lower.get('candidate_name')}: {lower.get('reasoning', 'n/a')}\n"
    )
    if only_higher:
        response += f"- Skills favoring {higher.get('candidate_name')}: {', '.join(only_higher)}\n"
    if only_lower:
        response += f"- Skills only on {lower.get('candidate_name')}: {', '.join(only_lower)}\n"

    return {
        "assistant_response": response,
        "messages": [{"role": "assistant", "content": response}],
        "awaiting_feedback": True,
        "step": "explain_rank",
        "reasoning": _note({**state, "step": "explain_rank"}, response[:200]),
    }


def interview_node(state: AgentState) -> dict[str, Any]:
    """Generate interview questions for a candidate."""
    query = state.get("user_query", "")
    shortlist = state.get("shortlist") or []
    candidate_id = None

    for c in shortlist:
        name = c.get("candidate_name", "")
        if name and name.lower() in query.lower():
            candidate_id = c.get("candidate_id")
            break
    if not candidate_id and shortlist:
        candidate_id = shortlist[0].get("candidate_id")

    if not candidate_id:
        msg = "No candidate identified. Run a search first or name a candidate."
        return {
            "assistant_response": msg,
            "messages": [{"role": "assistant", "content": msg}],
            "awaiting_feedback": True,
            "step": "interview",
        }

    pack = generate_interview_questions(
        candidate_id,
        job_context=state.get("jd_text") or "",
    )
    if not pack.get("success"):
        msg = pack.get("error", {}).get("message", "Failed to generate questions")
        return {
            "error": msg,
            "assistant_response": msg,
            "messages": [{"role": "assistant", "content": msg}],
            "awaiting_feedback": True,
            "step": "interview",
        }

    lines = [f"Interview pack for {pack.get('candidate_name')}:", "", "Behavioral:"]
    for q in pack.get("behavioral_questions", [])[:3]:
        lines.append(f"- {q['question']}")
    lines.append("\nTechnical:")
    for q in pack.get("technical_questions", [])[:5]:
        lines.append(f"- ({q.get('skill')}) {q['question']}")
    if pack.get("gap_questions"):
        lines.append("\nGap probes:")
        for q in pack["gap_questions"]:
            lines.append(f"- {q['question']}")

    response = "\n".join(lines)
    return {
        "interview_pack": pack,
        "assistant_response": response,
        "messages": [{"role": "assistant", "content": response}],
        "awaiting_feedback": True,
        "step": "interview",
        "reasoning": _note({**state, "step": "interview"}, pack.get("summary", "")),
    }


def multi_round_node(state: AgentState) -> dict[str, Any]:
    """Run multi-round screening pipeline."""
    jd = state.get("jd_text") or state.get("user_query", "")
    result = run_multi_round_screening(jd)
    if not result.get("success"):
        msg = result.get("error", {}).get("message", "Multi-round screening failed")
        return {
            "error": msg,
            "assistant_response": msg,
            "messages": [{"role": "assistant", "content": msg}],
            "awaiting_feedback": True,
            "step": "multi_round",
        }

    recs = result.get("round3_recommendations", [])
    lines = [result.get("summary", ""), "", "Final recommendations:"]
    for r in recs[:10]:
        lines.append(
            f"- {r['decision']}: {r['candidate_name']} "
            f"(score {r['match_score']}) — {r['rationale']}"
        )

    response = "\n".join(lines)
    return {
        "requirements": result.get("requirements") or state.get("requirements") or {},
        "round1_shortlist": result.get("round1_shortlist", []),
        "round2_deep_analysis": result.get("round2_deep_analysis", []),
        "recommendations": recs,
        "shortlist": result.get("round1_shortlist", []),
        "report": result.get("report_markdown", ""),
        "screening_round": 3,
        "assistant_response": response,
        "messages": [{"role": "assistant", "content": response}],
        "awaiting_feedback": True,
        "step": "multi_round",
        "reasoning": _note({**state, "step": "multi_round"}, result.get("summary", "")),
    }


def error_node(state: AgentState) -> dict[str, Any]:
    """Surface a recoverable error to the user and wait for new input."""
    msg = state.get("error") or "Something went wrong."
    response = f"Error: {msg}\nYou can retry with a different query, or run `python scripts/setup_index.py`."
    return {
        "assistant_response": response,
        "messages": [{"role": "assistant", "content": response}],
        "awaiting_feedback": True,
        "step": "error",
        "reasoning": _note({**state, "step": "error"}, msg),
    }
