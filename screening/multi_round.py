"""Multi-round screening: initial → deep analysis → hire/no-hire."""

from __future__ import annotations

from typing import Any

from config import ROUND1_TOP_K, ROUND2_TOP_K, ROUND3_TOP_K
from core.exceptions import error_response, success_response
from core.job_matcher import JobDescription, JobMatcher, MustHaveRequirement
from screening.explainability import analyze_candidate_fit, build_match_report
from tools.extract_requirements import extract_requirements


def _requirements_to_jd(requirements: dict[str, Any]) -> JobDescription:
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


def run_multi_round_screening(
    jd: str,
    *,
    round1_k: int = ROUND1_TOP_K,
    round2_k: int = ROUND2_TOP_K,
    round3_k: int = ROUND3_TOP_K,
    matcher: JobMatcher | None = None,
) -> dict[str, Any]:
    """Execute three-round screening pipeline.

    Round 1 — Initial screen: top N from full corpus (hybrid match).
    Round 2 — Deep analysis: strengths/gaps on round-1 shortlist.
    Round 3 — Final hire / no-hire / maybe recommendations.
    """
    tool = "multi_round_screening"
    try:
        req = extract_requirements(jd)
        if not req.get("success"):
            return req

        job = _requirements_to_jd(req)
        engine = matcher or JobMatcher()

        # Round 1: broad screen (retrieve more than K, no hard filter drop for visibility)
        round1 = engine.match_job(
            job,
            top_k=max(round1_k, 10),
            apply_must_have_filter=False,
        )
        if not round1.get("success"):
            return round1

        initial = round1.get("top_matches", [])[:round1_k]
        for i, c in enumerate(initial, start=1):
            c["round"] = 1
            c["rank"] = i
            c["candidate_id"] = (
                c.get("resume_path", "").split("\\")[-1].split("/")[-1].replace(".txt", "")
                or c.get("candidate_name", f"cand_{i}")
            )

        # Round 2: deep analysis on top 10
        deep: list[dict[str, Any]] = []
        for c in initial[:round2_k]:
            analysis = analyze_candidate_fit(
                c["candidate_id"],
                requirements=req,
                match_score=c.get("match_score"),
            )
            if analysis.get("success"):
                merged = {**c, "deep_analysis": analysis, "round": 2}
                deep.append(merged)
            else:
                deep.append({**c, "round": 2, "deep_analysis_error": analysis})

        # Prefer strong/borderline with fewer gaps
        def _deep_key(item: dict[str, Any]) -> tuple:
            da = item.get("deep_analysis") or {}
            verdict = da.get("verdict", "weak_match")
            verdict_rank = {"strong_match": 0, "borderline": 1, "weak_match": 2}.get(
                verdict, 3
            )
            gaps = len(da.get("gaps", []))
            return (verdict_rank, -int(item.get("match_score", 0)), gaps)

        deep_sorted = sorted(deep, key=_deep_key)
        for i, c in enumerate(deep_sorted, start=1):
            c["deep_rank"] = i

        # Round 3: final recommendations
        finalists = deep_sorted[:round3_k]
        recommendations: list[dict[str, Any]] = []
        for c in deep_sorted:
            da = c.get("deep_analysis") or {}
            verdict = da.get("verdict", "weak_match")
            score = int(c.get("match_score", 0))
            missing = da.get("missing_must_have", [])

            if verdict == "strong_match" and score >= 70 and not missing:
                decision = "HIRE"
                rationale = "Strong skill alignment, meets must-haves, solid score."
            elif verdict == "borderline" or (score >= 55 and len(missing) <= 1):
                decision = "MAYBE"
                rationale = (
                    "Borderline fit — proceed only after targeted technical screen "
                    "on listed gaps."
                )
            else:
                decision = "NO_HIRE"
                rationale = "Material gaps vs must-haves or weak overall match."

            recommendations.append(
                {
                    "candidate_name": c.get("candidate_name"),
                    "candidate_id": c.get("candidate_id"),
                    "match_score": score,
                    "decision": decision,
                    "rationale": rationale,
                    "verdict": verdict,
                    "gaps": da.get("gaps", [])[:5],
                    "strengths": da.get("strengths", [])[:5],
                }
            )

        report = build_match_report(initial, req, job_title=req.get("title", "Role"))

        hire_count = sum(1 for r in recommendations if r["decision"] == "HIRE")
        maybe_count = sum(1 for r in recommendations if r["decision"] == "MAYBE")

        summary = (
            f"Multi-round complete for '{req.get('title')}'. "
            f"Round 1 screened {round1.get('total_candidates_evaluated', '?')} candidates → "
            f"top {len(initial)}. Round 2 deep-analyzed {len(deep_sorted)}. "
            f"Round 3: {hire_count} HIRE, {maybe_count} MAYBE, "
            f"{len(recommendations) - hire_count - maybe_count} NO_HIRE."
        )

        return success_response(
            tool,
            requirements=req,
            round1_shortlist=initial,
            round2_deep_analysis=deep_sorted,
            round3_recommendations=recommendations,
            finalists=finalists,
            report_markdown=report.get("markdown", ""),
            summary=summary,
            total_evaluated=round1.get("total_candidates_evaluated"),
        )
    except Exception as exc:  # noqa: BLE001
        return error_response(
            f"multi_round_screening failed: {exc}",
            "UNEXPECTED",
            tool=tool,
        )


def explain_ranking_change(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    criteria_note: str = "",
) -> dict[str, Any]:
    """Explain how a mid-conversation requirement change affected rankings."""
    before_rank = {
        c.get("candidate_name") or c.get("candidate_id"): i + 1
        for i, c in enumerate(before)
    }
    after_rank = {
        c.get("candidate_name") or c.get("candidate_id"): i + 1
        for i, c in enumerate(after)
    }

    changes: list[dict[str, Any]] = []
    all_names = set(before_rank) | set(after_rank)
    for name in all_names:
        b = before_rank.get(name)
        a = after_rank.get(name)
        if b is None and a is not None:
            changes.append(
                {"candidate": name, "change": "entered_shortlist", "new_rank": a}
            )
        elif a is None and b is not None:
            changes.append(
                {"candidate": name, "change": "left_shortlist", "old_rank": b}
            )
        elif a is not None and b is not None and a != b:
            direction = "up" if a < b else "down"
            changes.append(
                {
                    "candidate": name,
                    "change": direction,
                    "old_rank": b,
                    "new_rank": a,
                    "delta": b - a,
                }
            )

    changes.sort(key=lambda x: abs(x.get("delta", 10)), reverse=True)
    lines = []
    if criteria_note:
        lines.append(f"Criteria update: {criteria_note}")
    if not changes:
        lines.append("Rankings unchanged after refinement.")
    else:
        for ch in changes[:8]:
            if ch["change"] == "up":
                lines.append(
                    f"{ch['candidate']} moved up {ch['old_rank']} → {ch['new_rank']}."
                )
            elif ch["change"] == "down":
                lines.append(
                    f"{ch['candidate']} moved down {ch['old_rank']} → {ch['new_rank']}."
                )
            elif ch["change"] == "entered_shortlist":
                lines.append(f"{ch['candidate']} entered shortlist at rank {ch['new_rank']}.")
            else:
                lines.append(f"{ch['candidate']} dropped off the shortlist (was #{ch['old_rank']}).")

    return success_response(
        "explain_ranking_change",
        changes=changes,
        explanation=" ".join(lines),
    )
