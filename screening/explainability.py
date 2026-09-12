"""Explainability: strengths, gaps, and improvement suggestions."""

from __future__ import annotations

from typing import Any

from core.exceptions import error_response, success_response
from tools.compare_candidates import _resolve_candidate
from tools.extract_requirements import extract_requirements


def _normalize(s: str) -> str:
    return " ".join(s.strip().lower().split())


def analyze_candidate_fit(
    candidate_id: str,
    requirements: dict[str, Any] | None = None,
    jd_text: str | None = None,
    match_score: int | None = None,
) -> dict[str, Any]:
    """Generate strengths, gaps, and improvement tips for one candidate."""
    tool = "analyze_candidate_fit"
    try:
        doc = _resolve_candidate(candidate_id)
        if not doc:
            return error_response(
                f"Candidate not found: {candidate_id}",
                "CANDIDATE_NOT_FOUND",
                tool=tool,
            )

        if requirements is None:
            if not jd_text:
                return error_response(
                    "Provide requirements or jd_text",
                    "INVALID_INPUT",
                    tool=tool,
                )
            req = extract_requirements(jd_text)
            if not req.get("success"):
                return req
            requirements = req

        meta = doc.get("metadata", {})
        name = meta.get("name", candidate_id)
        skills = meta.get("skills", [])
        years = float(meta.get("experience_years", 0))
        text_lower = (doc.get("text") or "").lower()
        skill_norm = {_normalize(s) for s in skills}

        must_have = requirements.get("must_have", [])
        nice_to_have = requirements.get("nice_to_have", [])
        min_years = requirements.get("min_experience_years")

        strengths: list[str] = []
        gaps: list[str] = []
        suggestions: list[str] = []

        matched_must: list[str] = []
        missing_must: list[str] = []
        for req in must_have:
            skill = req.get("skill", "")
            sn = _normalize(skill)
            present = sn in skill_norm or sn in text_lower
            req_years = req.get("min_years")
            if present:
                matched_must.append(skill)
                if req_years is not None and years < float(req_years):
                    gaps.append(
                        f"Has {skill} but experience ({years} yrs) is below "
                        f"required {req_years}+ yrs"
                    )
                    suggestions.append(
                        f"Probe depth of {skill} experience in interview; "
                        f"consider stretch if other strengths compensate."
                    )
                else:
                    strengths.append(f"Meets must-have: {skill}")
            else:
                missing_must.append(skill)
                gaps.append(f"Missing must-have skill: {skill}")
                suggestions.append(
                    f"Ask whether {skill} was used under a different label, "
                    f"or treat as a hard screen-out."
                )

        matched_nice: list[str] = []
        for req in nice_to_have:
            skill = req.get("skill", "")
            sn = _normalize(skill)
            if sn in skill_norm or sn in text_lower:
                matched_nice.append(skill)
                strengths.append(f"Nice-to-have present: {skill}")

        if min_years is not None:
            if years >= float(min_years):
                strengths.append(f"Meets overall experience bar ({years} >= {min_years} yrs)")
            else:
                gaps.append(f"Below overall experience bar ({years} < {min_years} yrs)")
                suggestions.append(
                    "If skills are strong, consider for a mid-level track instead."
                )

        if meta.get("education"):
            strengths.append(f"Education: {', '.join(meta['education'][:2])}")

        # Borderline scoring guidance
        score = match_score
        if score is None:
            # Heuristic 0-100
            must_total = max(len(must_have), 1)
            nice_total = max(len(nice_to_have), 1)
            score = int(
                round(
                    70 * (len(matched_must) / must_total)
                    + 20 * (len(matched_nice) / nice_total)
                    + (10 if min_years is None or years >= float(min_years) else 0)
                )
            )

        verdict = "strong_match"
        if score < 50 or missing_must:
            verdict = "weak_match"
        elif score < 70:
            verdict = "borderline"

        if verdict == "borderline":
            suggestions.append(
                "Borderline candidate: run a focused technical screen on gap areas "
                "before advancing."
            )
            suggestions.append(
                "Ask for a work sample or take-home covering the weakest must-have."
            )
        elif verdict == "weak_match":
            suggestions.append(
                "Likely no-hire for this role unless requirements are relaxed."
            )
        else:
            suggestions.append(
                "Strong fit — advance to onsite; prepare senior-level depth questions."
            )

        report = {
            "candidate_name": name,
            "candidate_id": candidate_id,
            "match_score": score,
            "verdict": verdict,
            "strengths": strengths,
            "gaps": gaps,
            "matched_must_have": matched_must,
            "missing_must_have": missing_must,
            "matched_nice_to_have": matched_nice,
            "improvement_suggestions": suggestions,
            "summary": (
                f"{name}: {verdict.replace('_', ' ')} (score {score}). "
                f"Strengths: {len(strengths)}. Gaps: {len(gaps)}."
            ),
        }
        return success_response(tool, **report)
    except Exception as exc:  # noqa: BLE001
        return error_response(f"analyze_candidate_fit failed: {exc}", "UNEXPECTED", tool=tool)


def build_match_report(
    shortlist: list[dict[str, Any]],
    requirements: dict[str, Any],
    job_title: str = "Role",
) -> dict[str, Any]:
    """Build a detailed multi-candidate match report with explainability."""
    analyses: list[dict[str, Any]] = []
    lines = [
        f"# Match Report — {job_title}",
        "",
        requirements.get("summary", ""),
        "",
        "## Shortlist",
        "",
    ]

    for i, cand in enumerate(shortlist, start=1):
        cid = cand.get("candidate_id") or cand.get("resume_path") or cand.get("candidate_name")
        analysis = analyze_candidate_fit(
            str(cid),
            requirements=requirements,
            match_score=cand.get("match_score"),
        )
        if analysis.get("success"):
            analyses.append(analysis)
            lines.append(
                f"### {i}. {analysis['candidate_name']} — score {analysis['match_score']} "
                f"({analysis['verdict'].replace('_', ' ')})"
            )
            lines.append(f"- Reasoning: {cand.get('reasoning', 'n/a')}")
            if analysis["strengths"]:
                lines.append("- Strengths:")
                for s in analysis["strengths"][:5]:
                    lines.append(f"  - {s}")
            if analysis["gaps"]:
                lines.append("- Gaps:")
                for g in analysis["gaps"][:5]:
                    lines.append(f"  - {g}")
            if analysis["improvement_suggestions"]:
                lines.append("- Suggestions:")
                for s in analysis["improvement_suggestions"][:3]:
                    lines.append(f"  - {s}")
            lines.append("")
        else:
            lines.append(
                f"### {i}. {cand.get('candidate_name', cid)} — "
                f"score {cand.get('match_score', '?')}"
            )
            lines.append(f"- Reasoning: {cand.get('reasoning', 'n/a')}")
            lines.append("")

    markdown = "\n".join(lines)
    return success_response(
        "build_match_report",
        job_title=job_title,
        markdown=markdown,
        analyses=analyses,
        shortlist_size=len(shortlist),
    )
