"""Parse must-have vs nice-to-have requirements from a job description."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from core.exceptions import error_response, success_response
from core.job_matcher import JobDescription, _extract_critical_skills_from_jd

_YEARS_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)",
    re.IGNORECASE,
)

_MUST_MARKERS = (
    "must have",
    "must-have",
    "required",
    "requirement",
    "mandatory",
    "essential",
    "minimum",
    "at least",
)

_NICE_MARKERS = (
    "nice to have",
    "nice-to-have",
    "preferred",
    "bonus",
    "plus",
    "optional",
    "good to have",
    "desired",
)


def _years_near(text: str, skill: str) -> float | None:
    """Find years-of-experience mentioned near a skill token."""
    lower = text.lower()
    skill_l = skill.lower()
    idx = lower.find(skill_l)
    if idx < 0:
        return None
    window = text[max(0, idx - 40) : idx + len(skill) + 60]
    match = _YEARS_PATTERN.search(window)
    if match:
        return float(match.group(1))
    # Also check global years if skill appears in a must-have style sentence
    for sentence in re.split(r"[.!;\n]", text):
        if skill_l in sentence.lower() and any(m in sentence.lower() for m in _MUST_MARKERS):
            m = _YEARS_PATTERN.search(sentence)
            if m:
                return float(m.group(1))
    return None


def _classify_skill(skill: str, jd_text: str) -> str:
    lower = jd_text.lower()
    skill_l = skill.lower()
    for sentence in re.split(r"[.!;\n]", jd_text):
        s = sentence.lower()
        if skill_l not in s:
            continue
        if any(m in s for m in _MUST_MARKERS):
            return "must_have"
        if any(m in s for m in _NICE_MARKERS):
            return "nice_to_have"
    # Default: first half of critical skills are must-have if years mentioned globally
    if skill_l in lower and _YEARS_PATTERN.search(lower):
        return "must_have"
    return "nice_to_have"


def extract_requirements(jd: str) -> dict[str, Any]:
    """Parse must-have vs nice-to-have requirements from a JD string or file path.

    Args:
        jd: Raw job description text, or a path to a .json/.txt JD file.

    Returns:
        Structured requirements with must_have, nice_to_have, critical_skills,
        experience expectations, and a plain-language summary.

    Example:
        >>> extract_requirements("Find React developers with 3+ years experience")
    """
    tool = "extract_requirements"
    try:
        if not jd or not str(jd).strip():
            return error_response("Job description is empty", "EMPTY_JD", tool=tool)

        raw = str(jd).strip()
        title = "Custom Role"
        critical_skills: list[str] = []
        must_have: list[dict[str, Any]] = []
        nice_to_have: list[dict[str, Any]] = []
        text = raw

        path = Path(raw)
        if path.exists() and path.is_file():
            try:
                parsed = JobDescription.from_file(path)
                title = parsed.title
                text = parsed.text
                critical_skills = list(parsed.critical_skills)
                must_have = [
                    {
                        "skill": m.skill,
                        "min_years": m.min_years,
                        "description": m.description,
                        "category": "must_have",
                    }
                    for m in parsed.must_have
                ]
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                return error_response(
                    f"Failed to load JD file: {exc}",
                    "JD_LOAD_FAILED",
                    tool=tool,
                )

        if not critical_skills:
            critical_skills = _extract_critical_skills_from_jd(text)

        if not must_have:
            for skill in critical_skills:
                category = _classify_skill(skill, text)
                years = _years_near(text, skill)
                entry = {
                    "skill": skill,
                    "min_years": years,
                    "description": "",
                    "category": category,
                }
                if category == "must_have":
                    must_have.append(entry)
                else:
                    nice_to_have.append(entry)

        # Ensure nice_to_have covers remaining critical skills not in must_have
        must_skills = {m["skill"].lower() for m in must_have}
        for skill in critical_skills:
            if skill.lower() in must_skills:
                continue
            if any(n["skill"].lower() == skill.lower() for n in nice_to_have):
                continue
            nice_to_have.append(
                {
                    "skill": skill,
                    "min_years": _years_near(text, skill),
                    "description": "",
                    "category": "nice_to_have",
                }
            )

        # Overall experience floor from JD text
        years_matches = [float(m.group(1)) for m in _YEARS_PATTERN.finditer(text)]
        min_experience_years = max(years_matches) if years_matches else None

        # Soft title extraction from free text
        if title == "Custom Role":
            title_match = re.search(
                r"(?:for|hire|looking for|seeking|role[:\s]+)\s*"
                r"([A-Z][A-Za-z0-9+#./\s-]{2,60}?)(?:\s+with|\s+who|\.|,|$)",
                text,
            )
            if title_match:
                title = title_match.group(1).strip()

        summary_parts = [f"Role: {title}."]
        if must_have:
            summary_parts.append(
                "Must-have: "
                + ", ".join(
                    f"{m['skill']}"
                    + (f" ({m['min_years']}+ yrs)" if m.get("min_years") else "")
                    for m in must_have
                )
                + "."
            )
        if nice_to_have:
            summary_parts.append(
                "Nice-to-have: " + ", ".join(n["skill"] for n in nice_to_have) + "."
            )
        if min_experience_years is not None:
            summary_parts.append(f"Overall experience signal: {min_experience_years}+ years.")

        return success_response(
            tool,
            title=title,
            jd_text=text,
            critical_skills=critical_skills,
            must_have=must_have,
            nice_to_have=nice_to_have,
            min_experience_years=min_experience_years,
            summary=" ".join(summary_parts),
        )
    except Exception as exc:  # noqa: BLE001
        return error_response(f"extract_requirements failed: {exc}", "UNEXPECTED", tool=tool)
