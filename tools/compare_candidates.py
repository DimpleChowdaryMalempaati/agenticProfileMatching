"""Head-to-head candidate comparison tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import RESUMES_DIR
from core.exceptions import error_response, success_response
from tools.resume_loader import list_resumes, load_resume


def _resolve_candidate(candidate_id: str) -> dict[str, Any] | None:
    """Resolve a candidate by name, filename stem, or resume path."""
    cid = candidate_id.strip()
    if not cid:
        return None

    # Direct path
    direct = Path(cid)
    if direct.exists() and direct.is_file():
        result = load_resume(direct)
        return result.get("document") if result.get("success") else None

    listing = list_resumes(RESUMES_DIR)
    if not listing.get("success"):
        return None

    cid_norm = cid.lower().replace(" ", "_").replace("-", "_")
    cid_name = cid.lower().replace("_", " ")

    for file_info in listing.get("files", []):
        path = Path(file_info["path"])
        stem = path.stem.lower()
        name_from_stem = stem.replace("_", " ")
        if (
            cid_norm == stem
            or cid_norm in stem
            or cid_name == name_from_stem
            or cid_name in name_from_stem
            or cid.lower() in str(path).lower()
        ):
            result = load_resume(path)
            if result.get("success"):
                return result["document"]
    return None


def _skill_overlap(a: list[str], b: list[str]) -> list[str]:
    a_set = {s.lower() for s in a}
    return [s for s in b if s.lower() in a_set]


def compare_candidates(candidate_ids: list) -> dict[str, Any]:
    """Head-to-head comparison of two or more candidates.

    Args:
        candidate_ids: List of candidate names, resume stems, or file paths.

    Returns:
        Side-by-side profiles, shared/unique skills, experience deltas,
        and a plain-language comparison summary.

    Example:
        >>> compare_candidates(["alice_smith", "bob_johnson"])
    """
    tool = "compare_candidates"
    try:
        if not isinstance(candidate_ids, list) or len(candidate_ids) < 2:
            return error_response(
                "Provide at least two candidate_ids",
                "INVALID_INPUT",
                tool=tool,
            )

        profiles: list[dict[str, Any]] = []
        missing: list[str] = []

        for cid in candidate_ids:
            doc = _resolve_candidate(str(cid))
            if not doc:
                missing.append(str(cid))
                continue
            meta = doc.get("metadata", {})
            profiles.append(
                {
                    "candidate_id": Path(doc["path"]).stem,
                    "name": meta.get("name", Path(doc["path"]).stem),
                    "resume_path": doc.get("path"),
                    "experience_years": meta.get("experience_years", 0),
                    "skills": meta.get("skills", []),
                    "education": meta.get("education", []),
                    "word_count": meta.get("word_count", 0),
                    "excerpt": (doc.get("text") or "")[:400],
                }
            )

        if missing:
            return error_response(
                f"Could not resolve candidates: {', '.join(missing)}",
                "CANDIDATE_NOT_FOUND",
                tool=tool,
                missing=missing,
                resolved=[p["name"] for p in profiles],
            )

        if len(profiles) < 2:
            return error_response(
                "Need at least two resolved candidates to compare",
                "INSUFFICIENT_CANDIDATES",
                tool=tool,
            )

        # Shared skills across all
        shared = set(s.lower() for s in profiles[0]["skills"])
        for p in profiles[1:]:
            shared &= {s.lower() for s in p["skills"]}

        unique_by_candidate: dict[str, list[str]] = {}
        for p in profiles:
            unique = [
                s for s in p["skills"] if s.lower() not in shared
            ]
            unique_by_candidate[p["name"]] = unique

        # Experience ranking
        by_exp = sorted(profiles, key=lambda p: p["experience_years"], reverse=True)
        by_skills = sorted(profiles, key=lambda p: len(p["skills"]), reverse=True)

        lines = [
            f"Comparing {len(profiles)} candidates: "
            + ", ".join(p["name"] for p in profiles)
            + "."
        ]
        lines.append(
            "Experience ranking: "
            + " > ".join(f"{p['name']} ({p['experience_years']} yrs)" for p in by_exp)
            + "."
        )
        if shared:
            lines.append(
                "Shared skills: "
                + ", ".join(sorted({s.title() if s.islower() else s for s in shared})[:12])
                + "."
            )
        for name, skills in unique_by_candidate.items():
            if skills:
                lines.append(f"{name} unique skills: {', '.join(skills[:8])}.")

        winner_hint = by_exp[0]["name"]
        if by_skills[0]["name"] == winner_hint:
            lines.append(
                f"Edge: {winner_hint} leads on both experience and skill breadth."
            )
        else:
            lines.append(
                f"Trade-off: {by_exp[0]['name']} has more experience; "
                f"{by_skills[0]['name']} has broader listed skills."
            )

        # Pairwise skill overlaps for top pairs
        pairwise = []
        for i in range(len(profiles)):
            for j in range(i + 1, len(profiles)):
                overlap = _skill_overlap(profiles[i]["skills"], profiles[j]["skills"])
                pairwise.append(
                    {
                        "a": profiles[i]["name"],
                        "b": profiles[j]["name"],
                        "overlap_count": len(overlap),
                        "overlap_skills": overlap[:10],
                        "experience_delta": round(
                            profiles[i]["experience_years"] - profiles[j]["experience_years"],
                            1,
                        ),
                    }
                )

        return success_response(
            tool,
            candidates=profiles,
            shared_skills=sorted(shared),
            unique_skills=unique_by_candidate,
            pairwise=pairwise,
            summary=" ".join(lines),
        )
    except Exception as exc:  # noqa: BLE001
        return error_response(f"compare_candidates failed: {exc}", "UNEXPECTED", tool=tool)
