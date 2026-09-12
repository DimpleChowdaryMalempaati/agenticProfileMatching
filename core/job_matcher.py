"""Job matching engine: semantic + keyword hybrid search with scoring."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import JOB_DESCRIPTIONS_DIR, PROJECT_ROOT, TOP_K
from core.resume_rag import ResumeRAG
from tools.resume_loader import load_resume


@dataclass
class MustHaveRequirement:
    skill: str
    min_years: float | None = None
    description: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MustHaveRequirement":
        return cls(
            skill=data.get("skill", ""),
            min_years=data.get("min_years"),
            description=data.get("description", ""),
        )


@dataclass
class JobDescription:
    title: str
    text: str
    critical_skills: list[str] = field(default_factory=list)
    must_have: list[MustHaveRequirement] = field(default_factory=list)
    source_path: str | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "JobDescription":
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Job description not found: {path}")

        if file_path.suffix.lower() == ".json":
            data = json.loads(file_path.read_text(encoding="utf-8"))
            must_have = [
                MustHaveRequirement.from_dict(m) for m in data.get("must_have", [])
            ]
            return cls(
                title=data.get("title", file_path.stem),
                text=data.get("description", data.get("text", "")),
                critical_skills=data.get("critical_skills", []),
                must_have=must_have,
                source_path=str(file_path.resolve()),
            )

        text = file_path.read_text(encoding="utf-8")
        return cls(
            title=file_path.stem.replace("_", " ").title(),
            text=text,
            source_path=str(file_path.resolve()),
        )


def _relative_resume_path(resume_path: str) -> str:
    """Return project-relative path for cleaner output."""
    try:
        return str(Path(resume_path).resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return resume_path


def _normalize_skill(skill: str) -> str:
    return re.sub(r"\s+", " ", skill.strip().lower())


def _extract_critical_skills_from_jd(jd_text: str) -> list[str]:
    """Extract likely critical skills from free-text job description."""
    known_skills = [
        "Python", "Java", "JavaScript", "TypeScript", "SQL", "AWS", "Docker",
        "Kubernetes", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
        "React", "Node.js", "Go", "Rust", "C++", "Scala", "Spark", "PostgreSQL",
        "MongoDB", "Azure", "GCP", "DevOps", "FastAPI", "Django", "Flask",
        "NLP", "Computer Vision", "Data Analysis", "Pandas", "Scikit-learn",
        "Terraform", "CI/CD", "Microservices", "GraphQL", "REST", "Linux", "Git",
    ]
    found = []
    lower_jd = jd_text.lower()
    for skill in known_skills:
        if skill.lower() in lower_jd:
            found.append(skill)
    return found


def _keyword_score(jd_skills: list[str], candidate_skills: list[str]) -> float:
    if not jd_skills:
        return 0.5
    jd_set = {_normalize_skill(s) for s in jd_skills}
    cand_set = {_normalize_skill(s) for s in candidate_skills}
    if not jd_set:
        return 0.0
    overlap = len(jd_set & cand_set)
    return overlap / len(jd_set)


def _check_must_have(
    requirements: list[MustHaveRequirement],
    candidate_skills: list[str],
    experience_years: float,
    resume_text: str,
) -> tuple[bool, list[str]]:
    """Return (passes_all, list of failure reasons)."""
    failures: list[str] = []
    cand_normalized = {_normalize_skill(s) for s in candidate_skills}
    text_lower = resume_text.lower()

    for req in requirements:
        skill_norm = _normalize_skill(req.skill)
        skill_found = skill_norm in cand_normalized or skill_norm in text_lower
        if not skill_found:
            failures.append(f"Missing required skill: {req.skill}")
            continue
        if req.min_years is not None and experience_years < req.min_years:
            failures.append(
                f"Requires {req.min_years}+ years for {req.skill}, "
                f"candidate has {experience_years}"
            )
    return len(failures) == 0, failures


def _build_reasoning(
    semantic_score: float,
    keyword_score: float,
    matched_skills: list[str],
    sections: list[str],
    must_have_ok: bool,
    must_have_failures: list[str],
) -> str:
    parts: list[str] = []

    if semantic_score >= 0.75:
        parts.append("Strong semantic alignment with job description.")
    elif semantic_score >= 0.55:
        parts.append("Moderate semantic match with relevant background.")
    else:
        parts.append("Partial semantic overlap; review carefully.")

    if matched_skills:
        parts.append(
            f"Matched critical skills: {', '.join(matched_skills[:8])}."
        )
    else:
        parts.append("Few explicit critical skill matches detected.")

    if sections:
        parts.append(f"Relevant sections: {', '.join(sorted(set(sections)))}.")

    if must_have_ok:
        parts.append("Meets all must-have requirements.")
    elif must_have_failures:
        parts.append(
            "Must-have gaps: " + "; ".join(must_have_failures[:3]) + "."
        )

    return " ".join(parts)


class JobMatcher:
    """Hybrid job-resume matching engine."""

    def __init__(self, rag: ResumeRAG | None = None):
        self.rag = rag or ResumeRAG()

    def match_job(
        self,
        job_description: str | JobDescription,
        top_k: int = TOP_K,
        semantic_weight: float = 0.65,
        keyword_weight: float = 0.35,
        apply_must_have_filter: bool = True,
    ) -> dict[str, Any]:
        """Match resumes to a job description.

        Args:
            job_description: Raw JD text or parsed JobDescription object.
            top_k: Number of top candidates to return.
            semantic_weight: Weight for vector similarity (0-1).
            keyword_weight: Weight for keyword overlap (0-1).
            apply_must_have_filter: Drop candidates failing must-have rules.

        Returns:
            JSON-serializable dict matching the assignment output format.

        Example:
            >>> matcher = JobMatcher()
            >>> result = matcher.match_job("Senior Python ML Engineer with 5+ years...")
            >>> print(result["top_matches"][0]["match_score"])
        """
        start = time.perf_counter()

        if isinstance(job_description, str):
            jd = JobDescription(
                title="Custom Job",
                text=job_description,
                critical_skills=_extract_critical_skills_from_jd(job_description),
            )
        else:
            jd = job_description

        if not jd.critical_skills:
            jd.critical_skills = _extract_critical_skills_from_jd(jd.text)

        if not jd.text.strip():
            return {
                "success": False,
                "error": {"message": "Job description is empty", "code": "EMPTY_JD"},
            }

        # Retrieve more chunks than K to aggregate per candidate
        retrieval_k = max(top_k * 5, 30)
        query_result = self.rag.query(jd.text, top_k=retrieval_k)

        if not query_result.get("success"):
            return query_result

        # Aggregate chunks by candidate
        candidate_map: dict[str, dict[str, Any]] = {}

        for hit in query_result.get("results", []):
            meta = hit.get("metadata", {})
            resume_path = meta.get("resume_path", "")
            if not resume_path:
                continue

            if resume_path not in candidate_map:
                candidate_map[resume_path] = {
                    "resume_path": resume_path,
                    "candidate_name": meta.get("candidate_name", "Unknown"),
                    "semantic_scores": [],
                    "sections": [],
                    "excerpts": [],
                    "chunk_metadata": meta,
                }

            entry = candidate_map[resume_path]
            entry["semantic_scores"].append(hit.get("similarity", 0))
            entry["sections"].append(meta.get("section", ""))
            if hit.get("text"):
                entry["excerpts"].append(hit["text"][:300])

        matches: list[dict[str, Any]] = []

        for resume_path, data in candidate_map.items():
            doc_result = load_resume(resume_path)
            if not doc_result.get("success"):
                continue

            doc = doc_result["document"]
            meta = doc.get("metadata", {})
            candidate_skills = meta.get("skills", [])
            experience_years = float(meta.get("experience_years", 0))
            resume_text = doc.get("text", "")

            semantic_score = max(data["semantic_scores"]) if data["semantic_scores"] else 0
            kw_score = _keyword_score(jd.critical_skills, candidate_skills)

            combined = (
                semantic_weight * semantic_score + keyword_weight * kw_score
            ) * 100

            matched_skills = [
                s for s in jd.critical_skills
                if _normalize_skill(s) in {_normalize_skill(c) for c in candidate_skills}
                or _normalize_skill(s) in resume_text.lower()
            ]

            must_have_ok, must_have_failures = _check_must_have(
                jd.must_have, candidate_skills, experience_years, resume_text
            )

            if apply_must_have_filter and jd.must_have and not must_have_ok:
                continue

            # Penalize score if must-haves not fully met (when filter is off)
            if not must_have_ok:
                combined *= 0.7

            reasoning = _build_reasoning(
                semantic_score,
                kw_score,
                matched_skills,
                data["sections"],
                must_have_ok,
                must_have_failures,
            )

            matches.append(
                {
                    "candidate_name": data["candidate_name"],
                    "resume_path": _relative_resume_path(resume_path),
                    "match_score": int(round(min(combined, 100))),
                    "matched_skills": matched_skills,
                    "relevant_excerpts": data["excerpts"][:3],
                    "reasoning": reasoning,
                    "_semantic_score": round(semantic_score, 4),
                    "_keyword_score": round(kw_score, 4),
                    "_experience_years": experience_years,
                }
            )

        matches.sort(key=lambda m: m["match_score"], reverse=True)
        top_matches = matches[:top_k]

        # Remove internal debug fields from output
        for m in top_matches:
            m.pop("_semantic_score", None)
            m.pop("_keyword_score", None)
            m.pop("_experience_years", None)

        elapsed = time.perf_counter() - start
        return {
            "success": True,
            "job_description": jd.text,
            "job_title": jd.title,
            "critical_skills": jd.critical_skills,
            "top_matches": top_matches,
            "total_candidates_evaluated": len(candidate_map),
            "latency_seconds": round(elapsed, 3),
        }

    def match_job_file(
        self,
        job_path: str | Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Load a job description file and run matching.

        Example:
            >>> matcher = JobMatcher()
            >>> result = matcher.match_job_file("job_descriptions/senior_python_ml.json")
        """
        try:
            jd = JobDescription.from_file(job_path)
        except FileNotFoundError as exc:
            return {
                "success": False,
                "error": {"message": str(exc), "code": "JD_NOT_FOUND"},
            }
        except json.JSONDecodeError as exc:
            return {
                "success": False,
                "error": {"message": f"Invalid JSON: {exc}", "code": "INVALID_JSON"},
            }
        return self.match_job(jd, **kwargs)


def match_all_jobs(
    jobs_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run matching for all job descriptions in a directory."""
    root = Path(jobs_dir or JOB_DESCRIPTIONS_DIR)
    matcher = JobMatcher()
    results: list[dict[str, Any]] = []

    for job_file in sorted(root.glob("*")):
        if job_file.suffix.lower() not in {".json", ".txt", ".md"}:
            continue
        result = matcher.match_job_file(job_file)
        results.append({"job_file": str(job_file), "result": result})

    payload = {"success": True, "job_count": len(results), "results": results}
    if output_path:
        Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Match resumes to a job description")
    parser.add_argument(
        "--job",
        required=True,
        help="Path to job description file (.json, .txt) or inline text with --text",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Treat --job argument as raw job description text",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--output", help="Write JSON results to this file")
    args = parser.parse_args()

    matcher = JobMatcher()
    if args.text:
        result = matcher.match_job(args.job, top_k=args.top_k)
    else:
        result = matcher.match_job_file(args.job, top_k=args.top_k)

    output = json.dumps(result, indent=2)
    print(output)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
