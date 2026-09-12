"""Generate screening interview questions for a candidate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import RESUMES_DIR
from core.exceptions import error_response, success_response
from tools.compare_candidates import _resolve_candidate
from tools.resume_loader import load_resume


_GENERIC_BEHAVIORAL = [
    "Tell me about a project you owned end-to-end. What was hardest?",
    "Describe a time you disagreed with a technical decision. How did you handle it?",
    "How do you prioritize when requirements change mid-sprint?",
]

_SKILL_QUESTION_BANK: dict[str, list[str]] = {
    "python": [
        "Walk through how you structure a production Python service.",
        "How do you approach testing and packaging Python code?",
    ],
    "react": [
        "How do you manage state and performance in a large React app?",
        "Describe your approach to reusable component design.",
    ],
    "javascript": [
        "Explain event loop pitfalls you've debugged in production.",
        "How do you ensure type safety in a JS/TS codebase?",
    ],
    "typescript": [
        "When do you prefer interfaces vs types, and why?",
        "How have you used TypeScript to prevent API contract bugs?",
    ],
    "node.js": [
        "How do you design Node.js APIs for reliability under load?",
        "Describe your approach to async error handling in Node.",
    ],
    "java": [
        "How do you design Spring/microservices boundaries?",
        "Explain a concurrency issue you solved in Java.",
    ],
    "sql": [
        "How do you diagnose a slow query in production?",
        "Describe an indexing strategy you designed.",
    ],
    "aws": [
        "Walk through an AWS architecture you owned and its failure modes.",
        "How do you balance cost vs reliability in cloud design?",
    ],
    "docker": [
        "How do you keep container images small and secure?",
        "Describe a Docker/K8s debugging incident.",
    ],
    "kubernetes": [
        "How do you approach rolling deployments and rollbacks?",
        "Explain how you'd debug a CrashLoopBackOff pod.",
    ],
    "machine learning": [
        "How do you decide when a model is ready for production?",
        "Describe how you monitor model drift after launch.",
    ],
    "tensorflow": [
        "Walk through a TensorFlow training pipeline you built.",
        "How do you manage experiment reproducibility?",
    ],
    "pytorch": [
        "How do you structure PyTorch training code for iteration speed?",
        "Describe a debugging story with training instability.",
    ],
    "nlp": [
        "How do you evaluate NLP model quality beyond accuracy?",
        "Describe a text data quality issue you fixed.",
    ],
    "devops": [
        "Describe a CI/CD pipeline you improved and the metrics you tracked.",
        "How do you handle secrets and environment promotion?",
    ],
}


def generate_interview_questions(
    candidate_id: str,
    focus_skills: list[str] | None = None,
    job_context: str | None = None,
) -> dict[str, Any]:
    """Create tailored screening questions for a candidate.

    Args:
        candidate_id: Candidate name, resume stem, or file path.
        focus_skills: Optional skills to emphasize (defaults to resume skills).
        job_context: Optional JD snippet to bias questions toward the role.

    Returns:
        Behavioral, technical, and gap-probing questions with rationale.

    Example:
        >>> generate_interview_questions("alice_smith", focus_skills=["React"])
    """
    tool = "generate_interview_questions"
    try:
        if not candidate_id or not str(candidate_id).strip():
            return error_response("candidate_id is required", "INVALID_INPUT", tool=tool)

        doc = _resolve_candidate(str(candidate_id))
        if not doc:
            # Last attempt: absolute under resumes
            guess = RESUMES_DIR / f"{candidate_id}.txt"
            if guess.exists():
                loaded = load_resume(guess)
                doc = loaded.get("document") if loaded.get("success") else None
        if not doc:
            return error_response(
                f"Candidate not found: {candidate_id}",
                "CANDIDATE_NOT_FOUND",
                tool=tool,
            )

        meta = doc.get("metadata", {})
        name = meta.get("name", Path(doc["path"]).stem)
        skills = [s for s in (focus_skills or meta.get("skills", [])) if s]
        years = meta.get("experience_years", 0)
        text_lower = (doc.get("text") or "").lower()

        technical: list[dict[str, str]] = []
        used_skills: set[str] = set()
        for skill in skills:
            key = skill.lower()
            bank = _SKILL_QUESTION_BANK.get(key)
            if not bank:
                # fuzzy key match
                for k, qs in _SKILL_QUESTION_BANK.items():
                    if k in key or key in k:
                        bank = qs
                        break
            if not bank:
                bank = [
                    f"Describe a recent project where you used {skill} in production.",
                    f"What trade-offs do you consider when applying {skill}?",
                ]
            if key in used_skills:
                continue
            used_skills.add(key)
            technical.append(
                {
                    "skill": skill,
                    "question": bank[0],
                    "rationale": f"Validates claimed {skill} experience ({years} yrs overall).",
                }
            )
            if len(bank) > 1 and len(technical) < 8:
                technical.append(
                    {
                        "skill": skill,
                        "question": bank[1],
                        "rationale": f"Probes depth with {skill}.",
                    }
                )
            if len(technical) >= 6:
                break

        # Gap probes: skills mentioned in job_context but missing from resume
        gap_questions: list[dict[str, str]] = []
        if job_context:
            from core.job_matcher import _extract_critical_skills_from_jd

            needed = _extract_critical_skills_from_jd(job_context)
            resume_skills = {s.lower() for s in meta.get("skills", [])}
            for skill in needed:
                if skill.lower() not in resume_skills and skill.lower() not in text_lower:
                    gap_questions.append(
                        {
                            "skill": skill,
                            "question": (
                                f"Your resume does not clearly list {skill}. "
                                f"Have you used it, and in what context?"
                            ),
                            "rationale": f"Probes potential gap vs role need: {skill}.",
                        }
                    )
                if len(gap_questions) >= 3:
                    break

        behavioral = [
            {"question": q, "rationale": "Assesses collaboration and ownership."}
            for q in _GENERIC_BEHAVIORAL
        ]

        # Seniority-calibrated follow-up
        if years >= 7:
            behavioral.append(
                {
                    "question": "How have you mentored engineers or raised team standards?",
                    "rationale": "Senior profile — validate leadership impact.",
                }
            )
        elif years <= 3:
            behavioral.append(
                {
                    "question": "What support or mentorship helps you ramp fastest?",
                    "rationale": "Earlier-career profile — assess learning velocity.",
                }
            )

        return success_response(
            tool,
            candidate_id=Path(doc["path"]).stem,
            candidate_name=name,
            experience_years=years,
            focus_skills=skills[:10],
            behavioral_questions=behavioral,
            technical_questions=technical,
            gap_questions=gap_questions,
            total_questions=len(behavioral) + len(technical) + len(gap_questions),
            summary=(
                f"Generated {len(behavioral) + len(technical) + len(gap_questions)} "
                f"screening questions for {name}."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return error_response(
            f"generate_interview_questions failed: {exc}",
            "UNEXPECTED",
            tool=tool,
        )
