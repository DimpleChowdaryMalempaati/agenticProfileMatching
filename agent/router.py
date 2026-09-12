"""Intent routing for natural-language recruiter queries."""

from __future__ import annotations

import re
from typing import Literal

Intent = Literal[
    "full_pipeline",
    "search",
    "compare",
    "explain_rank",
    "refine",
    "interview",
    "multi_round",
    "report",
    "feedback",
    "general",
]


_PATTERNS: list[tuple[Intent, re.Pattern[str]]] = [
    (
        "compare",
        re.compile(r"\b(compare|side[\s-]?by[\s-]?side|vs\.?|versus|head[\s-]?to[\s-]?head)\b", re.I),
    ),
    (
        "explain_rank",
        re.compile(r"\b(why\s+did|why\s+is|rank(?:ed|ing)?\s+higher|higher than|explain\s+rank)\b", re.I),
    ),
    (
        "interview",
        re.compile(
            r"\b(interview\s+questions?|screening\s+questions?|ask\s+.+\bcandidate)\b",
            re.I,
        ),
    ),
    (
        "multi_round",
        re.compile(r"\b(multi[\s-]?round|deep\s+analys|hire\s*/?\s*no[\s-]?hire|final\s+recommend)\b", re.I),
    ),
    (
        "refine",
        re.compile(
            r"\b(also\s+need|instead|change\s+requirement|re-?rank|prioriti[sz]e|update\s+criteria|"
            r"now\s+focus|add\s+requirement|remove\s+requirement|adjust)\b",
            re.I,
        ),
    ),
    (
        "report",
        re.compile(r"\b(match\s+report|generate\s+report|detailed\s+report|strengths?\s+and\s+gaps?)\b", re.I),
    ),
    (
        "search",
        re.compile(
            r"\b(find\s+me|search|candidates?\s+with|who\s+has|looking\s+for|need\s+someone|"
            r"match\s+(?:for|to)|screen\s+for)\b",
            re.I,
        ),
    ),
    (
        "full_pipeline",
        re.compile(r"\b(run\s+pipeline|parse\s+jd|full\s+match|start\s+matching)\b", re.I),
    ),
]


def detect_intent(query: str, has_shortlist: bool = False) -> Intent:
    """Classify a user utterance into an agent intent."""
    q = (query or "").strip()
    if not q:
        return "general"

    for intent, pattern in _PATTERNS:
        if pattern.search(q):
            # refine only meaningful if we already have a shortlist, else treat as search
            if intent == "refine" and not has_shortlist:
                return "search"
            return intent

    # JD file path or long JD text → full pipeline
    if q.endswith(".json") or len(q) > 280:
        return "full_pipeline"

    if has_shortlist and re.search(r"\b(top\s+\d+|shortlist|matches)\b", q, re.I):
        return "report"

    return "search"
