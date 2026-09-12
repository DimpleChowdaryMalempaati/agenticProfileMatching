"""Build or rebuild the ChromaDB resume index."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.resume_rag import ResumeRAG, build_index


def main() -> None:
    print("Building resume RAG index…")
    result = build_index()
    print(json.dumps(result, indent=2))
    if not result.get("success"):
        sys.exit(1)
    rag = ResumeRAG()
    print("Stats:", json.dumps(rag.get_stats(), indent=2))


if __name__ == "__main__":
    main()
