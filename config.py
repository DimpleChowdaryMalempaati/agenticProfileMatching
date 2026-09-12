"""Central configuration for the agentic profile matching system."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent

RESUMES_DIR = Path(os.getenv("RESUMES_DIR", PROJECT_ROOT / "resumes"))
JOB_DESCRIPTIONS_DIR = Path(
    os.getenv("JOB_DESCRIPTIONS_DIR", PROJECT_ROOT / "job_descriptions")
)
CHROMA_PERSIST_DIR = Path(
    os.getenv("CHROMA_PERSIST_DIR", PROJECT_ROOT / "data" / "chroma_db")
)
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", PROJECT_ROOT / "reports"))

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "chroma").lower()
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SUPPORTED_RESUME_EXTENSIONS = {".txt", ".pdf", ".docx", ".md"}
MAX_FILE_SIZE_MB = 10
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 10
INITIAL_SCREEN_K = 10
DEEP_ANALYSIS_K = 10
COLLECTION_NAME = "resumes"

# Multi-round screening defaults
ROUND1_TOP_K = 10
ROUND2_TOP_K = 5
ROUND3_TOP_K = 3
