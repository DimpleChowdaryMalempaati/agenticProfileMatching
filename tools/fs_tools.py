"""
File System Tools for LLM Function Calling

Provides structured tools for reading, listing, writing, and searching files.
Supports PDF, TXT, and DOCX formats for document parsing.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _extract_text_from_pdf(filepath: Path) -> str:
    """Extract text content from a PDF file."""
    from pypdf import PdfReader

    reader = PdfReader(str(filepath))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _extract_text_from_docx(filepath: Path) -> str:
    """Extract text content from a DOCX file."""
    from docx import Document

    doc = Document(str(filepath))
    return "\n".join(p.text for p in doc.paragraphs if p.text).strip()


def _extract_text_from_txt(filepath: Path) -> str:
    """Extract text content from a plain text file."""
    return filepath.read_text(encoding="utf-8", errors="replace").strip()


_EXTRACTORS = {
    ".pdf": _extract_text_from_pdf,
    ".docx": _extract_text_from_docx,
    ".txt": _extract_text_from_txt,
    ".md": _extract_text_from_txt,
}


def _file_metadata(filepath: Path) -> dict[str, Any]:
    """Build common file metadata."""
    stat = filepath.stat()
    return {
        "name": filepath.name,
        "path": str(filepath.resolve()),
        "extension": filepath.suffix.lower(),
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def read_file(filepath: str) -> dict[str, Any]:
    """
    Read a resume/document file and extract its text content.

    Supports PDF, TXT, DOCX (and MD treated as text).

    Returns:
        dict with keys:
          - success (bool)
          - content (str) — extracted text when successful
          - metadata (dict) — file metadata when successful
          - error (str) — error message when unsuccessful
    """
    try:
        path = Path(filepath)
        if not path.exists():
            return {"success": False, "error": f"File not found: {filepath}"}
        if not path.is_file():
            return {"success": False, "error": f"Not a file: {filepath}"}

        ext = path.suffix.lower()
        extractor = _EXTRACTORS.get(ext)
        if extractor is None:
            supported = ", ".join(sorted(_EXTRACTORS))
            return {
                "success": False,
                "error": f"Unsupported file type '{ext}'. Supported: {supported}",
            }

        content = extractor(path)
        return {
            "success": True,
            "content": content,
            "metadata": _file_metadata(path),
        }
    except Exception as exc:  # noqa: BLE001 — surface any I/O/parse failure cleanly
        return {"success": False, "error": f"Failed to read file: {exc}"}


def list_files(directory: str, extension: Optional[str] = None) -> list[dict[str, Any]]:
    """
    List files in a directory, optionally filtered by extension.

    Args:
        directory: Path to the directory to list.
        extension: Optional filter, e.g. ".pdf" or "pdf".

    Returns:
        List of file metadata dicts (name, path, extension, size_bytes, modified).
        On error, returns a single-item list with an "error" key.
    """
    try:
        path = Path(directory)
        if not path.exists():
            return [{"error": f"Directory not found: {directory}"}]
        if not path.is_dir():
            return [{"error": f"Not a directory: {directory}"}]

        ext_filter: Optional[str] = None
        if extension:
            ext_filter = extension if extension.startswith(".") else f".{extension}"
            ext_filter = ext_filter.lower()

        results: list[dict[str, Any]] = []
        for entry in sorted(path.iterdir()):
            if not entry.is_file():
                continue
            if ext_filter and entry.suffix.lower() != ext_filter:
                continue
            results.append(_file_metadata(entry))
        return results
    except Exception as exc:  # noqa: BLE001
        return [{"error": f"Failed to list files: {exc}"}]


def write_file(filepath: str, content: str) -> dict[str, Any]:
    """
    Write content to a file, creating parent directories if needed.

    Returns:
        dict with success status, path, and bytes written (or error).
    """
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {
            "success": True,
            "path": str(path.resolve()),
            "bytes_written": len(content.encode("utf-8")),
            "message": f"Successfully wrote to {path}",
        }
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": f"Failed to write file: {exc}"}


def search_in_file(filepath: str, keyword: str) -> dict[str, Any]:
    """
    Case-insensitive search for a keyword in a file's content.

    Returns matches with surrounding context (up to ~80 chars each side).

    Returns:
        dict with success, filepath, keyword, match_count, and matches list.
    """
    try:
        if not keyword or not keyword.strip():
            return {"success": False, "error": "Keyword must be a non-empty string"}

        read_result = read_file(filepath)
        if not read_result.get("success"):
            return {
                "success": False,
                "error": read_result.get("error", "Unable to read file"),
            }

        content: str = read_result["content"]
        lower_content = content.lower()
        lower_keyword = keyword.lower()
        keyword_len = len(keyword)

        matches: list[dict[str, Any]] = []
        start = 0
        context_radius = 80

        while True:
            idx = lower_content.find(lower_keyword, start)
            if idx == -1:
                break

            ctx_start = max(0, idx - context_radius)
            ctx_end = min(len(content), idx + keyword_len + context_radius)
            snippet = content[ctx_start:ctx_end]
            if ctx_start > 0:
                snippet = "..." + snippet
            if ctx_end < len(content):
                snippet = snippet + "..."

            matches.append(
                {
                    "position": idx,
                    "matched_text": content[idx : idx + keyword_len],
                    "context": snippet,
                }
            )
            start = idx + keyword_len

        return {
            "success": True,
            "filepath": str(Path(filepath).resolve()),
            "keyword": keyword,
            "match_count": len(matches),
            "matches": matches,
        }
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": f"Search failed: {exc}"}


# ---------------------------------------------------------------------------
# OpenAI / Anthropic-compatible tool schemas for LLM function calling
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a resume or document file (PDF, TXT, DOCX) and extract its text content. "
                "Use this when you need to inspect the contents of a specific file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the file to read",
                    }
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files in a directory with metadata (name, size, modified date). "
                "Optionally filter by file extension such as .pdf or .txt."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Path to the directory to list",
                    },
                    "extension": {
                        "type": "string",
                        "description": "Optional extension filter, e.g. '.pdf' or 'txt'",
                    },
                },
                "required": ["directory"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write text content to a file. Creates parent directories if they do not exist. "
                "Use this to save summaries or notes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path where the file should be written",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to write",
                    },
                },
                "required": ["filepath", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_file",
            "description": (
                "Case-insensitive keyword search inside a file. "
                "Returns each match with surrounding context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the file to search",
                    },
                    "keyword": {
                        "type": "string",
                        "description": "Keyword or phrase to search for",
                    },
                },
                "required": ["filepath", "keyword"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "read_file": read_file,
    "list_files": list_files,
    "write_file": write_file,
    "search_in_file": search_in_file,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> Any:
    """Dispatch a tool call by name with the given arguments."""
    if name not in TOOL_DISPATCH:
        return {"success": False, "error": f"Unknown tool: {name}"}
    return TOOL_DISPATCH[name](**arguments)


if __name__ == "__main__":
    # Quick manual smoke test when run directly
    import json
    import sys

    sample_dir = Path(__file__).parent / "resumes"
    print("=== list_files ===")
    print(json.dumps(list_files(str(sample_dir)), indent=2))

    if len(sys.argv) > 1:
        print("\n=== read_file ===")
        print(json.dumps(read_file(sys.argv[1]), indent=2)[:2000])
