"""RAG pipeline: chunk resumes, embed, and store in ChromaDB with metadata."""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from config import (
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    OPENAI_API_KEY,
    RESUMES_DIR,
)
from tools.resume_loader import load_all_resumes


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    section: str
    resume_path: str
    candidate_name: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


class EmbeddingProvider:
    """Unified embedding interface for ChromaDB ONNX, SentenceTransformers, or OpenAI."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
    ):
        self.provider = (provider or EMBEDDING_PROVIDER).lower()
        self.model = model or EMBEDDING_MODEL
        self._model = None
        self._openai_client = None
        self._chroma_ef = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.provider == "openai":
            return self._embed_openai(texts)
        if self.provider == "sentence-transformers":
            return self._embed_sentence_transformers(texts)
        return self._embed_chroma_default(texts)

    def _embed_chroma_default(self, texts: list[str]) -> list[list[float]]:
        """Use ChromaDB's bundled ONNX MiniLM model (no PyTorch required)."""
        if self._chroma_ef is None:
            from chromadb.utils import embedding_functions

            self._chroma_ef = embedding_functions.DefaultEmbeddingFunction()
        return self._chroma_ef(texts)

    def _embed_sentence_transformers(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "Install sentence-transformers for this provider: "
                    "pip install sentence-transformers"
                ) from exc

            self._model = SentenceTransformer(self.model)
        vectors = self._model.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai"
            )
        if self._openai_client is None:
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=OPENAI_API_KEY)

        model = "text-embedding-3-small" if "embedding" not in self.model else self.model
        response = self._openai_client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in response.data]


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_document(document: dict[str, Any]) -> list[TextChunk]:
    """Chunk a resume document, preserving section boundaries when possible."""
    chunks: list[TextChunk] = []
    sections = document.get("sections") or {"full_text": document.get("text", "")}
    meta = document.get("metadata", {})
    candidate_name = meta.get("name", "Unknown")
    resume_path = document.get("path", "")

    global_index = 0
    for section_name, section_text in sections.items():
        if not section_text or not section_text.strip():
            continue
        section_chunks = _split_text(section_text, CHUNK_SIZE, CHUNK_OVERLAP)
        for local_idx, chunk_text in enumerate(section_chunks):
            chunk_id = hashlib.md5(
                f"{resume_path}:{section_name}:{global_index}".encode()
            ).hexdigest()
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    section=section_name,
                    resume_path=resume_path,
                    candidate_name=candidate_name,
                    chunk_index=global_index,
                    metadata={
                        "skills": meta.get("skills", []),
                        "experience_years": meta.get("experience_years", 0),
                        "education": meta.get("education", []),
                        "section": section_name,
                        "local_chunk_index": local_idx,
                    },
                )
            )
            global_index += 1
    return chunks


class ResumeRAG:
    """End-to-end RAG system for resume indexing and retrieval."""

    def __init__(
        self,
        persist_dir: Path | str | None = None,
        collection_name: str = COLLECTION_NAME,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self.persist_dir = Path(persist_dir or CHROMA_PERSIST_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedder = embedding_provider or EmbeddingProvider()
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self):
        return self._collection

    def reset_index(self) -> dict[str, Any]:
        """Delete and recreate the vector collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return {"success": True, "message": "Index reset complete"}

    def index_resumes(
        self,
        directory: str | Path | None = None,
        batch_size: int = 32,
    ) -> dict[str, Any]:
        """Load resumes, chunk, embed, and store in ChromaDB.

        Example:
            >>> rag = ResumeRAG()
            >>> stats = rag.index_resumes("resumes")
            >>> print(stats["chunks_indexed"])
        """
        start = time.perf_counter()
        load_result = load_all_resumes(directory or RESUMES_DIR)

        if load_result.get("loaded_count", 0) == 0:
            return {
                "success": False,
                "error": {
                    "message": "No resumes loaded",
                    "code": "NO_DOCUMENTS",
                    "failures": load_result.get("failures", []),
                },
            }

        all_chunks: list[TextChunk] = []
        for doc in load_result.get("documents", []):
            all_chunks.extend(chunk_document(doc))

        if not all_chunks:
            return {
                "success": False,
                "error": {"message": "No chunks produced", "code": "NO_CHUNKS"},
            }

        indexed = 0
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self.embedder.embed(texts)

            ids = [c.chunk_id for c in batch]
            metadatas = []
            for c in batch:
                metadatas.append(
                    {
                        "resume_path": c.resume_path,
                        "candidate_name": c.candidate_name,
                        "section": c.section,
                        "chunk_index": c.chunk_index,
                        "skills": json.dumps(c.metadata.get("skills", [])),
                        "experience_years": float(
                            c.metadata.get("experience_years", 0)
                        ),
                        "education": json.dumps(c.metadata.get("education", [])),
                    }
                )

            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            indexed += len(batch)

        elapsed = time.perf_counter() - start
        return {
            "success": True,
            "documents_indexed": load_result.get("loaded_count", 0),
            "chunks_indexed": indexed,
            "failed_count": load_result.get("failed_count", 0),
            "failures": load_result.get("failures", []),
            "latency_seconds": round(elapsed, 3),
            "collection": self.collection_name,
            "persist_dir": str(self.persist_dir),
        }

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Semantic search over indexed resume chunks.

        Example:
            >>> rag = ResumeRAG()
            >>> results = rag.query("Python machine learning engineer", top_k=5)
        """
        if not query_text.strip():
            return {
                "success": False,
                "error": {"message": "Query text is empty", "code": "EMPTY_QUERY"},
            }

        start = time.perf_counter()
        query_embedding = self.embedder.embed([query_text])[0]

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        try:
            raw = self._collection.query(**kwargs)
        except Exception as exc:
            return {
                "success": False,
                "error": {"message": str(exc), "code": "QUERY_FAILED"},
            }

        results = []
        if raw and raw.get("ids") and raw["ids"][0]:
            for idx, chunk_id in enumerate(raw["ids"][0]):
                distance = raw["distances"][0][idx] if raw.get("distances") else 0
                similarity = max(0.0, 1.0 - distance)
                meta = raw["metadatas"][0][idx] if raw.get("metadatas") else {}
                results.append(
                    {
                        "chunk_id": chunk_id,
                        "text": raw["documents"][0][idx] if raw.get("documents") else "",
                        "similarity": round(similarity, 4),
                        "metadata": meta,
                    }
                )

        elapsed = time.perf_counter() - start
        return {
            "success": True,
            "query": query_text,
            "count": len(results),
            "results": results,
            "latency_seconds": round(elapsed, 3),
        }

    def get_stats(self) -> dict[str, Any]:
        """Return collection statistics."""
        count = self._collection.count()
        return {
            "success": True,
            "collection": self.collection_name,
            "chunk_count": count,
            "persist_dir": str(self.persist_dir),
            "embedding_provider": self.embedder.provider,
            "embedding_model": self.embedder.model,
        }


def build_index(resumes_dir: str | Path | None = None) -> dict[str, Any]:
    """Convenience function to build the resume index from scratch."""
    rag = ResumeRAG()
    rag.reset_index()
    return rag.index_resumes(resumes_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build resume RAG index")
    parser.add_argument(
        "--resumes-dir",
        default=str(RESUMES_DIR),
        help="Directory containing resume files",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset existing index before indexing",
    )
    args = parser.parse_args()

    rag = ResumeRAG()
    if args.reset:
        rag.reset_index()
    result = rag.index_resumes(args.resumes_dir)
    print(json.dumps(result, indent=2))
