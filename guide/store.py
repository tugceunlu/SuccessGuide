"""Storing course materials (syllabi, objectives, notes) in a searchable vector database."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# A model trained for question-answer retrieval, which suits "How do I pass this course?"-style questions.
EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"
COLLECTION = "course_docs"


@dataclass
class Passage:
    text: str
    source: str
    index: int


def read_text(name: str, data: bytes) -> str:
    suffix = Path(name).suffix.lower()
    if suffix == ".txt":
        for encoding in ("utf-8", "cp1254", "latin-1"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode {name}")
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(data)).pages)
    raise ValueError(f"Unsupported file type: {name} (use .txt or .pdf)")


class CourseLibrary:
    def __init__(self, path: str | None = "data/chroma"):
        self.client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        self.embed = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        self._open()
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)

    def _open(self):
        self.collection = self.client.get_or_create_collection(
            COLLECTION, embedding_function=self.embed, metadata={"hnsw:space": "cosine"}
        )

    def add_file(self, name: str, data: bytes) -> int:
        """Add (or replace) one file. Returns the number of passages stored."""
        texts = [t.strip() for t in self.splitter.split_text(read_text(name, data)) if t.strip()]
        self.collection.delete(where={"source": name})
        if texts:
            self.collection.add(
                ids=[hashlib.sha1(f"{name}:{i}".encode()).hexdigest() for i in range(len(texts))],
                documents=texts,
                metadatas=[{"source": name, "index": i} for i in range(len(texts))],
            )
        return len(texts)

    def add_folder(self, folder: str | Path) -> int:
        return sum(
            self.add_file(p.name, p.read_bytes())
            for p in sorted(Path(folder).glob("*"))
            if p.suffix.lower() in {".txt", ".pdf"}
        )

    def search(self, query: str, k: int = 4) -> list[Passage]:
        if self.collection.count() == 0:
            return []
        res = self.collection.query(query_texts=[query], n_results=min(k, self.collection.count()))
        return [Passage(d, m["source"], m["index"]) for d, m in zip(res["documents"][0], res["metadatas"][0])]

    def sources(self) -> list[str]:
        metas = self.collection.get(include=["metadatas"])["metadatas"] or []
        return sorted({m["source"] for m in metas})

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        self.client.delete_collection(COLLECTION)
        self._open()
