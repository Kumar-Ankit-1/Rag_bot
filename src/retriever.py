"""Thin retriever over the persisted FAISS index."""
from dataclasses import dataclass

from .chunking import Chunk
from .config import TOP_K_RETRIEVE
from .vectorstore import load_index, search


@dataclass
class Hit:
    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]


class Retriever:
    def __init__(self):
        self.index, self.chunks = load_index()

    def retrieve(self, query: str, k: int = TOP_K_RETRIEVE) -> list[Hit]:
        return [Hit(chunk=c, score=s) for c, s in search(self.index, self.chunks, query, k)]
