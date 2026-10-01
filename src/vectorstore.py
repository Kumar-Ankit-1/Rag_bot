"""FAISS vector store: build once, load fast afterwards."""
import pickle

import faiss
import numpy as np

from .chunking import Chunk
from .config import INDEX_DIR
from .embeddings import embed_query, embed_texts

INDEX_FILE = INDEX_DIR / "index.faiss"
META_FILE = INDEX_DIR / "chunks.pkl"


def build_index(chunks: list[Chunk]) -> None:
    """Embed all chunks and persist a FAISS index + chunk metadata."""
    vectors = embed_texts([c.text for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])  # exact inner-product (= cosine) search
    index.add(vectors)
    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Indexed {len(chunks)} chunks -> {INDEX_FILE}")


def load_index() -> tuple[faiss.Index, list[Chunk]]:
    if not INDEX_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError(
            "No index found. Run `python ingest.py` first to build it."
        )
    index = faiss.read_index(str(INDEX_FILE))
    with open(META_FILE, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def search(index: faiss.Index, chunks: list[Chunk],
           query: str, k: int) -> list[tuple[Chunk, float]]:
    """Return [(chunk, cosine_score)] for the top-k chunks, best first."""
    q = embed_query(query).reshape(1, -1)
    scores, ids = index.search(q, k)
    return [(chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
