"""Local embeddings via sentence-transformers.

No API key needed: the model runs on CPU and is downloaded once from
Hugging Face on first use, then cached.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return L2-normalised embeddings, shape (n, dim), dtype float32.

    Normalised vectors turn FAISS inner-product search into cosine search.
    """
    model = get_model()
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (emb / norms).astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
