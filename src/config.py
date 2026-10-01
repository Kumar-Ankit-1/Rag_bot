"""Central configuration for the Agentic RAG Assistant."""
import os
from pathlib import Path

# Project root (the folder this file's parent lives in -> src/, so go up one)
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"          # built FAISS index + chunk metadata
INDEX_DIR.mkdir(exist_ok=True)

# Chunking
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 80      # characters of overlap between consecutive chunks

# Embeddings (local model, downloaded once from Hugging Face on first run)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Retrieval
TOP_K_RETRIEVE = 5      # chunks pulled from FAISS before grading
TOP_K_ANSWER = 3        # chunks kept after grading and passed to the answerer
GRADE_MIN_SCORE = 0.25  # cosine-similarity floor; chunks below are dropped

# Generation (Anthropic). Used only when ANTHROPIC_API_KEY is set.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
