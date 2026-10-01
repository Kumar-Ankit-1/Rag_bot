"""One-shot ingestion: data/ -> chunks -> embeddings -> FAISS index.

Usage:
    python ingest.py
"""
from src.chunking import build_chunks, load_documents
from src.vectorstore import build_index


def main() -> None:
    docs = load_documents()
    print(f"Loaded {len(docs)} documents.")
    chunks = build_chunks(docs)
    print(f"Built {len(chunks)} chunks.")
    build_index(chunks)


if __name__ == "__main__":
    main()
