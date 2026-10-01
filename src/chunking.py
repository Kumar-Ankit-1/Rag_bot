"""Document loading and chunking.

Reads .md / .txt / .pdf files from the data folder and splits them into
overlapping chunks on sentence boundaries, so no chunk starts or ends
mid-sentence unless the sentence itself is longer than the chunk size.
"""
import re
from dataclasses import dataclass
from pathlib import Path

from .config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR


@dataclass
class Chunk:
    doc_id: str      # e.g. "password-policy" (file stem)
    title: str       # first markdown heading, or the file stem
    text: str
    chunk_index: int


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(data_dir: Path = DATA_DIR) -> list[tuple[str, str, str]]:
    """Return [(doc_id, title, full_text)] for every supported file in data_dir."""
    docs = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in (".md", ".txt"):
            text = _read_text_file(path)
        elif suffix == ".pdf":
            text = _read_pdf(path)
        else:
            continue
        if not text.strip():
            continue
        # Title = first markdown heading, else the file stem.
        title = path.stem
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("#"):
                title = line.lstrip("#").strip()
                break
        docs.append((path.stem, title, text.strip()))
    return docs


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, protecting numbered-list markers ("1.", "2.")
    so they are not mistaken for sentence ends."""
    flat = re.sub(r"\s+", " ", text.strip())
    flat = re.sub(r"\b(\d+)\.\s+", r"\1<DOT> ", flat)  # hide list markers
    parts = re.split(r"(?<=[.!?])\s+", flat)
    out: list[str] = []
    for p in parts:
        p = p.replace("<DOT>", ".").strip()
        if out and re.fullmatch(r"\d+\.", out[-1]):
            out[-1] = f"{out[-1]} {p}".strip()  # re-attach orphaned marker
        elif p:
            out.append(p)
    return out


def _split_sentences(text: str) -> list[str]:
    return split_sentences(text)


def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Greedy sentence packing: fill each chunk up to chunk_size chars,
    then start the next chunk with `overlap` chars of trailing context."""
    sentences = _split_sentences(re.sub(r"\s+", " ", text))
    chunks, current = [], ""
    for sent in sentences:
        candidate = (current + " " + sent).strip() if current else sent
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
                # Carry overlap characters into the next chunk.
                current = current[-overlap:] + " " + sent if overlap else sent
                current = current.strip()
            else:
                # A single over-long sentence: hard-split it.
                chunks.append(sent[:chunk_size])
                current = sent[chunk_size - overlap:].strip() if len(sent) > chunk_size else ""
    if current:
        chunks.append(current)
    return chunks


def _clean_markdown(text: str) -> str:
    """Strip markdown heading/bold markers so quoted chunks read cleanly.

    The document title is stored separately (see load_documents), so the
    first heading line is dropped from the body text entirely.
    """
    lines = text.splitlines()
    if lines and re.match(r"^#{1,6}\s+", lines[0]):
        lines = lines[1:]
    text = "\n".join(lines)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.M)
    return text.replace("**", "")


def build_chunks(docs: list[tuple[str, str, str]]) -> list[Chunk]:
    """Chunk every loaded document, keeping doc_id + title on each chunk."""
    out: list[Chunk] = []
    for doc_id, title, text in docs:
        for i, piece in enumerate(chunk_text(_clean_markdown(text))):
            out.append(Chunk(doc_id=doc_id, title=title, text=piece, chunk_index=i))
    return out
