"""Answer generation.

Primary path: Anthropic Claude, when ANTHROPIC_API_KEY is set.
Fallback path: extractive mode. It picks the retrieved sentences with the
highest keyword overlap with the question and presents them as quoted
evidence with [n] citations. The fallback needs no API key, costs nothing,
and (by construction) cannot hallucinate facts that are not in the docs.
"""
import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv

from .chunking import split_sentences
from .config import ANTHROPIC_MODEL
from .retriever import Hit

load_dotenv()

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "by", "as", "at", "from",
    "that", "this", "it", "its", "you", "your", "we", "our", "they", "their",
    "do", "does", "did", "can", "will", "would", "should", "could", "how",
    "what", "when", "where", "which", "who", "whom", "if", "then", "than",
    "so", "such", "no", "not", "all", "any", "each", "per",
}


@dataclass
class Answer:
    text: str
    citations: list[int]  # 1-based indices into the graded chunk list
    used_llm: bool


def _stem(word: str) -> str:
    """Tiny naive stemmer: folds simple English plurals so that e.g.
    'passwords' matches 'password'. Not a real stemmer; kept deliberately
    simple and documented as such."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def content_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {_stem(w) for w in words if w not in STOPWORDS and len(w) > 2}


class Answerer:
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.client = None
        if self.api_key:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)

    @property
    def llm_available(self) -> bool:
        return self.client is not None

    # ---------------- primary: Claude ----------------
    def _claude_answer(self, question: str, hits: list[Hit]) -> Answer:
        context = "\n\n".join(
            f"[{i + 1}] ({h.chunk.title})\n{h.chunk.text}"
            for i, h in enumerate(hits)
        )
        prompt = (
            "Answer the question using ONLY the context below. "
            "Cite every factual claim with the chunk number like [1], [2]. "
            "If the context does not contain the answer, say so explicitly.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        msg = self.client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in msg.content if getattr(block, "type", "") == "text"
        )
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)
                        if 1 <= int(n) <= len(hits)})
        return Answer(text=text.strip(), citations=cited, used_llm=True)

    # ---------------- fallback: extractive ----------------
    def _extractive_answer(self, question: str, hits: list[Hit]) -> Answer:
        q_tokens = content_tokens(question)
        scored: list[tuple[int, str, int]] = []  # (overlap, sentence, hit_idx)
        for i, h in enumerate(hits):
            for sent in split_sentences(h.chunk.text):
                overlap = len(q_tokens & content_tokens(sent))
                if overlap:
                    scored.append((overlap, sent.strip(), i))
        scored.sort(key=lambda t: -t[0])
        seen, picked = set(), []
        for overlap, sent, hit_idx in scored:
            if sent not in seen:
                seen.add(sent)
                picked.append((sent, hit_idx))
            if len(picked) == 3:
                break
        if not picked:
            return Answer(
                text="I could not find relevant information in the company documents.",
                citations=[],
                used_llm=False,
            )
        lines = [f"[{i + 1}] {sent}" for sent, i in picked]
        cited = sorted({i + 1 for _, i in picked})
        text = ("Based on the company documents:\n\n" + "\n".join(f"- {l}" for l in lines))
        return Answer(text=text, citations=cited, used_llm=False)

    def answer(self, question: str, hits: list[Hit]) -> Answer:
        if not hits:
            return Answer(
                text="I could not find this in the company documents.",
                citations=[],
                used_llm=False,
            )
        if self.llm_available:
            try:
                return self._claude_answer(question, hits)
            except Exception as e:  # network/auth failure -> fall back, never crash
                print(f"Claude call failed ({e}); using extractive fallback.")
        return self._extractive_answer(question, hits)
