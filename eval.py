"""Evaluation: retrieval quality + answer faithfulness.

Metrics (all computed against the 15 hand-written Q/A pairs in eval_qa.json):
  - recall@3:    fraction of gold documents found in the top-3 retrieved chunks
  - precision@3: fraction of the top-3 chunks coming from a gold document
  - keyword recall: fraction of the expected answer keywords present in the
    generated answer (a proxy for answer completeness)
  - faithfulness: fraction of answers where every sentence is grounded in
    the retrieved chunks (>= 2 shared content tokens), or the agent
    honestly abstained. This is a simple lexical proxy, not an LLM judge.

Usage:
    python eval.py
Results are printed and saved to eval_results.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import ask  # noqa: E402
from src.chunking import split_sentences  # noqa: E402
from src.llm import content_tokens  # noqa: E402
from src.retriever import Retriever  # noqa: E402

ROOT = Path(__file__).resolve().parent
K = 3


def sentences(text: str) -> list[str]:
    return [s.strip().strip("-[] ") for s in split_sentences(text) if s.strip()]


def is_abstention(text: str) -> bool:
    t = text.lower()
    return "could not find" in t and "document" in t


def faithfulness_pass(answer: str, chunk_texts: list[str]) -> bool:
    """Every substantive sentence must share >=2 content tokens with a chunk."""
    if is_abstention(answer):
        return True  # honest abstention counts as faithful
    chunk_token_sets = [content_tokens(c) for c in chunk_texts]
    for sent in sentences(answer):
        # Skip the boilerplate intro line of the extractive fallback.
        if sent.lower().startswith("based on the company documents"):
            continue
        toks = content_tokens(sent)
        if len(toks) < 2:
            continue  # too short to judge; don't penalise
        if not any(len(toks & cs) >= 2 for cs in chunk_token_sets):
            return False
    return True


def main() -> None:
    qa = json.loads((ROOT / "eval_qa.json").read_text())
    retriever = Retriever()

    rows = []
    for item in qa:
        q = item["question"]
        gold = set(item["gold_docs"])

        hits = retriever.retrieve(q, k=K)
        retrieved_docs = [h.chunk.doc_id for h in hits]
        relevant = sum(1 for d in retrieved_docs if d in gold)
        recall = len(set(retrieved_docs) & gold) / len(gold)
        precision = relevant / K

        result = ask(q)
        answer = result["answer"]
        # Faithfulness is checked against the top-K retrieved chunk texts,
        # which are exactly what the grader and answerer saw.
        chunk_texts = [h.chunk.text for h in hits]

        faithful = faithfulness_pass(answer, chunk_texts)
        kw_hits = sum(1 for kw in item["keywords"] if kw.lower() in answer.lower())
        kw_recall = kw_hits / len(item["keywords"])

        rows.append({
            "id": item["id"],
            "question": q,
            "recall@3": round(recall, 3),
            "precision@3": round(precision, 3),
            "keyword_recall": round(kw_recall, 3),
            "faithful": faithful,
            "used_llm": result["used_llm"],
            "abstained": is_abstention(answer),
        })
        print(f"{item['id']}: recall@3={recall:.2f} precision@3={precision:.2f} "
              f"kw_recall={kw_recall:.2f} faithful={faithful} llm={result['used_llm']}")

    summary = {
        "n": len(rows),
        "recall@3": round(sum(r["recall@3"] for r in rows) / len(rows), 3),
        "precision@3": round(sum(r["precision@3"] for r in rows) / len(rows), 3),
        "keyword_recall": round(sum(r["keyword_recall"] for r in rows) / len(rows), 3),
        "faithfulness": round(sum(r["faithful"] for r in rows) / len(rows), 3),
        "abstentions": sum(r["abstained"] for r in rows),
        "llm_mode": rows[0]["used_llm"] if rows else False,
    }
    print("\n==== SUMMARY ====")
    print(json.dumps(summary, indent=2))

    (ROOT / "eval_results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("\nSaved to eval_results.json")


if __name__ == "__main__":
    main()
