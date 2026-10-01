"""Command-line interface.

Usage:
    python -m app.cli "How many days of annual leave do we get?"
"""
import json
import sys

from agent import ask


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print('Usage: python -m app.cli "your question"')
        sys.exit(1)
    result = ask(question)
    print(f"\nQ: {result['question']}\n")
    print(result["answer"])
    print()
    if result["sources"]:
        print("Sources:")
        for i, s in enumerate(result["sources"], 1):
            print(f"  [{i}] {s['title']} (score {s['score']})")
    print(f"\n(mode: {'Claude' if result['used_llm'] else 'extractive fallback'})")


if __name__ == "__main__":
    main()
