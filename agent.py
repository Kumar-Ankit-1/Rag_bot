"""The agent, built with LangGraph.

Pipeline:
    router  ->  (smalltalk | retrieve)  ->  grade  ->  answer  -> END

- router:    small-talk ("hi", "thanks", ...) is answered directly;
             anything else goes to retrieval.
- retrieve:  top-k dense retrieval from the FAISS index.
- grade:     heuristic re-ranker. Keeps at most TOP_K_ANSWER chunks whose
             cosine score clears GRADE_MIN_SCORE. (A natural upgrade is an
             LLM-as-judge grader; the interface already supports swapping
             the scoring function.)
- answer:    Claude with citations when ANTHROPIC_API_KEY is set, otherwise
             the extractive fallback. If nothing survives grading the agent
             abstains instead of hallucinating.

Run it directly:
    python agent.py "How many days of annual leave do we get?"
"""
import re
import sys
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.config import GRADE_MIN_SCORE, TOP_K_ANSWER, TOP_K_RETRIEVE
from src.llm import Answerer
from src.retriever import Hit, Retriever

# Lazily created so `import agent` stays cheap and testable.
_retriever: Retriever | None = None
_answerer: Answerer | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def get_answerer() -> Answerer:
    global _answerer
    if _answerer is None:
        _answerer = Answerer()
    return _answerer


# ---------------------------------------------------------------- state
class AgentState(TypedDict):
    question: str
    route: str                 # "smalltalk" or "docs"
    hits: list                 # list[Hit] from retrieval
    graded_hits: list          # list[Hit] after grading
    answer: str
    citations: list            # 1-based indices into graded_hits
    used_llm: bool


# ---------------------------------------------------------------- nodes
SMALLTALK_PATTERNS = [
    r"^(hi|hello|hey|yo|good\s?(morning|afternoon|evening))\b",
    r"\b(thank|thanks|thx)\b",
    r"^(bye|goodbye|see you)\b",
    r"how are you",
    r"who are you",
]


def router_node(state: AgentState) -> AgentState:
    """Decide whether the question needs the document corpus at all."""
    q = state["question"].strip().lower()
    if any(re.search(p, q) for p in SMALLTALK_PATTERNS):
        state["route"] = "smalltalk"
    else:
        state["route"] = "docs"
    return state


def smalltalk_node(state: AgentState) -> AgentState:
    state["answer"] = (
        "Hi! I'm the Northwind Traders document assistant. "
        "Ask me anything about our HR and IT policies."
    )
    state["citations"] = []
    state["used_llm"] = False
    return state


def retrieve_node(state: AgentState) -> AgentState:
    state["hits"] = get_retriever().retrieve(state["question"], k=TOP_K_RETRIEVE)
    return state


def grade_node(state: AgentState) -> AgentState:
    """Heuristic re-ranker: score floor + keep the best TOP_K_ANSWER.

    The raw cosine score already ranks by relevance; grading just drops
    weak matches so the answerer never sees off-topic context.
    """
    graded = [h for h in state["hits"] if h.score >= GRADE_MIN_SCORE]
    state["graded_hits"] = graded[:TOP_K_ANSWER]
    return state


def answer_node(state: AgentState) -> AgentState:
    ans = get_answerer().answer(state["question"], state["graded_hits"])
    state["answer"] = ans.text
    state["citations"] = ans.citations
    state["used_llm"] = ans.used_llm
    return state


# ---------------------------------------------------------------- graph
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("smalltalk", smalltalk_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        lambda s: s["route"],
        {"smalltalk": "smalltalk", "docs": "retrieve"},
    )
    graph.add_edge("smalltalk", END)
    graph.add_edge("retrieve", "grade")
    graph.add_edge("grade", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def ask(question: str) -> dict:
    """Run the agent end to end and return a plain dict (CLI/API friendly)."""
    result = build_graph().invoke({"question": question})
    graded = result.get("graded_hits", [])
    return {
        "question": question,
        "route": result.get("route"),
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "used_llm": result.get("used_llm", False),
        "sources": [
            {"doc_id": h.chunk.doc_id, "title": h.chunk.title, "score": round(h.score, 3)}
            for h in graded
        ],
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How many days of annual leave do we get?"
    import json
    print(json.dumps(ask(q), indent=2))
