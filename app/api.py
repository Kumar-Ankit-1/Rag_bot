"""Minimal FastAPI service.

Run with:
    uvicorn app.api:app --reload
Then POST {"question": "..."} to http://localhost:8000/ask
"""
from fastapi import FastAPI
from pydantic import BaseModel

from agent import ask

app = FastAPI(title="Agentic RAG Assistant")


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    return ask(req.question)
