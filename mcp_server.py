"""MCP server: exposes the document retriever as a tool.

Any MCP-compatible client (Claude Desktop, IDE agents, ...) can call
`search_docs` to run dense retrieval over the company corpus.

Run with:
    python mcp_server.py
Configure the client to launch this script with the `mcp` stdio transport.
"""
try:  # mcp >= 2.x: FastMCP was renamed to MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from src.config import TOP_K_RETRIEVE
from src.retriever import Retriever

mcp = _Server("northwind-docs")
_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


@mcp.tool()
def search_docs(query: str, k: int = TOP_K_RETRIEVE) -> list[dict]:
    """Search the Northwind Traders HR/IT policy corpus.

    Args:
        query: natural-language search query.
        k: number of chunks to return (default 5).
    """
    k = max(1, min(int(k), 10))
    hits = _get_retriever().retrieve(query, k=k)
    return [
        {
            "doc_id": h.chunk.doc_id,
            "title": h.chunk.title,
            "chunk_index": h.chunk.chunk_index,
            "score": round(h.score, 4),
            "text": h.chunk.text,
        }
        for h in hits
    ]


if __name__ == "__main__":
    mcp.run()
