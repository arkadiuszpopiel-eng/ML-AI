"""
RAG Search Tool - allows the agent to search through indexed documents.
"""
from .base import Tool, registry
from ..rag.engine import rag_engine


class RAGSearchTool(Tool):
    name = "search_documents"
    description = (
        "Search through user's indexed documents (PDFs, text files, notes). "
        "Use this when the user asks questions about their documents or uploaded files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to find relevant document fragments.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    async def execute(self, query: str, max_results: int = 5) -> dict:
        results = rag_engine.search(query, top_k=max_results)
        if not results:
            return {
                "success": True,
                "result": "No relevant documents found. The user may need to index some documents first.",
            }

        formatted = []
        for r in results:
            formatted.append(
                f"[{r['filename']}] (score: {r['score']}):\n{r['text'][:500]}"
            )

        return {
            "success": True,
            "result": "\n\n---\n\n".join(formatted),
            "count": len(results),
        }


registry.register(RAGSearchTool())
