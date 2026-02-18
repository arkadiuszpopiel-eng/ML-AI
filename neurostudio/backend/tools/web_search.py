"""
Web Search tool - searches the internet using DuckDuckGo (no API key needed).
"""
from typing import Any

from .base import Tool, registry
from ..config import load_config


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the internet using DuckDuckGo. "
        "Returns a list of results with titles, URLs, and snippets. "
        "Useful for finding current information, documentation, tutorials, etc."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default: 5, max: 10).",
            },
        },
        "required": ["query"],
    }

    async def execute(self, **kwargs) -> dict[str, Any]:
        query = kwargs["query"]

        config = load_config()
        ws_config = config.get("tools", {}).get("web_search", {})
        max_results = min(kwargs.get("max_results", ws_config.get("max_results", 5)), 10)

        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            formatted = []
            for r in results:
                formatted.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                })

            return {"success": True, "result": formatted, "count": len(formatted), "query": query}
        except ImportError:
            return {"success": False, "error": "duckduckgo-search package not installed. Run: pip install duckduckgo-search"}
        except Exception as e:
            return {"success": False, "error": f"Search failed: {str(e)}"}


registry.register(WebSearchTool())
