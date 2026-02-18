"""
Web Fetch tool - downloads and extracts text content from web pages.
"""
from typing import Any

from .base import Tool, registry
from ..config import load_config


class WebFetchTool(Tool):
    name = "web_fetch"
    description = (
        "Fetch a web page and extract its text content. "
        "Useful for reading documentation, articles, code repositories, etc. "
        "Returns the main text content of the page."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL of the web page to fetch.",
            },
            "extract_links": {
                "type": "boolean",
                "description": "If true, also extract all links from the page (default: false).",
            },
        },
        "required": ["url"],
    }

    async def execute(self, **kwargs) -> dict[str, Any]:
        import httpx

        url = kwargs["url"]
        extract_links = kwargs.get("extract_links", False)

        config = load_config()
        wf_config = config.get("tools", {}).get("web_fetch", {})
        max_size = wf_config.get("max_size", 1_048_576)

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                resp = await client.get(url, headers={"User-Agent": "NeuroForge/0.1"})
                resp.raise_for_status()

                content_type = resp.headers.get("content-type", "")
                if "text" not in content_type and "json" not in content_type and "xml" not in content_type:
                    return {
                        "success": True,
                        "result": f"Binary content ({content_type}), size: {len(resp.content)} bytes",
                        "url": str(resp.url),
                    }

                text = resp.text[:max_size]

            # Extract readable text from HTML
            if "html" in content_type:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(text, "html.parser")

                # Remove script and style elements
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()

                main_text = soup.get_text(separator="\n", strip=True)
                # Collapse multiple newlines
                lines = [line.strip() for line in main_text.split("\n") if line.strip()]
                main_text = "\n".join(lines)[:max_size]

                result = {"success": True, "result": main_text, "url": str(resp.url)}

                if extract_links:
                    links = []
                    for a in soup.find_all("a", href=True):
                        href = a["href"]
                        link_text = a.get_text(strip=True)[:100]
                        if href.startswith(("http://", "https://")):
                            links.append({"text": link_text, "url": href})
                    result["links"] = links[:50]

                return result
            else:
                return {"success": True, "result": text[:max_size], "url": str(resp.url)}

        except httpx.HTTPStatusError as e:
            return {"success": False, "error": f"HTTP {e.response.status_code}: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Fetch failed: {str(e)}"}


registry.register(WebFetchTool())
