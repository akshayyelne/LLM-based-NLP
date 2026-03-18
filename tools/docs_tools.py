"""LangChain Documentation Search Tool using DuckDuckGo."""
import logging
from langchain.tools import tool

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)


@tool
def SearchDocsByLangChain(query: str) -> str:
    """Search LangChain, LangGraph, and LangSmith documentation.

    Args:
        query: Natural language search query about LangChain, LangGraph, or LangSmith.

    Returns:
        Formatted results with title, link, and content for each match.
    """
    try:
        search_query = f"LangChain {query}"
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=4))

        if not results:
            return (
                f"No documentation results found for '{query}'. "
                "Please visit https://python.langchain.com for the latest docs."
            )

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"Result {i}:\n"
                f"Title: {r.get('title', 'Untitled')}\n"
                f"Link: {r.get('href', 'N/A')}\n"
                f"Content: {r.get('body', 'No content')}\n"
            )

        return "\n---\n\n".join(formatted)

    except Exception as e:
        logger.error(f"Documentation search failed: {e}")
        return (
            f"Documentation search is temporarily unavailable. "
            "Please visit https://python.langchain.com directly."
        )
