# Pylon Knowledge Base Tools
import json
import logging
import os
from typing import Any

import requests
from langchain.tools import tool

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Pylon API configuration
PYLON_API_BASE_URL = "https://api.usepylon.com"


def _get_kb_id() -> str:
    """Get knowledge base ID from environment."""
    kb_id = os.getenv("PYLON_KB_ID")
    if not kb_id:
        raise ValueError("PYLON_KB_ID not configured")
    return kb_id


def _get_api_key() -> str:
    """Get Pylon API key from environment."""
    api_key = os.getenv("PYLON_API_KEY")
    if not api_key:
        raise ValueError("PYLON_API_KEY not configured")
    return api_key


def _pylon_available() -> bool:
    """Return True if Pylon credentials are configured."""
    return bool(os.getenv("PYLON_API_KEY") and os.getenv("PYLON_KB_ID"))


# =============================================================================
# Cache & API Helpers
# =============================================================================

_articles_cache: list[dict[str, Any]] | None = None
_collections_cache: dict[str, str] | None = None


def _get_headers() -> dict[str, str]:
    """Get API headers with authentication."""
    return {"Authorization": f"Bearer {_get_api_key()}", "Accept": "application/json"}


def _fetch_collections() -> dict[str, str]:
    """Fetch collections from Pylon API and cache them."""
    global _collections_cache

    if _collections_cache is not None:
        return _collections_cache

    kb_id = _get_kb_id()
    url = f"{PYLON_API_BASE_URL}/knowledge-bases/{kb_id}/collections"
    response = requests.get(url, headers=_get_headers(), timeout=30)
    response.raise_for_status()

    collections_data = response.json().get("data", [])

    _collections_cache = {
        coll["title"]: coll["id"]
        for coll in collections_data
        if coll.get("visibility_config", {}).get("visibility") == "public"
    }

    return _collections_cache


def _fetch_all_articles() -> list[dict[str, Any]]:
    """Fetch all articles from Pylon API and cache them."""
    global _articles_cache

    if _articles_cache is not None:
        return _articles_cache

    kb_id = _get_kb_id()
    url = f"{PYLON_API_BASE_URL}/knowledge-bases/{kb_id}/articles"
    response = requests.get(url, headers=_get_headers(), timeout=30)
    response.raise_for_status()

    _articles_cache = response.json().get("data", [])
    return _articles_cache or []


# =============================================================================
# LangChain Tools
# =============================================================================


@tool
def search_support_articles(collections: str = "all") -> str:
    """Get LangChain support article titles from Pylon KB, filtered by collection(s).

    Args:
        collections: Comma-separated list of collection names to filter by.
    """
    if not _pylon_available():
        return json.dumps({"error": "Pylon KB not configured (PYLON_API_KEY / PYLON_KB_ID missing)."}, indent=2)

    try:
        articles = _fetch_all_articles()

        if not articles:
            return json.dumps(
                {
                    "collections": collections,
                    "total": 0,
                    "articles": [],
                    "note": "No articles returned from API",
                },
                indent=2,
            )

        published_articles = []
        for article in articles:
            if (
                article.get("is_published", False)
                and article.get("title")
                and article.get("title") != "Untitled"
                and article.get("visibility_config", {}).get("visibility") == "public"
                and article.get("identifier")
                and article.get("slug")
            ):
                identifier = article.get("identifier")
                slug = article.get("slug")
                support_url = f"https://support.langchain.com/articles/{identifier}-{slug}"

                published_articles.append(
                    {
                        "id": article.get("id"),
                        "title": article.get("title", ""),
                        "url": support_url,
                        "collection_id": article.get("collection_id"),
                    }
                )

        if not published_articles:
            return "No published articles available in the knowledge base."

        try:
            collection_map = _fetch_collections()
        except Exception as e:
            return json.dumps({"error": f"Failed to fetch collections: {e}"}, indent=2)

        if collections.lower() != "all":
            requested_collections = [c.strip() for c in collections.split(",")]
            collection_ids = []
            for coll_name in requested_collections:
                if coll_name in collection_map:
                    collection_ids.append(collection_map[coll_name])
                else:
                    matched = False
                    for key in collection_map.keys():
                        if key.lower() == coll_name.lower():
                            collection_ids.append(collection_map[key])
                            matched = True
                            break
                    if not matched:
                        return json.dumps(
                            {
                                "error": f"Collection '{coll_name}' not found. Available: {', '.join(collection_map.keys())}"
                            },
                            indent=2,
                        )

            published_articles = [
                article
                for article in published_articles
                if article.get("collection_id") in collection_ids
            ]

        collection_id_to_name = {v: k for k, v in collection_map.items()}
        for article in published_articles:
            coll_id = article.get("collection_id")
            article["collection"] = collection_id_to_name.get(coll_id, "Unknown")
            article.pop("collection_id", None)

        return json.dumps({
            "collections": collections,
            "total": len(published_articles),
            "articles": published_articles,
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@tool
def get_article_content(article_id: str) -> str:
    """Fetch the full content of a specific support article.

    Args:
        article_id: The article ID from search_support_articles
    """
    if not _pylon_available():
        return json.dumps({"error": "Pylon KB not configured (PYLON_API_KEY / PYLON_KB_ID missing)."}, indent=2)

    try:
        articles = _fetch_all_articles()

        if not articles:
            return "Error: No articles available."

        for article in articles:
            if article.get("id") == article_id:
                title = article.get("title", "Untitled")
                identifier = article.get("identifier", "")
                slug = article.get("slug", "")
                support_url = (
                    f"https://support.langchain.com/articles/{identifier}-{slug}"
                    if identifier and slug
                    else "N/A"
                )

                return (
                    f"ID: {article.get('id')}\n"
                    f"Title: {title}\n"
                    f"URL: {support_url}\n\n"
                    f"Content:\n"
                    f"{article.get('current_published_content_html', 'No content available')[:5000]}"
                )

        return f"Article ID {article_id} not found."

    except Exception as e:
        return f"Error: {e}"
