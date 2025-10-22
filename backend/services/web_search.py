"""
DuckDuckGo Web Search Service

Provides web search functionality as fallback when RAG doesn't find relevant results.
"""
import logging
from typing import List, Dict, Optional
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import duckduckgo_search
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logger.warning("duckduckgo_search not installed. Web search will be disabled.")


class WebSearchService:
    """
    Web search service using DuckDuckGo.
    """

    def __init__(self, log_dir: str = "backend/logs", enabled: bool = True, search_prefix: str = ""):
        """
        Initialize web search service.

        Args:
            log_dir: Directory to store search logs
            enabled: Whether web search is enabled
            search_prefix: Prefix to add to all search queries (e.g., persona name)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled and DDGS_AVAILABLE
        self.search_prefix = search_prefix

        if not DDGS_AVAILABLE:
            logger.warning("DuckDuckGo search not available - install with: pip install duckduckgo-search")

    def search(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        Search DuckDuckGo for information.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of search results with title, snippet, and URL
        """
        if not self.enabled:
            logger.info("Web search is disabled")
            return []

        try:
            # Add prefix to query if configured
            full_query = f"{self.search_prefix} {query}" if self.search_prefix else query
            logger.info(f"Searching web for: {full_query}")

            # Perform search
            search_results = DDGS().text(
                full_query,
                max_results=max_results,
                region="us-en",
                safesearch="moderate",
                timelimit=None
            )
            search_results = list(search_results)

            # Log the search
            self._log_search(full_query, search_results)

            # Format results
            results = []
            for i, result in enumerate(search_results):
                results.append({
                    "rank": i + 1,
                    "title": result.get("title", ""),
                    "snippet": result.get("body", ""),
                    "url": result.get("href", ""),
                    "source": "web"
                })

            logger.info(f"Found {len(results)} web results")
            return results

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            self._log_error(query, str(e))
            return []

    def format_results(self, results: List[Dict]) -> str:
        """
        Format web search results into context string for LLM.

        Args:
            results: List of search results

        Returns:
            Formatted context string
        """
        if not results:
            return ""

        context_parts = []
        for r in results:
            title = r["title"]
            snippet = r["snippet"]
            url = r["url"]

            context_parts.append(f"[Web: {title}]\n{snippet}\nSource: {url}")

        return "\n\n---\n\n".join(context_parts)

    def _log_search(self, query: str, results: List[Dict]):
        """Log search query and results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        log_file = self.log_dir / f"{timestamp}_web_search_request.txt"

        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Query: {query}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n\n")
            f.write(json.dumps({"query": query, "max_results": len(results)}, indent=2, ensure_ascii=False))

        # Log response
        response_file = self.log_dir / f"{timestamp}_web_search_response.txt"
        with open(response_file, "w", encoding="utf-8") as f:
            f.write(f"Query: {query}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n\n")
            f.write(json.dumps(results, indent=2, ensure_ascii=False))

    def _log_error(self, query: str, error: str):
        """Log search errors."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        error_file = self.log_dir / f"{timestamp}_web_search_error.txt"

        with open(error_file, "w", encoding="utf-8") as f:
            f.write(f"Query: {query}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Error: {error}\n")
