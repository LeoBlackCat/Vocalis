"""
RAG (Retrieval-Augmented Generation) Service

Provides document search and context formatting for RAG-enhanced conversations.
"""
import logging
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import json

import numpy as np
from rag_module import (
    load_chunk_records,
    load_doc_records,
    load_embeddings,
    search_chunks,
    format_context_blocks,
    DEFAULT_PROMPTS,
)

logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG service for document retrieval and context formatting.

    Loads chunks, embeddings, and documents on initialization.
    Provides search functionality with fresh retrieval on each query.
    """

    def __init__(
        self,
        chunks_path: str,
        embeddings_path: str,
        docs_path: str,
        top_k: int = 3,
        dataset_name: str = "documents",
        strict_context: bool = True,
        min_score: Optional[float] = None,
        log_dir: str = "backend/logs",
    ):
        """
        Initialize RAG service.

        Args:
            chunks_path: Path to chunks JSONL file
            embeddings_path: Path to embeddings .npy file
            docs_path: Path to docs JSONL file
            top_k: Number of chunks to retrieve per query
            dataset_name: Name of the dataset (for greeting)
            strict_context: Whether to use strict context-only mode
            min_score: Minimum similarity score threshold (lower = more similar)
            log_dir: Directory for logging search requests/responses
        """
        self.top_k = top_k
        self.dataset_name = dataset_name
        self.strict_context = strict_context
        self.min_score = min_score
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        logger.info(f"Loading RAG data from {chunks_path}")
        chunks_path_obj = Path(chunks_path)
        embeddings_path_obj = Path(embeddings_path)
        docs_path_obj = Path(docs_path)

        if not chunks_path_obj.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
        if not embeddings_path_obj.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        if not docs_path_obj.exists():
            raise FileNotFoundError(f"Docs file not found: {docs_path}")

        # Load chunks and embeddings
        self.chunks = load_chunk_records(chunks_path_obj)
        self.embeddings = load_embeddings(embeddings_path_obj)
        self.docs = load_doc_records(docs_path_obj)

        # Build chunk lookup
        self.chunk_lookup: Dict[int, dict] = {}
        for record in self.chunks:
            chunk_id = record.get("id")
            if chunk_id is not None:
                try:
                    self.chunk_lookup[int(chunk_id)] = record
                except (TypeError, ValueError):
                    continue

        logger.info(f"Loaded {len(self.chunks)} chunks, {len(self.embeddings)} embeddings, {len(self.docs)} docs")

        # Validate data
        if self.embeddings.shape[0] != len(self.chunks):
            raise ValueError(
                f"Mismatch: {self.embeddings.shape[0]} embeddings vs {len(self.chunks)} chunks"
            )

    def search(self, query: str) -> List[Dict]:
        """
        Search for relevant chunks.

        This performs a FRESH search on every call, ensuring up-to-date context.

        Args:
            query: User query text

        Returns:
            List of search results with chunk_id, text, score, etc.
        """
        logger.info(f"RAG search for: {query}")

        # Log the search request
        self._log_search_request(query)

        # Perform search
        results = search_chunks(
            query=query,
            records=self.chunks,
            embeddings=self.embeddings,
            top_k=self.top_k,
            include_neighbors=False,
        )

        # Apply score filtering if configured
        if self.min_score is not None:
            filtered_results = []
            for r in results:
                # Note: search_chunks returns cosine similarity (higher = better)
                # But in FAISS L2 distance (lower = better)
                # The score in results is the reranked score (higher = better)
                score = r.get("score", 0.0)
                cosine = r.get("cosine", 0.0)

                # Since we're using FAISS IndexFlatIP (inner product), higher = better
                # So we keep results with score >= threshold
                # But the user config says "lower = more similar" referring to distance
                # Let's use cosine score and invert: keep if (1 - cosine) <= min_score
                # Actually, let's just use score threshold directly
                if cosine >= (1.0 - self.min_score):  # Convert distance to similarity
                    filtered_results.append(r)

            results = filtered_results

        # Log the search response
        self._log_search_response(query, results)

        logger.info(f"Found {len(results)} relevant chunks")
        return results

    def format_context(self, results: List[Dict]) -> str:
        """
        Format search results into context blocks for LLM.

        Args:
            results: List of search results from search()

        Returns:
            Formatted context string
        """
        if not results:
            return ""

        context_blocks = format_context_blocks(results, self.chunk_lookup, self.docs)
        return "".join(context_blocks)

    def get_prompt_template(self) -> str:
        """
        Get the appropriate prompt template based on configuration.

        Returns:
            Prompt template string
        """
        if self.strict_context:
            return DEFAULT_PROMPTS["context_only"]
        else:
            return DEFAULT_PROMPTS["standard"]

    def get_dataset_name(self) -> str:
        """
        Get the dataset name for greeting.

        Returns:
            Dataset name string
        """
        return self.dataset_name

    def is_enabled(self) -> bool:
        """
        Check if RAG is enabled.

        Returns:
            True if RAG service is initialized and ready
        """
        return True

    def _log_search_request(self, query: str):
        """Log RAG search request to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        log_file = self.log_dir / f"{timestamp}_rag_search_request.txt"

        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Query: {query}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n\n")
            f.write(json.dumps({
                "query": query,
                "top_k": self.top_k,
                "min_score": self.min_score,
                "strict_context": self.strict_context,
            }, indent=2, ensure_ascii=False))

    def _log_search_response(self, query: str, results: List[Dict]):
        """Log RAG search response to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        log_file = self.log_dir / f"{timestamp}_rag_search_response.txt"

        # Format results for logging (exclude heavy data like vectors)
        log_results = []
        for r in results:
            log_results.append({
                "chunk_id": r.get("chunk_id"),
                "doc_id": r.get("doc_id"),
                "score": r.get("score"),
                "cosine": r.get("cosine"),
                "text_preview": r.get("text", "")[:200] + "..." if len(r.get("text", "")) > 200 else r.get("text", ""),
            })

        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Query: {query}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Results: {len(results)}\n")
            f.write("=" * 60 + "\n\n")
            f.write(json.dumps(log_results, indent=2, ensure_ascii=False))
