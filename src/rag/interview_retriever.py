"""Interview-aware retriever extending CENTAUR's RAG capabilities.

This module provides specialized retrieval for disaster interview data,
supporting theme-based filtering, tenure comparisons, and respondent lookups.

Usage:
    from src.rag.interview_retriever import InterviewRetriever, get_retriever

    retriever = get_retriever()

    # Search by theme
    results = await retriever.retrieve_by_theme(
        query="financial hardship after hurricane",
        themes=["financial_hardship"],
        tenure_filter="renter",
        k=10,
    )

    # Compare renter vs homeowner responses
    comparison = await retriever.retrieve_comparative(
        query="insurance coverage challenges",
        k_per_group=5,
    )
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from centaur_link import (
    create_vector_store,
    get_document_class,
    get_search_result_class,
    get_embedding_provider,
    ensure_centaur_available,
)
from config.settings import get_settings
from src.rag.interview_metadata import (
    InterviewMetadata,
    InterviewSearchResult,
    TenureType,
)


@dataclass
class RetrievalConfig:
    """Configuration for interview retrieval."""

    k: int = 10
    similarity_threshold: float = 0.5
    tenure_filter: Optional[str] = None  # "renter", "homeowner", or None
    theme_filter: Optional[list[str]] = None
    county_filter: Optional[str] = None
    displaced_only: bool = False
    insured_only: Optional[bool] = None  # True=insured, False=uninsured, None=all


@dataclass
class RetrievalResult:
    """Result from interview retrieval."""

    query: str
    passages: list[InterviewSearchResult] = field(default_factory=list)
    total_found: int = 0
    config: Optional[RetrievalConfig] = None

    @property
    def has_results(self) -> bool:
        """Check if any results were found."""
        return len(self.passages) > 0

    @property
    def top_passage(self) -> Optional[InterviewSearchResult]:
        """Get the top-scoring passage."""
        return self.passages[0] if self.passages else None

    def by_tenure(self) -> dict[str, list[InterviewSearchResult]]:
        """Group results by tenure type."""
        groups: dict[str, list[InterviewSearchResult]] = {
            "renter": [],
            "homeowner": [],
            "unknown": [],
        }
        for passage in self.passages:
            tenure = passage.tenure_type.lower()
            if tenure in groups:
                groups[tenure].append(passage)
            else:
                groups["unknown"].append(passage)
        return groups

    def summarize(self) -> str:
        """Generate a summary of retrieval results."""
        if not self.has_results:
            return f"No results found for: {self.query}"

        tenure_counts = {}
        for p in self.passages:
            t = p.tenure_type
            tenure_counts[t] = tenure_counts.get(t, 0) + 1

        tenure_str = ", ".join(f"{k}: {v}" for k, v in tenure_counts.items())
        return (
            f"Found {len(self.passages)} results for: {self.query}\n"
            f"By tenure: {tenure_str}\n"
            f"Score range: {self.passages[-1].score:.3f} - {self.passages[0].score:.3f}"
        )


class InterviewRetriever:
    """Retriever specialized for disaster interview data.

    Extends CENTAUR's vector search with interview-specific filtering
    and comparison capabilities.
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """Initialize the interview retriever.

        Args:
            index_path: Path to FAISS index. If None, uses settings default.
            embedding_model: Embedding model name. If None, uses settings default.
        """
        self.settings = get_settings()
        self._index_path = index_path or str(self.settings.interviews_index_path)
        self._embedding_model = embedding_model or self.settings.embedding_model

        self._vector_store = None
        self._embedder = None

    @property
    def vector_store(self):
        """Lazy-load the vector store."""
        if self._vector_store is None:
            index_file = Path(self._index_path)
            if index_file.with_suffix(".faiss").exists():
                self._vector_store = create_vector_store(
                    dimensions=self.settings.embedding_dimensions,
                    index_path=str(index_file),
                )
            else:
                # Create empty store if index doesn't exist
                self._vector_store = create_vector_store(
                    dimensions=self.settings.embedding_dimensions,
                )
        return self._vector_store

    @property
    def embedder(self):
        """Lazy-load the embedding provider."""
        if self._embedder is None:
            self._embedder = get_embedding_provider(self._embedding_model)
        return self._embedder

    async def _embed_query(self, query: str) -> list[float]:
        """Generate embedding for a query string."""
        result = await self.embedder.embed([query])
        return result.embeddings[0]

    def _apply_metadata_filter(
        self,
        results: list,
        config: RetrievalConfig,
    ) -> list:
        """Apply metadata filters to search results."""
        filtered = []

        for result in results:
            doc = result.document
            metadata = doc.metadata

            # Tenure filter
            if config.tenure_filter:
                doc_tenure = metadata.get("tenure_type", "").lower()
                if doc_tenure != config.tenure_filter.lower():
                    continue

            # Theme filter (must have high score in at least one theme)
            if config.theme_filter:
                theme_scores = metadata.get("theme_scores", {})
                has_theme = any(
                    theme_scores.get(theme, 0) >= 0.5
                    for theme in config.theme_filter
                )
                if not has_theme:
                    continue

            # County filter
            if config.county_filter:
                doc_county = metadata.get("county", "").lower()
                if doc_county != config.county_filter.lower():
                    continue

            # Displaced filter
            if config.displaced_only:
                if not metadata.get("displaced", False):
                    continue

            # Insurance filter
            if config.insured_only is not None:
                doc_insured = metadata.get("insured", False)
                if doc_insured != config.insured_only:
                    continue

            # Passed all filters
            filtered.append(result)

        return filtered

    def _convert_to_interview_results(
        self,
        results: list,
    ) -> list[InterviewSearchResult]:
        """Convert CENTAUR search results to interview results."""
        interview_results = []

        for i, result in enumerate(results):
            doc = result.document
            metadata = InterviewMetadata.from_dict(doc.metadata)

            interview_result = InterviewSearchResult(
                content=doc.content,
                score=result.score,
                rank=i + 1,
                metadata=metadata,
            )
            interview_results.append(interview_result)

        return interview_results

    async def retrieve(
        self,
        query: str,
        config: Optional[RetrievalConfig] = None,
    ) -> RetrievalResult:
        """Retrieve interview passages matching a query.

        Args:
            query: Search query string.
            config: Optional retrieval configuration.

        Returns:
            RetrievalResult with matching passages.
        """
        config = config or RetrievalConfig()

        # Generate query embedding
        query_embedding = await self._embed_query(query)

        # Search vector store (retrieve more than k to allow for filtering)
        k_oversample = config.k * 3 if (config.tenure_filter or config.theme_filter) else config.k
        raw_results = self.vector_store.search(query_embedding, k=k_oversample)

        # Apply metadata filters
        filtered_results = self._apply_metadata_filter(raw_results, config)

        # Apply similarity threshold
        filtered_results = [
            r for r in filtered_results
            if r.score >= config.similarity_threshold
        ]

        # Limit to k
        filtered_results = filtered_results[:config.k]

        # Convert to interview results
        interview_results = self._convert_to_interview_results(filtered_results)

        return RetrievalResult(
            query=query,
            passages=interview_results,
            total_found=len(interview_results),
            config=config,
        )

    async def retrieve_by_theme(
        self,
        query: str,
        themes: list[str],
        tenure_filter: Optional[str] = None,
        k: int = 10,
    ) -> RetrievalResult:
        """Retrieve passages relevant to specific themes.

        Args:
            query: Search query string.
            themes: List of themes to filter by (e.g., ["financial_hardship"]).
            tenure_filter: Optional tenure filter ("renter" or "homeowner").
            k: Number of results to return.

        Returns:
            RetrievalResult with theme-filtered passages.
        """
        config = RetrievalConfig(
            k=k,
            theme_filter=themes,
            tenure_filter=tenure_filter,
        )
        return await self.retrieve(query, config)

    async def retrieve_comparative(
        self,
        query: str,
        k_per_group: int = 5,
        theme_filter: Optional[list[str]] = None,
    ) -> dict[str, RetrievalResult]:
        """Retrieve matching passages split by renter vs homeowner.

        Useful for comparing how different tenure types discuss similar topics.

        Args:
            query: Search query string.
            k_per_group: Number of results per tenure group.
            theme_filter: Optional theme filter to apply.

        Returns:
            Dictionary with "renter" and "homeowner" RetrievalResults.
        """
        renter_config = RetrievalConfig(
            k=k_per_group,
            tenure_filter="renter",
            theme_filter=theme_filter,
        )
        homeowner_config = RetrievalConfig(
            k=k_per_group,
            tenure_filter="homeowner",
            theme_filter=theme_filter,
        )

        renter_results = await self.retrieve(query, renter_config)
        homeowner_results = await self.retrieve(query, homeowner_config)

        return {
            "renter": renter_results,
            "homeowner": homeowner_results,
        }

    async def retrieve_by_respondent(
        self,
        respondent_id: str,
    ) -> RetrievalResult:
        """Retrieve all passages from a specific respondent.

        Args:
            respondent_id: The respondent ID to look up.

        Returns:
            RetrievalResult with all passages from the respondent.
        """
        # Use respondent ID as query and filter by metadata
        # This is a specialized search that looks for exact ID match

        # Get all documents (brute force for now - could optimize with ID index)
        all_docs = self.vector_store.get_all_documents() if hasattr(self.vector_store, 'get_all_documents') else []

        matching = []
        for doc in all_docs:
            if doc.metadata.get("respondent_id") == respondent_id:
                matching.append(InterviewSearchResult(
                    content=doc.content,
                    score=1.0,  # Exact match
                    rank=len(matching) + 1,
                    metadata=InterviewMetadata.from_dict(doc.metadata),
                ))

        return RetrievalResult(
            query=f"respondent:{respondent_id}",
            passages=matching,
            total_found=len(matching),
        )

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the indexed interviews.

        Returns:
            Dictionary with index statistics.
        """
        # This would query the vector store for statistics
        return {
            "total_documents": getattr(self.vector_store, 'size', 0),
            "embedding_dimensions": self.settings.embedding_dimensions,
            "embedding_model": self._embedding_model,
            "index_path": self._index_path,
        }


# =============================================================================
# Factory Function
# =============================================================================

_retriever_instance: Optional[InterviewRetriever] = None


def get_retriever(
    index_path: Optional[str] = None,
    force_new: bool = False,
) -> InterviewRetriever:
    """Get or create an interview retriever instance.

    Args:
        index_path: Optional path to FAISS index.
        force_new: If True, create a new instance even if one exists.

    Returns:
        InterviewRetriever instance.
    """
    global _retriever_instance

    if force_new or _retriever_instance is None:
        _retriever_instance = InterviewRetriever(index_path=index_path)

    return _retriever_instance


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Classes
    "InterviewRetriever",
    "RetrievalConfig",
    "RetrievalResult",

    # Factory
    "get_retriever",
]
