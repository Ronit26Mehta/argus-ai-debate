"""
arXiv API Connector for ARGUS.

Fetches academic papers from arXiv with full metadata.

Example:
    >>> connector = ArxivConnector()
    >>> result = connector.fetch("machine learning transformers")
    >>> 
    >>> for doc in result.documents:
    ...     print(f"{doc.title} - {doc.metadata['authors']}")
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import Field

from argus.knowledge.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectorResult,
)
from argus.core.models import Document, SourceType

logger = logging.getLogger(__name__)


@dataclass
class ArxivPaper:
    """Represents an arXiv paper with metadata."""
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    published: datetime
    updated: datetime
    doi: Optional[str] = None
    pdf_url: Optional[str] = None
    comment: Optional[str] = None
    journal_ref: Optional[str] = None
    
    def to_document(self) -> Document:
        """Convert to ARGUS Document."""
        content = f"# {self.title}\n\n"
        content += f"**Authors:** {', '.join(self.authors)}\n\n"
        content += f"**Published:** {self.published.strftime('%Y-%m-%d')}\n\n"
        content += f"**Categories:** {', '.join(self.categories)}\n\n"
        if self.doi:
            content += f"**DOI:** {self.doi}\n\n"
        content += f"## Abstract\n\n{self.abstract}\n"
        
        return Document(
            url=f"https://arxiv.org/abs/{self.arxiv_id}",
            title=self.title,
            content=content,
            source_type=SourceType.TEXT,
            metadata={
                "arxiv_id": self.arxiv_id,
                "authors": self.authors,
                "categories": self.categories,
                "primary_category": self.primary_category,
                "published": self.published.isoformat(),
                "updated": self.updated.isoformat(),
                "doi": self.doi,
                "pdf_url": self.pdf_url,
                "journal_ref": self.journal_ref,
                "comment": self.comment,
                "source": "arxiv",
            },
        )


class ArxivConnectorConfig(ConnectorConfig):
    """Configuration for arXiv connector.
    
    Attributes:
        sort_by: Sort results by relevance, lastUpdatedDate, or submittedDate
        sort_order: ascending or descending
        include_abstract: Include full abstract in results
    """
    sort_by: str = Field(
        default="relevance",
        description="Sort by: relevance, lastUpdatedDate, or submittedDate",
    )
    sort_order: str = Field(
        default="descending",
        description="Sort order: ascending or descending",
    )
    include_abstract: bool = Field(
        default=True,
        description="Include full abstract in results",
    )


class ArxivConnector(BaseConnector):
    """Connector for fetching papers from arXiv.
    
    Uses the arXiv API to search for and retrieve academic papers.
    Returns papers as Document objects with full metadata.
    
    Example:
        >>> connector = ArxivConnector()
        >>> result = connector.fetch("quantum computing", max_results=5)
        >>> 
        >>> if result.success:
        ...     for doc in result.documents:
        ...         print(doc.title)
        
    Supported Query Features:
        - Full-text search: "machine learning"
        - Author search: "au:Einstein"
        - Title search: "ti:quantum"
        - Abstract search: "abs:neural network"
        - Category search: "cat:cs.AI"
        - Combined: "au:LeCun AND cat:cs.LG"
    """
    
    name = "arxiv"
    description = "Fetch academic papers from arXiv"
    version = "1.0.0"
    
    # arXiv category mappings for convenience
    CATEGORIES = {
        "ai": "cs.AI",
        "ml": "cs.LG",
        "cv": "cs.CV",
        "nlp": "cs.CL",
        "ir": "cs.IR",
        "physics": "physics",
        "math": "math",
        "stats": "stat",
        "qc": "quant-ph",
        "bio": "q-bio",
    }
    
    def __init__(self, config: Optional[ArxivConnectorConfig] = None):
        """Initialize arXiv connector.
        
        Args:
            config: Connector configuration
        """
        super().__init__(config or ArxivConnectorConfig())
        self._client = None
    
    @property
    def arxiv_config(self) -> ArxivConnectorConfig:
        """Get typed config."""
        return self.config  # type: ignore
    
    def _get_client(self):
        """Get or create arxiv client."""
        if self._client is None:
            try:
                import arxiv
                self._client = arxiv.Client(
                    page_size=100,
                    delay_seconds=3.0,  # Be nice to arXiv
                    num_retries=3,
                )
            except ImportError:
                raise ImportError(
                    "arxiv package is required. Install with: pip install arxiv"
                )
        return self._client
    
    def _build_query(
        self,
        query: str,
        categories: Optional[list[str]] = None,
        authors: Optional[list[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> str:
        """Build arXiv search query.
        
        Args:
            query: Base search query
            categories: Filter by categories
            authors: Filter by authors
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            
        Returns:
            Formatted arXiv query string
        """
        parts = []
        
        # Add main query (search in all fields if no prefix specified)
        if query:
            if not any(
                query.lower().startswith(p) 
                for p in ["au:", "ti:", "abs:", "cat:", "all:"]
            ):
                # Default to searching all fields
                parts.append(f"all:{query}")
            else:
                parts.append(query)
        
        # Add category filters
        if categories:
            cat_parts = []
            for cat in categories:
                # Resolve category aliases
                resolved = self.CATEGORIES.get(cat.lower(), cat)
                cat_parts.append(f"cat:{resolved}")
            if len(cat_parts) > 1:
                parts.append(f"({' OR '.join(cat_parts)})")
            else:
                parts.append(cat_parts[0])
        
        # Add author filters
        if authors:
            author_parts = [f"au:{author}" for author in authors]
            if len(author_parts) > 1:
                parts.append(f"({' OR '.join(author_parts)})")
            else:
                parts.append(author_parts[0])
        
        return " AND ".join(parts) if parts else "all:*"
    
    def _parse_paper(self, result: Any) -> ArxivPaper:
        """Parse arxiv result into ArxivPaper.
        
        Args:
            result: arxiv.Result object
            
        Returns:
            ArxivPaper dataclass
        """
        # Extract arXiv ID from entry_id URL
        arxiv_id = result.entry_id.split("/abs/")[-1]
        
        return ArxivPaper(
            arxiv_id=arxiv_id,
            title=result.title.replace("\n", " ").strip(),
            abstract=result.summary.strip() if self.arxiv_config.include_abstract else "",
            authors=[author.name for author in result.authors],
            categories=result.categories,
            primary_category=result.primary_category,
            published=result.published,
            updated=result.updated,
            doi=result.doi,
            pdf_url=result.pdf_url,
            comment=result.comment,
            journal_ref=result.journal_ref,
        )
    
    def fetch(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> ConnectorResult:
        """Fetch papers from arXiv.
        
        Args:
            query: Search query (supports arXiv query syntax)
            max_results: Maximum number of papers to return
            **kwargs: Additional options
                - categories: List of category filters
                - authors: List of author filters
                - date_from: Start date (YYYY-MM-DD)
                - date_to: End date (YYYY-MM-DD)
                - id_list: List of specific arXiv IDs to fetch
            
        Returns:
            ConnectorResult with paper documents
        """
        try:
            import arxiv
        except ImportError:
            return ConnectorResult.from_error(
                "arxiv package not installed. Run: pip install arxiv"
            )
        
        try:
            client = self._get_client()
            
            # Check for ID list (direct fetch by ID)
            id_list = kwargs.get("id_list", [])
            
            if id_list:
                # Fetch specific papers by ID
                search = arxiv.Search(
                    id_list=id_list,
                    max_results=len(id_list),
                )
            else:
                # Build search query
                full_query = self._build_query(
                    query=query,
                    categories=kwargs.get("categories"),
                    authors=kwargs.get("authors"),
                    date_from=kwargs.get("date_from"),
                    date_to=kwargs.get("date_to"),
                )
                
                # Determine sort criteria
                sort_by_map = {
                    "relevance": arxiv.SortCriterion.Relevance,
                    "lastupdateddate": arxiv.SortCriterion.LastUpdatedDate,
                    "submitteddate": arxiv.SortCriterion.SubmittedDate,
                }
                sort_by = sort_by_map.get(
                    self.arxiv_config.sort_by.lower(),
                    arxiv.SortCriterion.Relevance
                )
                
                sort_order_map = {
                    "ascending": arxiv.SortOrder.Ascending,
                    "descending": arxiv.SortOrder.Descending,
                }
                sort_order = sort_order_map.get(
                    self.arxiv_config.sort_order.lower(),
                    arxiv.SortOrder.Descending
                )
                
                search = arxiv.Search(
                    query=full_query,
                    max_results=max_results,
                    sort_by=sort_by,
                    sort_order=sort_order,
                )
            
            # Execute search
            papers: list[ArxivPaper] = []
            documents: list[Document] = []
            
            for result in client.results(search):
                paper = self._parse_paper(result)
                papers.append(paper)
                documents.append(paper.to_document())
                
                if len(papers) >= max_results:
                    break
            
            return ConnectorResult(
                success=True,
                documents=documents,
                metadata={
                    "query": query,
                    "total_results": len(documents),
                    "source": "arxiv",
                    "sort_by": self.arxiv_config.sort_by,
                },
            )
            
        except Exception as e:
            logger.error(f"arXiv fetch failed: {e}")
            return ConnectorResult.from_error(str(e))
    
    def fetch_by_id(self, arxiv_id: str) -> ConnectorResult:
        """Fetch a specific paper by arXiv ID.
        
        Args:
            arxiv_id: arXiv paper ID (e.g., "2103.14030")
            
        Returns:
            ConnectorResult with single paper document
        """
        # Normalize ID (remove version if present for initial fetch)
        clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
        return self.fetch("", max_results=1, id_list=[clean_id])
    
    def fetch_by_category(
        self,
        categories: list[str],
        max_results: int = 10,
        sort_by: str = "submittedDate",
    ) -> ConnectorResult:
        """Fetch recent papers from specified categories.
        
        Args:
            categories: List of arXiv categories
            max_results: Maximum papers to return
            sort_by: Sort criterion
            
        Returns:
            ConnectorResult with paper documents
        """
        # Store original sort_by and temporarily override
        original_sort = self.arxiv_config.sort_by
        self.config.sort_by = sort_by  # type: ignore
        
        try:
            return self.fetch("", max_results=max_results, categories=categories)
        finally:
            self.config.sort_by = original_sort  # type: ignore
    
    def test_connection(self) -> bool:
        """Test connection to arXiv API.
        
        Returns:
            True if successful
        """
        try:
            result = self.fetch("test", max_results=1)
            return result.success and len(result.documents) > 0
        except Exception:
            return False
    
    def validate_config(self) -> Optional[str]:
        """Validate connector configuration.
        
        Returns:
            Error message if invalid, None if valid
        """
        valid_sort_by = {"relevance", "lastupdateddate", "submitteddate"}
        if self.arxiv_config.sort_by.lower() not in valid_sort_by:
            return f"Invalid sort_by: {self.arxiv_config.sort_by}. Must be one of {valid_sort_by}"
        
        valid_sort_order = {"ascending", "descending"}
        if self.arxiv_config.sort_order.lower() not in valid_sort_order:
            return f"Invalid sort_order: {self.arxiv_config.sort_order}. Must be one of {valid_sort_order}"
        
        return None
