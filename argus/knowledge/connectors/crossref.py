"""
CrossRef API Connector for ARGUS.

Fetches citation metadata from CrossRef for academic references.

Example:
    >>> connector = CrossRefConnector()
    >>> result = connector.fetch("10.1038/nature12373")  # Fetch by DOI
    >>> 
    >>> # Or search by bibliographic query
    >>> result = connector.fetch("attention is all you need transformers")
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import quote_plus

from pydantic import Field

from argus.knowledge.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectorResult,
)
from argus.core.models import Document, SourceType

logger = logging.getLogger(__name__)


@dataclass
class CrossRefWork:
    """Represents a CrossRef work (publication) with metadata."""
    doi: str
    title: str
    authors: list[dict[str, str]]  # {"given": "...", "family": "..."}
    container_title: Optional[str] = None  # Journal/conference name
    publisher: Optional[str] = None
    published_date: Optional[datetime] = None
    type: Optional[str] = None  # journal-article, proceedings-article, etc.
    volume: Optional[str] = None
    issue: Optional[str] = None
    page: Optional[str] = None
    issn: list[str] = field(default_factory=list)
    isbn: list[str] = field(default_factory=list)
    abstract: Optional[str] = None
    reference_count: int = 0
    is_referenced_by_count: int = 0
    subject: list[str] = field(default_factory=list)
    url: Optional[str] = None
    license: Optional[str] = None
    
    @property
    def author_names(self) -> list[str]:
        """Get formatted author names."""
        names = []
        for author in self.authors:
            given = author.get("given", "")
            family = author.get("family", "")
            if given and family:
                names.append(f"{given} {family}")
            elif family:
                names.append(family)
            elif given:
                names.append(given)
        return names
    
    @property
    def citation_string(self) -> str:
        """Generate a citation string."""
        parts = []
        
        # Authors
        if self.author_names:
            if len(self.author_names) > 3:
                parts.append(f"{self.author_names[0]} et al.")
            else:
                parts.append(", ".join(self.author_names))
        
        # Year
        if self.published_date:
            parts.append(f"({self.published_date.year})")
        
        # Title
        parts.append(f'"{self.title}"')
        
        # Journal/Container
        if self.container_title:
            parts.append(self.container_title)
        
        # Volume/Issue/Pages
        loc = []
        if self.volume:
            loc.append(f"vol. {self.volume}")
        if self.issue:
            loc.append(f"no. {self.issue}")
        if self.page:
            loc.append(f"pp. {self.page}")
        if loc:
            parts.append(", ".join(loc))
        
        # DOI
        parts.append(f"DOI: {self.doi}")
        
        return ". ".join(parts)
    
    def to_document(self) -> Document:
        """Convert to ARGUS Document."""
        content = f"# {self.title}\n\n"
        content += f"**Authors:** {', '.join(self.author_names)}\n\n"
        
        if self.container_title:
            content += f"**Published in:** {self.container_title}\n\n"
        
        if self.published_date:
            content += f"**Date:** {self.published_date.strftime('%Y-%m-%d')}\n\n"
        
        content += f"**DOI:** {self.doi}\n\n"
        
        if self.abstract:
            content += f"## Abstract\n\n{self.abstract}\n\n"
        
        content += f"## Citation\n\n{self.citation_string}\n"
        
        return Document(
            url=self.url or f"https://doi.org/{self.doi}",
            title=self.title,
            content=content,
            source_type=SourceType.TEXT,
            metadata={
                "doi": self.doi,
                "authors": self.authors,
                "author_names": self.author_names,
                "container_title": self.container_title,
                "publisher": self.publisher,
                "published_date": self.published_date.isoformat() if self.published_date else None,
                "type": self.type,
                "volume": self.volume,
                "issue": self.issue,
                "page": self.page,
                "reference_count": self.reference_count,
                "cited_by_count": self.is_referenced_by_count,
                "subjects": self.subject,
                "source": "crossref",
            },
        )


class CrossRefConnectorConfig(ConnectorConfig):
    """Configuration for CrossRef connector.
    
    Attributes:
        mailto: Email for polite pool (faster rate limits)
        sort: Sort results by score, relevance, published, etc.
        order: asc or desc
    """
    mailto: Optional[str] = Field(
        default=None,
        description="Email for CrossRef polite pool (recommended for better rate limits)",
    )
    sort: str = Field(
        default="score",
        description="Sort by: score, relevance, published, updated, created",
    )
    order: str = Field(
        default="desc",
        description="Sort order: asc or desc",
    )


class CrossRefConnector(BaseConnector):
    """Connector for fetching citation metadata from CrossRef.
    
    Uses the CrossRef REST API to look up DOIs and search for publications.
    Returns works as Document objects with citation metadata.
    
    Example:
        >>> connector = CrossRefConnector()
        >>> 
        >>> # Lookup by DOI
        >>> result = connector.fetch_by_doi("10.1038/nature12373")
        >>> 
        >>> # Search by query
        >>> result = connector.fetch("deep learning neural networks")
    
    Note:
        Set mailto in config for faster API access via CrossRef's polite pool.
    """
    
    name = "crossref"
    description = "Fetch citation metadata from CrossRef"
    version = "1.0.0"
    
    BASE_URL = "https://api.crossref.org"
    
    def __init__(self, config: Optional[CrossRefConnectorConfig] = None):
        """Initialize CrossRef connector.
        
        Args:
            config: Connector configuration
        """
        super().__init__(config or CrossRefConnectorConfig())
        self._session = None
    
    @property
    def crossref_config(self) -> CrossRefConnectorConfig:
        """Get typed config."""
        return self.config  # type: ignore
    
    def _get_session(self):
        """Get or create HTTP session."""
        if self._session is None:
            try:
                import httpx
                headers = {
                    "User-Agent": "ARGUS-Bot/1.0 (https://github.com/argus; mailto:argus@example.com)",
                }
                if self.crossref_config.mailto:
                    headers["User-Agent"] = f"ARGUS-Bot/1.0 (mailto:{self.crossref_config.mailto})"
                
                self._session = httpx.Client(
                    timeout=self.config.timeout,
                    headers=headers,
                )
            except ImportError:
                logger.warning("httpx not available, using urllib")
        return self._session
    
    def _make_request(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make a request to CrossRef API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            JSON response as dict
            
        Raises:
            Exception: On request failure
        """
        url = f"{self.BASE_URL}{endpoint}"
        
        if params is None:
            params = {}
        
        # Add mailto for polite pool
        if self.crossref_config.mailto:
            params["mailto"] = self.crossref_config.mailto
        
        session = self._get_session()
        
        if session:
            response = session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        else:
            # Fallback to urllib
            import urllib.request
            import json
            
            query_string = "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items())
            full_url = f"{url}?{query_string}" if query_string else url
            
            headers = {"User-Agent": "ARGUS-Bot/1.0"}
            req = urllib.request.Request(full_url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
    
    def _parse_date(self, date_parts: Optional[list]) -> Optional[datetime]:
        """Parse CrossRef date-parts into datetime.
        
        Args:
            date_parts: List of [year, month, day] (partial allowed)
            
        Returns:
            datetime or None
        """
        if not date_parts or not date_parts[0]:
            return None
        
        parts = date_parts[0]
        year = parts[0] if len(parts) > 0 else 1970
        month = parts[1] if len(parts) > 1 else 1
        day = parts[2] if len(parts) > 2 else 1
        
        try:
            return datetime(year, month, day)
        except (ValueError, TypeError):
            return None
    
    def _parse_work(self, item: dict) -> CrossRefWork:
        """Parse CrossRef work JSON into CrossRefWork.
        
        Args:
            item: CrossRef work item
            
        Returns:
            CrossRefWork dataclass
        """
        # Get title (can be a list)
        title_list = item.get("title", [])
        title = title_list[0] if title_list else "Untitled"
        
        # Get container title
        container_list = item.get("container-title", [])
        container_title = container_list[0] if container_list else None
        
        # Get published date
        published = item.get("published-print") or item.get("published-online") or item.get("created")
        published_date = self._parse_date(published.get("date-parts") if published else None)
        
        # Get abstract (may have JATS formatting)
        abstract = item.get("abstract")
        if abstract:
            # Remove JATS tags
            import re
            abstract = re.sub(r"<[^>]+>", "", abstract)
        
        # Get license URL
        licenses = item.get("license", [])
        license_url = licenses[0].get("URL") if licenses else None
        
        return CrossRefWork(
            doi=item.get("DOI", ""),
            title=title,
            authors=item.get("author", []),
            container_title=container_title,
            publisher=item.get("publisher"),
            published_date=published_date,
            type=item.get("type"),
            volume=item.get("volume"),
            issue=item.get("issue"),
            page=item.get("page"),
            issn=item.get("ISSN", []),
            isbn=item.get("ISBN", []),
            abstract=abstract,
            reference_count=item.get("reference-count", 0),
            is_referenced_by_count=item.get("is-referenced-by-count", 0),
            subject=item.get("subject", []),
            url=item.get("URL"),
            license=license_url,
        )
    
    def fetch(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> ConnectorResult:
        """Fetch works from CrossRef.
        
        Args:
            query: Search query or DOI
            max_results: Maximum results to return
            **kwargs: Additional options
                - filter: Dict of CrossRef filters
                - select: Fields to select
                - sample: Random sample size
            
        Returns:
            ConnectorResult with work documents
        """
        # Check if query looks like a DOI
        if query.startswith("10.") or query.startswith("doi:"):
            return self.fetch_by_doi(query.replace("doi:", ""))
        
        try:
            params = {
                "query": query,
                "rows": min(max_results, 100),  # CrossRef max is 1000
                "sort": self.crossref_config.sort,
                "order": self.crossref_config.order,
            }
            
            # Add filters
            filters = kwargs.get("filter", {})
            if filters:
                filter_str = ",".join(f"{k}:{v}" for k, v in filters.items())
                params["filter"] = filter_str
            
            # Add field selection
            if "select" in kwargs:
                params["select"] = ",".join(kwargs["select"])
            
            response = self._make_request("/works", params)
            
            if response.get("status") != "ok":
                return ConnectorResult.from_error(
                    f"CrossRef API error: {response.get('message', 'Unknown error')}"
                )
            
            message = response.get("message", {})
            items = message.get("items", [])
            total_results = message.get("total-results", 0)
            
            documents: list[Document] = []
            for item in items[:max_results]:
                work = self._parse_work(item)
                documents.append(work.to_document())
            
            return ConnectorResult(
                success=True,
                documents=documents,
                metadata={
                    "query": query,
                    "total_results": total_results,
                    "returned_results": len(documents),
                    "source": "crossref",
                },
            )
            
        except Exception as e:
            logger.error(f"CrossRef fetch failed: {e}")
            return ConnectorResult.from_error(str(e))
    
    def fetch_by_doi(self, doi: str) -> ConnectorResult:
        """Fetch a specific work by DOI.
        
        Args:
            doi: DOI string (e.g., "10.1038/nature12373")
            
        Returns:
            ConnectorResult with single work document
        """
        try:
            # Clean DOI
            doi = doi.strip()
            if doi.startswith("https://doi.org/"):
                doi = doi.replace("https://doi.org/", "")
            elif doi.startswith("http://dx.doi.org/"):
                doi = doi.replace("http://dx.doi.org/", "")
            
            response = self._make_request(f"/works/{quote_plus(doi)}")
            
            if response.get("status") != "ok":
                return ConnectorResult.from_error(
                    f"DOI not found: {doi}"
                )
            
            item = response.get("message", {})
            work = self._parse_work(item)
            document = work.to_document()
            
            return ConnectorResult(
                success=True,
                documents=[document],
                metadata={
                    "doi": doi,
                    "source": "crossref",
                },
            )
            
        except Exception as e:
            logger.error(f"CrossRef DOI lookup failed for {doi}: {e}")
            return ConnectorResult.from_error(str(e))
    
    def fetch_references(
        self,
        doi: str,
        max_results: int = 50,
    ) -> ConnectorResult:
        """Fetch references for a work.
        
        Args:
            doi: DOI of the work
            max_results: Maximum references to return
            
        Returns:
            ConnectorResult with reference documents
        """
        try:
            response = self._make_request(f"/works/{quote_plus(doi)}")
            
            if response.get("status") != "ok":
                return ConnectorResult.from_error(f"DOI not found: {doi}")
            
            item = response.get("message", {})
            references = item.get("reference", [])
            
            documents: list[Document] = []
            
            for ref in references[:max_results]:
                # References have limited metadata
                ref_doi = ref.get("DOI")
                title = ref.get("article-title") or ref.get("unstructured", "")
                author = ref.get("author", "")
                year = ref.get("year", "")
                journal = ref.get("journal-title", "")
                
                content = []
                if author:
                    content.append(f"**Author:** {author}")
                if year:
                    content.append(f"**Year:** {year}")
                if journal:
                    content.append(f"**Journal:** {journal}")
                if ref_doi:
                    content.append(f"**DOI:** {ref_doi}")
                
                doc = Document(
                    url=f"https://doi.org/{ref_doi}" if ref_doi else "",
                    title=title[:200] if title else "Unknown Reference",
                    content="\n".join(content) if content else title,
                    source_type=SourceType.TEXT,
                    metadata={
                        "doi": ref_doi,
                        "author": author,
                        "year": year,
                        "journal": journal,
                        "source": "crossref_reference",
                    },
                )
                documents.append(doc)
            
            return ConnectorResult(
                success=True,
                documents=documents,
                metadata={
                    "parent_doi": doi,
                    "total_references": len(references),
                    "returned_references": len(documents),
                    "source": "crossref",
                },
            )
            
        except Exception as e:
            logger.error(f"CrossRef references fetch failed: {e}")
            return ConnectorResult.from_error(str(e))
    
    def fetch_citing_works(
        self,
        doi: str,
        max_results: int = 20,
    ) -> ConnectorResult:
        """Fetch works that cite a given DOI.
        
        Note: This uses the CrossRef filter API which may not be complete.
        
        Args:
            doi: DOI to find citations for
            max_results: Maximum citing works to return
            
        Returns:
            ConnectorResult with citing work documents
        """
        return self.fetch(
            "",
            max_results=max_results,
            filter={"references": doi},
        )
    
    def test_connection(self) -> bool:
        """Test connection to CrossRef API.
        
        Returns:
            True if successful
        """
        try:
            result = self.fetch("test", max_results=1)
            return result.success
        except Exception:
            return False
    
    def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            self._session.close()
            self._session = None
