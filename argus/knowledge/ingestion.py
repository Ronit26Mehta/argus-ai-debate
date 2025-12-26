"""
Document Ingestion for ARGUS.

Provides loaders for various document formats:
    - PDF (via PyMuPDF)
    - HTML (via BeautifulSoup)
    - Plain text
    - CSV, JSON
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Union
from urllib.parse import urlparse

from argus.core.models import Document, SourceType

logger = logging.getLogger(__name__)


def compute_checksum(content: str) -> str:
    """Compute SHA-256 checksum of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def detect_source_type(path_or_url: str) -> SourceType:
    """
    Detect source type from file path or URL.
    
    Args:
        path_or_url: File path or URL
        
    Returns:
        SourceType enum value
    """
    lower = path_or_url.lower()
    
    if lower.endswith(".pdf"):
        return SourceType.PDF
    elif lower.endswith((".html", ".htm")):
        return SourceType.HTML
    elif lower.endswith((".txt", ".text")):
        return SourceType.TEXT
    elif lower.endswith(".csv"):
        return SourceType.CSV
    elif lower.endswith(".json"):
        return SourceType.JSON
    elif lower.endswith((".md", ".markdown")):
        return SourceType.MARKDOWN
    else:
        return SourceType.UNKNOWN


class DocumentLoader:
    """
    Universal document loader.
    
    Automatically detects file format and extracts text content.
    Supports PDF, HTML, CSV, JSON, and plain text.
    
    Example:
        >>> loader = DocumentLoader()
        >>> doc = loader.load("paper.pdf")
        >>> print(doc.title)
        >>> print(doc.content[:500])
    """
    
    def __init__(
        self,
        compute_checksums: bool = True,
        max_content_length: Optional[int] = None,
    ):
        """
        Initialize document loader.
        
        Args:
            compute_checksums: Whether to compute content checksums
            max_content_length: Maximum content length (truncate if exceeded)
        """
        self.compute_checksums = compute_checksums
        self.max_content_length = max_content_length
    
    def load(
        self,
        source: Union[str, Path],
        title: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Document:
        """
        Load a document from file path or URL.
        
        Args:
            source: File path or URL
            title: Optional title (extracted if not provided)
            metadata: Additional metadata
            
        Returns:
            Document object
        """
        source_str = str(source)
        source_type = detect_source_type(source_str)
        
        if source_type == SourceType.PDF:
            return self._load_pdf(source_str, title, metadata)
        elif source_type == SourceType.HTML:
            return self._load_html(source_str, title, metadata)
        elif source_type == SourceType.CSV:
            return self._load_csv(source_str, title, metadata)
        elif source_type == SourceType.JSON:
            return self._load_json(source_str, title, metadata)
        else:
            return self._load_text(source_str, title, metadata)
    
    def _load_pdf(
        self,
        path: str,
        title: Optional[str],
        metadata: Optional[dict[str, Any]],
    ) -> Document:
        """Load PDF document using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF loading. "
                "Install with: pip install pymupdf"
            )
        
        path_obj = Path(path)
        
        with fitz.open(path) as doc:
            # Extract metadata
            pdf_metadata = doc.metadata
            extracted_title = (
                title or
                pdf_metadata.get("title") or
                path_obj.stem
            )
            
            # Extract text from all pages
            text_parts = []
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    text_parts.append(text)
            
            content = "\n\n".join(text_parts)
            
            # Truncate if needed
            if self.max_content_length and len(content) > self.max_content_length:
                content = content[:self.max_content_length]
            
            # Compute checksum
            checksum = compute_checksum(content) if self.compute_checksums else None
            
            return Document(
                url=str(path_obj.absolute()),
                title=extracted_title,
                content=content,
                source_type=SourceType.PDF,
                checksum=checksum,
                metadata={
                    **(metadata or {}),
                    "pdf_metadata": pdf_metadata,
                    "num_pages": len(doc),
                },
            )
    
    def _load_html(
        self,
        path_or_url: str,
        title: Optional[str],
        metadata: Optional[dict[str, Any]],
    ) -> Document:
        """Load HTML document using BeautifulSoup."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "BeautifulSoup is required for HTML loading. "
                "Install with: pip install beautifulsoup4"
            )
        
        # Determine if URL or file
        parsed = urlparse(path_or_url)
        is_url = bool(parsed.scheme and parsed.netloc)
        
        if is_url:
            import httpx
            response = httpx.get(path_or_url, follow_redirects=True)
            response.raise_for_status()
            html_content = response.text
            source_url = path_or_url
        else:
            path_obj = Path(path_or_url)
            html_content = path_obj.read_text(encoding="utf-8")
            source_url = str(path_obj.absolute())
        
        # Parse HTML
        soup = BeautifulSoup(html_content, "lxml")
        
        # Extract title
        extracted_title = title
        if not extracted_title:
            title_tag = soup.find("title")
            if title_tag:
                extracted_title = title_tag.get_text(strip=True)
        if not extracted_title:
            extracted_title = "Untitled"
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text content
        content = soup.get_text(separator="\n", strip=True)
        
        # Clean up whitespace
        content = re.sub(r"\n\s*\n", "\n\n", content)
        
        # Truncate if needed
        if self.max_content_length and len(content) > self.max_content_length:
            content = content[:self.max_content_length]
        
        checksum = compute_checksum(content) if self.compute_checksums else None
        
        return Document(
            url=source_url,
            title=extracted_title,
            content=content,
            source_type=SourceType.HTML,
            checksum=checksum,
            metadata=metadata or {},
        )
    
    def _load_csv(
        self,
        path: str,
        title: Optional[str],
        metadata: Optional[dict[str, Any]],
    ) -> Document:
        """Load CSV as text content."""
        import csv
        
        path_obj = Path(path)
        
        with open(path_obj, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Convert to text (header + rows)
        lines = []
        for row in rows:
            lines.append(" | ".join(row))
        
        content = "\n".join(lines)
        
        checksum = compute_checksum(content) if self.compute_checksums else None
        
        return Document(
            url=str(path_obj.absolute()),
            title=title or path_obj.stem,
            content=content,
            source_type=SourceType.CSV,
            checksum=checksum,
            metadata={
                **(metadata or {}),
                "num_rows": len(rows),
                "num_cols": len(rows[0]) if rows else 0,
            },
        )
    
    def _load_json(
        self,
        path: str,
        title: Optional[str],
        metadata: Optional[dict[str, Any]],
    ) -> Document:
        """Load JSON as text content."""
        import json
        
        path_obj = Path(path)
        
        with open(path_obj, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Convert to formatted string
        content = json.dumps(data, indent=2)
        
        checksum = compute_checksum(content) if self.compute_checksums else None
        
        return Document(
            url=str(path_obj.absolute()),
            title=title or path_obj.stem,
            content=content,
            source_type=SourceType.JSON,
            checksum=checksum,
            metadata=metadata or {},
        )
    
    def _load_text(
        self,
        path: str,
        title: Optional[str],
        metadata: Optional[dict[str, Any]],
    ) -> Document:
        """Load plain text file."""
        path_obj = Path(path)
        
        # Detect encoding
        try:
            import chardet
            with open(path_obj, "rb") as f:
                raw = f.read()
            detected = chardet.detect(raw)
            encoding = detected.get("encoding", "utf-8")
        except ImportError:
            encoding = "utf-8"
        
        content = path_obj.read_text(encoding=encoding)
        
        checksum = compute_checksum(content) if self.compute_checksums else None
        
        # Detect if markdown
        source_type = SourceType.TEXT
        if path.lower().endswith((".md", ".markdown")):
            source_type = SourceType.MARKDOWN
        
        return Document(
            url=str(path_obj.absolute()),
            title=title or path_obj.stem,
            content=content,
            source_type=source_type,
            checksum=checksum,
            metadata=metadata or {},
        )


# Convenience functions
def load_document(
    source: Union[str, Path],
    title: Optional[str] = None,
    **kwargs: Any,
) -> Document:
    """
    Load a document from file path or URL.
    
    Args:
        source: File path or URL
        title: Optional title
        **kwargs: Additional options
        
    Returns:
        Document object
    """
    loader = DocumentLoader()
    return loader.load(source, title, kwargs.get("metadata"))


def load_pdf(
    path: Union[str, Path],
    title: Optional[str] = None,
) -> Document:
    """Load a PDF document."""
    loader = DocumentLoader()
    return loader._load_pdf(str(path), title, None)


def load_html(
    path_or_url: str,
    title: Optional[str] = None,
) -> Document:
    """Load an HTML document or URL."""
    loader = DocumentLoader()
    return loader._load_html(path_or_url, title, None)


def load_text(
    path: Union[str, Path],
    title: Optional[str] = None,
) -> Document:
    """Load a text file."""
    loader = DocumentLoader()
    return loader._load_text(str(path), title, None)
