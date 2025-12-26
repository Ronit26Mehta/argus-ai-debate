"""
Document Chunking for ARGUS.

Provides semantic chunking strategies for splitting documents
into chunks suitable for embedding and retrieval.
"""

from __future__ import annotations

import re
import logging
from enum import Enum
from typing import Optional, Any, TYPE_CHECKING
from dataclasses import dataclass

from argus.core.models import Document, Chunk

logger = logging.getLogger(__name__)


class ChunkingStrategy(str, Enum):
    """Chunking strategy options."""
    FIXED_SIZE = "fixed_size"       # Fixed token count
    SENTENCE = "sentence"           # Sentence boundaries
    PARAGRAPH = "paragraph"         # Paragraph boundaries
    SEMANTIC = "semantic"           # Semantic units
    RECURSIVE = "recursive"         # Recursive splitting


@dataclass
class ChunkingConfig:
    """
    Configuration for chunking.
    
    Attributes:
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks
        min_chunk_size: Minimum chunk size
        max_chunk_size: Maximum chunk size
        strategy: Chunking strategy
    """
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    max_chunk_size: int = 1000
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE


class Chunker:
    """
    Document chunker.
    
    Splits documents into chunks using various strategies.
    Maintains character offsets for provenance.
    
    Example:
        >>> chunker = Chunker(chunk_size=512, chunk_overlap=50)
        >>> chunks = chunker.chunk(document)
        >>> for chunk in chunks:
        ...     print(f"Chunk {chunk.chunk_index}: {len(chunk.text)} chars")
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
            strategy: Chunking strategy
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.strategy = strategy
        
        # Approximate chars per token
        self._chars_per_token = 4
    
    def chunk(self, document: Document) -> list[Chunk]:
        """
        Chunk a document.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of Chunk objects
        """
        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._chunk_fixed(document)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            return self._chunk_sentences(document)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            return self._chunk_paragraphs(document)
        elif self.strategy == ChunkingStrategy.RECURSIVE:
            return self._chunk_recursive(document)
        else:
            return self._chunk_recursive(document)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text) // self._chars_per_token
    
    def _create_chunk(
        self,
        doc_id: str,
        text: str,
        start_char: int,
        end_char: int,
        chunk_index: int,
    ) -> Chunk:
        """Create a Chunk object."""
        return Chunk(
            doc_id=doc_id,
            text=text.strip(),
            start_char=start_char,
            end_char=end_char,
            chunk_index=chunk_index,
            token_count=self._estimate_tokens(text),
        )
    
    def _chunk_fixed(self, document: Document) -> list[Chunk]:
        """Chunk by fixed size with overlap."""
        text = document.content
        target_chars = self.chunk_size * self._chars_per_token
        overlap_chars = self.chunk_overlap * self._chars_per_token
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + target_chars, len(text))
            
            # Try to break at word boundary
            if end < len(text):
                # Find last space
                space_pos = text.rfind(" ", start, end)
                if space_pos > start + overlap_chars:
                    end = space_pos
            
            chunk_text = text[start:end]
            
            if len(chunk_text.strip()) >= self.min_chunk_size * self._chars_per_token:
                chunks.append(self._create_chunk(
                    document.id,
                    chunk_text,
                    start,
                    end,
                    chunk_index,
                ))
                chunk_index += 1
            
            start = end - overlap_chars
            if start >= len(text):
                break
        
        return chunks
    
    def _chunk_sentences(self, document: Document) -> list[Chunk]:
        """Chunk by sentence boundaries."""
        text = document.content
        
        # Split into sentences
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        
        chunks = []
        current_text = ""
        current_start = 0
        chunk_index = 0
        
        target_chars = self.chunk_size * self._chars_per_token
        
        for sentence in sentences:
            if len(current_text) + len(sentence) <= target_chars:
                current_text += sentence + " "
            else:
                if current_text.strip():
                    end_char = current_start + len(current_text)
                    chunks.append(self._create_chunk(
                        document.id,
                        current_text,
                        current_start,
                        end_char,
                        chunk_index,
                    ))
                    chunk_index += 1
                    current_start = end_char
                
                current_text = sentence + " "
        
        # Last chunk
        if current_text.strip():
            chunks.append(self._create_chunk(
                document.id,
                current_text,
                current_start,
                len(text),
                chunk_index,
            ))
        
        return chunks
    
    def _chunk_paragraphs(self, document: Document) -> list[Chunk]:
        """Chunk by paragraph boundaries."""
        text = document.content
        
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_text = ""
        current_start = 0
        chunk_index = 0
        char_pos = 0
        
        target_chars = self.chunk_size * self._chars_per_token
        
        for para in paragraphs:
            para_with_sep = para + "\n\n"
            
            if len(current_text) + len(para_with_sep) <= target_chars:
                current_text += para_with_sep
            else:
                if current_text.strip():
                    end_char = current_start + len(current_text)
                    chunks.append(self._create_chunk(
                        document.id,
                        current_text,
                        current_start,
                        end_char,
                        chunk_index,
                    ))
                    chunk_index += 1
                    current_start = end_char
                
                current_text = para_with_sep
        
        if current_text.strip():
            chunks.append(self._create_chunk(
                document.id,
                current_text,
                current_start,
                len(text),
                chunk_index,
            ))
        
        return chunks
    
    def _chunk_recursive(self, document: Document) -> list[Chunk]:
        """
        Recursively chunk using multiple separators.
        
        First tries paragraphs, then sentences, then fixed size.
        """
        text = document.content
        target_chars = self.chunk_size * self._chars_per_token
        
        # Separators in order of preference
        separators = [
            "\n\n",    # Paragraph
            "\n",      # Line
            ". ",      # Sentence
            "! ",
            "? ",
            "; ",
            ", ",
            " ",       # Word
        ]
        
        def split_text(txt: str, start_offset: int, sep_idx: int) -> list[tuple[str, int, int]]:
            """Recursively split text and return (text, start, end) tuples."""
            if len(txt) <= target_chars:
                return [(txt, start_offset, start_offset + len(txt))]
            
            if sep_idx >= len(separators):
                # Fall back to fixed split
                results = []
                pos = 0
                while pos < len(txt):
                    end = min(pos + target_chars, len(txt))
                    results.append((
                        txt[pos:end],
                        start_offset + pos,
                        start_offset + end,
                    ))
                    pos = end - self.chunk_overlap * self._chars_per_token
                    if pos < 0:
                        pos = end
                return results
            
            sep = separators[sep_idx]
            parts = txt.split(sep)
            
            if len(parts) == 1:
                return split_text(txt, start_offset, sep_idx + 1)
            
            results = []
            current = ""
            current_offset = start_offset
            
            for i, part in enumerate(parts):
                part_with_sep = part + sep if i < len(parts) - 1 else part
                
                if len(current) + len(part_with_sep) <= target_chars:
                    current += part_with_sep
                else:
                    if current:
                        if len(current) > target_chars:
                            results.extend(split_text(current, current_offset, sep_idx + 1))
                        else:
                            results.append((current, current_offset, current_offset + len(current)))
                        current_offset += len(current)
                    current = part_with_sep
            
            if current:
                if len(current) > target_chars:
                    results.extend(split_text(current, current_offset, sep_idx + 1))
                else:
                    results.append((current, current_offset, current_offset + len(current)))
            
            return results
        
        # Split and create chunks
        split_results = split_text(text, 0, 0)
        
        chunks = []
        for idx, (chunk_text, start, end) in enumerate(split_results):
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunks.append(self._create_chunk(
                    document.id,
                    chunk_text,
                    start,
                    end,
                    idx,
                ))
        
        # Re-index
        for i, chunk in enumerate(chunks):
            object.__setattr__(chunk, "chunk_index", i)
        
        logger.debug(
            f"Chunked document {document.id}: {len(chunks)} chunks"
        )
        
        return chunks


def chunk_document(
    document: Document,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
) -> list[Chunk]:
    """
    Convenience function to chunk a document.
    
    Args:
        document: Document to chunk
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        strategy: Chunking strategy
        
    Returns:
        List of Chunk objects
    """
    chunker = Chunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=strategy,
    )
    return chunker.chunk(document)
