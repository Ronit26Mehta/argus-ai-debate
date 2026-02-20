"""
Context Compression Module for ARGUS.

Intelligent compression techniques for reducing LLM context size
while preserving semantic meaning and important information.
"""

from __future__ import annotations

import re
import logging
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class CompressionLevel(Enum):
    """Compression intensity levels."""
    MINIMAL = "minimal"      # Light compression, maximum fidelity
    MODERATE = "moderate"    # Balanced compression
    AGGRESSIVE = "aggressive" # Heavy compression, may lose some detail
    EXTREME = "extreme"      # Maximum compression, significant detail loss


@dataclass
class CompressionResult:
    """Result of a compression operation."""
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    technique: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def tokens_saved(self) -> int:
        """Number of tokens saved."""
        return self.original_tokens - self.compressed_tokens
    
    @property
    def savings_percentage(self) -> float:
        """Percentage of tokens saved."""
        if self.original_tokens == 0:
            return 0.0
        return (self.tokens_saved / self.original_tokens) * 100


@dataclass
class Message:
    """A chat message."""
    role: str
    content: str
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TokenCounter:
    """Token counting utilities."""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self._encoder = None
    
    def _get_encoder(self):
        """Get tiktoken encoder lazily."""
        if self._encoder is None:
            try:
                import tiktoken
                try:
                    self._encoder = tiktoken.encoding_for_model(self.model)
                except KeyError:
                    self._encoder = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                self._encoder = None
        return self._encoder
    
    def count(self, text: str) -> int:
        """Count tokens in text."""
        encoder = self._get_encoder()
        if encoder:
            return len(encoder.encode(text))
        
        # Fallback: rough estimation (4 chars per token)
        return len(text) // 4
    
    def count_messages(self, messages: List[Message]) -> int:
        """Count tokens in messages."""
        total = 0
        for msg in messages:
            # Add overhead for message formatting
            total += 4  # Role + separators
            total += self.count(msg.content)
            if msg.name:
                total += self.count(msg.name) + 1
        total += 2  # Start/end tokens
        return total


class Compressor(ABC):
    """Abstract base class for compression techniques."""
    
    name: str = "base"
    
    @abstractmethod
    def compress(
        self,
        text: str,
        level: CompressionLevel = CompressionLevel.MODERATE,
        **kwargs,
    ) -> str:
        """Compress text."""
        pass
    
    def compress_with_result(
        self,
        text: str,
        level: CompressionLevel = CompressionLevel.MODERATE,
        token_counter: Optional[TokenCounter] = None,
        **kwargs,
    ) -> CompressionResult:
        """Compress text and return detailed result."""
        counter = token_counter or TokenCounter()
        
        original_tokens = counter.count(text)
        compressed = self.compress(text, level, **kwargs)
        compressed_tokens = counter.count(compressed)
        
        ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        
        return CompressionResult(
            original_text=text,
            compressed_text=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=ratio,
            technique=self.name,
        )


class WhitespaceCompressor(Compressor):
    """Compress by normalizing whitespace."""
    
    name = "whitespace"
    
    def compress(
        self,
        text: str,
        level: CompressionLevel = CompressionLevel.MODERATE,
        **kwargs,
    ) -> str:
        """Normalize whitespace in text."""
        if level == CompressionLevel.MINIMAL:
            # Just normalize multiple spaces
            text = re.sub(r" +", " ", text)
        elif level == CompressionLevel.MODERATE:
            # Normalize spaces and reduce blank lines
            text = re.sub(r" +", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
        elif level == CompressionLevel.AGGRESSIVE:
            # Single spaces, single newlines
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n+", "\n", text)
        else:  # EXTREME
            # Remove all extra whitespace
            text = re.sub(r"\s+", " ", text).strip()
        
        return text


class PunctuationCompressor(Compressor):
    """Compress by simplifying punctuation."""
    
    name = "punctuation"
    
    def compress(
        self,
        text: str,
        level: CompressionLevel = CompressionLevel.MODERATE,
        **kwargs,
    ) -> str:
        """Simplify punctuation."""
        if level in (CompressionLevel.MINIMAL, CompressionLevel.MODERATE):
            # Normalize ellipsis and dashes
            text = re.sub(r"\.{3,}", "...", text)
            text = re.sub(r"-{2,}", "—", text)
            text = re.sub(r"!{2,}", "!", text)
            text = re.sub(r"\?{2,}", "?", text)
        
        if level in (CompressionLevel.AGGRESSIVE, CompressionLevel.EXTREME):
            # Remove decorative punctuation
            text = re.sub(r"[~*_]{2,}", "", text)
            text = re.sub(r"\.{2,}", ".", text)
        
        return text


class StopwordCompressor(Compressor):
    """Compress by removing or abbreviating common words."""
    
    name = "stopword"
    
    # Common stopwords that can often be removed
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "need",
        "that", "which", "who", "whom", "this", "these", "those", "it",
        "its", "very", "really", "just", "also", "even", "only", "still",
    }
    
    # Common abbreviations
    ABBREVIATIONS = {
        "information": "info",
        "approximately": "approx",
        "configuration": "config",
        "documentation": "docs",
        "application": "app",
        "specification": "spec",
        "implementation": "impl",
        "development": "dev",
        "production": "prod",
        "environment": "env",
        "repository": "repo",
        "directory": "dir",
        "administrator": "admin",
        "authentication": "auth",
        "authorization": "authz",
        "database": "db",
        "function": "func",
        "parameter": "param",
        "argument": "arg",
        "variable": "var",
        "constant": "const",
        "reference": "ref",
        "message": "msg",
        "number": "num",
        "string": "str",
        "integer": "int",
        "boolean": "bool",
        "temporary": "temp",
        "maximum": "max",
        "minimum": "min",
        "average": "avg",
        "previous": "prev",
        "current": "curr",
        "original": "orig",
        "source": "src",
        "destination": "dest",
        "length": "len",
        "count": "cnt",
        "index": "idx",
        "position": "pos",
    }
    
    def compress(
        self,
        text: str,
        level: CompressionLevel = CompressionLevel.MODERATE,
        **kwargs,
    ) -> str:
        """Remove stopwords and apply abbreviations."""
        # Apply abbreviations (all levels)
        for full, abbrev in self.ABBREVIATIONS.items():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(full), re.IGNORECASE)
            text = pattern.sub(abbrev, text)
        
        if level in (CompressionLevel.AGGRESSIVE, CompressionLevel.EXTREME):
            # Remove stopwords (careful with context)
            words = text.split()
            filtered = []
            
            for i, word in enumerate(words):
                word_lower = word.lower().strip(".,!?;:")
                
                # Keep word if:
                # - Not a stopword
                # - Is at start of sentence (after period)
                # - Is capitalized (proper noun)
                if (
                    word_lower not in self.STOPWORDS
                    or (i > 0 and words[i-1].endswith("."))
                    or (word[0].isupper() and i > 0)
                ):
                    filtered.append(word)
            
            text = " ".join(filtered)
        
        return text


class SentenceCompressor(Compressor):
    """Compress by simplifying sentences."""
    
    name = "sentence"
    
    # Filler phrases that can be removed
    FILLERS = [
        r"as you can see,?\s*",
        r"it('s| is) (important|worth) (to note|noting) that\s*",
        r"please note that\s*",
        r"it should be noted that\s*",
        r"in other words,?\s*",
        r"that is to say,?\s*",
        r"to be more specific,?\s*",
        r"in this case,?\s*",
        r"as mentioned (earlier|before|above),?\s*",
        r"basically,?\s*",
        r"essentially,?\s*",
        r"fundamentally,?\s*",
        r"in fact,?\s*",
        r"as a matter of fact,?\s*",
        r"to be honest,?\s*",
        r"honestly,?\s*",
        r"frankly,?\s*",
        r"clearly,?\s*",
        r"obviously,?\s*",
        r"naturally,?\s*",
        r"certainly,?\s*",
        r"definitely,?\s*",
        r"absolutely,?\s*",
        r"generally speaking,?\s*",
        r"strictly speaking,?\s*",
        r"in general,?\s*",
        r"for the most part,?\s*",
        r"to some extent,?\s*",
        r"to a certain degree,?\s*",
        r"more or less,?\s*",
        r"kind of\s+",
        r"sort of\s+",
        r"you know,?\s*",
        r"I mean,?\s*",
        r"like,?\s+",
    ]
    
    def compress(
        self,
        text: str,
        level: CompressionLevel = CompressionLevel.MODERATE,
        **kwargs,
    ) -> str:
        """Simplify sentences."""
        if level == CompressionLevel.MINIMAL:
            return text
        
        # Remove filler phrases
        for filler in self.FILLERS:
            text = re.sub(filler, "", text, flags=re.IGNORECASE)
        
        if level in (CompressionLevel.AGGRESSIVE, CompressionLevel.EXTREME):
            # Simplify verbose constructions
            replacements = [
                (r"in order to", "to"),
                (r"due to the fact that", "because"),
                (r"for the purpose of", "for"),
                (r"in the event that", "if"),
                (r"in the process of", "while"),
                (r"with regard to", "about"),
                (r"with respect to", "about"),
                (r"in regard to", "about"),
                (r"pertaining to", "about"),
                (r"concerning", "about"),
                (r"at this point in time", "now"),
                (r"at the present time", "now"),
                (r"at the current time", "now"),
                (r"in the near future", "soon"),
                (r"in the event of", "if"),
                (r"in case of", "if"),
                (r"with the exception of", "except"),
                (r"on the basis of", "based on"),
                (r"in spite of the fact that", "although"),
                (r"despite the fact that", "although"),
                (r"regardless of the fact that", "although"),
                (r"a large number of", "many"),
                (r"a significant number of", "many"),
                (r"a great deal of", "much"),
                (r"a considerable amount of", "much"),
                (r"be able to", "can"),
                (r"has the ability to", "can"),
                (r"is capable of", "can"),
                (r"make use of", "use"),
                (r"utilize", "use"),
                (r"take into consideration", "consider"),
                (r"give consideration to", "consider"),
                (r"is an indication of", "indicates"),
                (r"is indicative of", "indicates"),
                (r"serves to", ""),
                (r"has a tendency to", "tends to"),
            ]
            
            for pattern, replacement in replacements:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text


class CodeCompressor(Compressor):
    """Compress code blocks and technical content."""
    
    name = "code"
    
    def compress(
        self,
        text: str,
        level: CompressionLevel = CompressionLevel.MODERATE,
        **kwargs,
    ) -> str:
        """Compress code-related content."""
        if level == CompressionLevel.MINIMAL:
            return text
        
        # Remove code comments (except docstrings)
        if level in (CompressionLevel.AGGRESSIVE, CompressionLevel.EXTREME):
            # Remove single-line comments
            text = re.sub(r"#[^\n]*(?=\n|$)", "", text)
            text = re.sub(r"//[^\n]*(?=\n|$)", "", text)
            
            # Remove multi-line comments (but not docstrings)
            text = re.sub(r"/\*(?!\*)[^*]*\*+(?:[^/*][^*]*\*+)*/", "", text)
        
        # Compress whitespace in code
        if level in (CompressionLevel.MODERATE, CompressionLevel.AGGRESSIVE, CompressionLevel.EXTREME):
            # Reduce indentation
            lines = text.split("\n")
            compressed_lines = []
            
            for line in lines:
                # Reduce leading whitespace
                stripped = line.lstrip()
                if stripped:
                    # Keep minimal indentation (2 spaces per level)
                    indent_count = (len(line) - len(stripped)) // 4
                    new_indent = "  " * indent_count
                    compressed_lines.append(new_indent + stripped)
                else:
                    compressed_lines.append("")
            
            text = "\n".join(compressed_lines)
        
        # Remove blank lines in code
        if level == CompressionLevel.EXTREME:
            text = re.sub(r"\n{2,}", "\n", text)
        
        return text


class SemanticCompressor(Compressor):
    """
    Semantic compression using LLM summarization.
    
    This compressor uses an LLM to intelligently summarize
    content while preserving key information.
    """
    
    name = "semantic"
    
    def __init__(
        self,
        llm_func: Optional[Callable[[str], str]] = None,
        max_output_tokens: int = 500,
    ):
        self.llm_func = llm_func
        self.max_output_tokens = max_output_tokens
    
    def compress(
        self,
        text: str,
        level: CompressionLevel = CompressionLevel.MODERATE,
        **kwargs,
    ) -> str:
        """Compress using LLM summarization."""
        if not self.llm_func:
            logger.warning("No LLM function provided for semantic compression")
            return text
        
        # Determine compression instructions based on level
        if level == CompressionLevel.MINIMAL:
            instruction = "Lightly condense this text, keeping most details:"
        elif level == CompressionLevel.MODERATE:
            instruction = "Summarize this text, keeping key information:"
        elif level == CompressionLevel.AGGRESSIVE:
            instruction = "Create a concise summary of the main points:"
        else:  # EXTREME
            instruction = "Extract only the essential information in minimal words:"
        
        prompt = f"""{instruction}

{text}

Compressed version:"""
        
        try:
            return self.llm_func(prompt)
        except Exception as e:
            logger.error(f"Semantic compression failed: {e}")
            return text


class MessageCompressor:
    """
    Compress conversation message history.
    
    Intelligently manages context window by compressing
    older messages while preserving recent context.
    """
    
    def __init__(
        self,
        token_counter: Optional[TokenCounter] = None,
        compressors: Optional[List[Compressor]] = None,
        summarizer: Optional[Callable[[str], str]] = None,
    ):
        self.token_counter = token_counter or TokenCounter()
        self.compressors = compressors or [
            WhitespaceCompressor(),
            PunctuationCompressor(),
            SentenceCompressor(),
        ]
        self.summarizer = summarizer
    
    def compress_message(
        self,
        message: Message,
        level: CompressionLevel = CompressionLevel.MODERATE,
    ) -> Message:
        """Compress a single message."""
        content = message.content
        
        for compressor in self.compressors:
            content = compressor.compress(content, level)
        
        return Message(
            role=message.role,
            content=content,
            name=message.name,
            metadata={**message.metadata, "compressed": True},
        )
    
    def compress_messages(
        self,
        messages: List[Message],
        max_tokens: int,
        preserve_recent: int = 5,
        preserve_system: bool = True,
    ) -> List[Message]:
        """
        Compress message history to fit within token limit.
        
        Args:
            messages: List of messages to compress
            max_tokens: Maximum token budget
            preserve_recent: Number of recent messages to keep uncompressed
            preserve_system: Whether to preserve system messages
        
        Returns:
            Compressed list of messages
        """
        current_tokens = self.token_counter.count_messages(messages)
        
        if current_tokens <= max_tokens:
            return messages
        
        # Separate messages
        system_messages = []
        other_messages = []
        
        for msg in messages:
            if preserve_system and msg.role == "system":
                system_messages.append(msg)
            else:
                other_messages.append(msg)
        
        # Preserve recent messages
        recent_messages = other_messages[-preserve_recent:] if preserve_recent > 0 else []
        old_messages = other_messages[:-preserve_recent] if preserve_recent > 0 else other_messages
        
        # Calculate available tokens for old messages
        system_tokens = sum(self.token_counter.count(m.content) for m in system_messages)
        recent_tokens = sum(self.token_counter.count(m.content) for m in recent_messages)
        available_tokens = max_tokens - system_tokens - recent_tokens - 100  # Buffer
        
        if available_tokens <= 0:
            # Can't fit old messages, just return system + recent
            return system_messages + recent_messages
        
        # Try progressive compression levels
        compressed_old = old_messages
        
        for level in CompressionLevel:
            compressed_old = [self.compress_message(m, level) for m in old_messages]
            old_tokens = sum(self.token_counter.count(m.content) for m in compressed_old)
            
            if old_tokens <= available_tokens:
                break
        
        # If still too large, summarize old messages
        old_tokens = sum(self.token_counter.count(m.content) for m in compressed_old)
        
        if old_tokens > available_tokens and self.summarizer:
            # Combine old messages into a summary
            old_content = "\n\n".join(
                f"{m.role}: {m.content}" for m in compressed_old
            )
            
            summary = self.summarizer(old_content)
            
            summary_message = Message(
                role="system",
                content=f"[Previous conversation summary]:\n{summary}",
                metadata={"is_summary": True},
            )
            
            return system_messages + [summary_message] + recent_messages
        
        return system_messages + compressed_old + recent_messages
    
    def sliding_window_compress(
        self,
        messages: List[Message],
        window_size: int,
        summarize_dropped: bool = True,
    ) -> Tuple[List[Message], Optional[str]]:
        """
        Keep most recent messages within window, optionally summarizing dropped ones.
        
        Args:
            messages: All messages
            window_size: Number of messages to keep
            summarize_dropped: Whether to summarize dropped messages
        
        Returns:
            Tuple of (kept messages, optional summary of dropped)
        """
        if len(messages) <= window_size:
            return messages, None
        
        dropped = messages[:-window_size]
        kept = messages[-window_size:]
        
        summary = None
        if summarize_dropped and self.summarizer:
            dropped_content = "\n\n".join(
                f"{m.role}: {m.content}" for m in dropped
            )
            summary = self.summarizer(dropped_content)
        
        return kept, summary


class ContextCompressor:
    """
    Main context compression interface.
    
    Provides unified compression for various content types
    with intelligent selection of compression techniques.
    """
    
    def __init__(
        self,
        token_counter: Optional[TokenCounter] = None,
        llm_summarizer: Optional[Callable[[str], str]] = None,
    ):
        self.token_counter = token_counter or TokenCounter()
        self.llm_summarizer = llm_summarizer
        
        # Initialize compressors
        self.compressors = {
            "whitespace": WhitespaceCompressor(),
            "punctuation": PunctuationCompressor(),
            "stopword": StopwordCompressor(),
            "sentence": SentenceCompressor(),
            "code": CodeCompressor(),
        }
        
        if llm_summarizer:
            self.compressors["semantic"] = SemanticCompressor(llm_summarizer)
        
        self.message_compressor = MessageCompressor(
            token_counter=self.token_counter,
            compressors=list(self.compressors.values()),
            summarizer=llm_summarizer,
        )
    
    def compress_text(
        self,
        text: str,
        target_tokens: Optional[int] = None,
        level: Optional[CompressionLevel] = None,
        techniques: Optional[List[str]] = None,
    ) -> CompressionResult:
        """
        Compress text.
        
        Args:
            text: Text to compress
            target_tokens: Target token count (optional)
            level: Compression level (optional, auto-determined if target_tokens set)
            techniques: List of technique names to use
        
        Returns:
            CompressionResult with compressed text
        """
        original_tokens = self.token_counter.count(text)
        
        # Determine compression level
        if level is None and target_tokens:
            ratio = target_tokens / original_tokens if original_tokens > 0 else 1.0
            if ratio >= 0.9:
                level = CompressionLevel.MINIMAL
            elif ratio >= 0.7:
                level = CompressionLevel.MODERATE
            elif ratio >= 0.5:
                level = CompressionLevel.AGGRESSIVE
            else:
                level = CompressionLevel.EXTREME
        
        level = level or CompressionLevel.MODERATE
        
        # Select compressors
        if techniques:
            selected = [self.compressors[t] for t in techniques if t in self.compressors]
        else:
            # Default compression pipeline
            selected = [
                self.compressors["whitespace"],
                self.compressors["punctuation"],
                self.compressors["sentence"],
            ]
            
            if level in (CompressionLevel.AGGRESSIVE, CompressionLevel.EXTREME):
                selected.append(self.compressors["stopword"])
        
        # Apply compression
        compressed = text
        techniques_used = []
        
        for compressor in selected:
            compressed = compressor.compress(compressed, level)
            techniques_used.append(compressor.name)
        
        # Check if more compression needed
        compressed_tokens = self.token_counter.count(compressed)
        
        if target_tokens and compressed_tokens > target_tokens and "semantic" in self.compressors:
            # Use semantic compression as last resort
            semantic = self.compressors["semantic"]
            compressed = semantic.compress(compressed, level)
            techniques_used.append("semantic")
            compressed_tokens = self.token_counter.count(compressed)
        
        ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        
        return CompressionResult(
            original_text=text,
            compressed_text=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=ratio,
            technique="+".join(techniques_used),
            metadata={"level": level.value},
        )
    
    def compress_messages(
        self,
        messages: List[Message],
        max_tokens: int,
        **kwargs,
    ) -> List[Message]:
        """Compress conversation messages."""
        return self.message_compressor.compress_messages(
            messages, max_tokens, **kwargs
        )
    
    def auto_compress(
        self,
        content: Any,
        target_tokens: int,
    ) -> Any:
        """
        Auto-detect content type and compress appropriately.
        
        Args:
            content: Text string or list of Messages
            target_tokens: Target token count
        
        Returns:
            Compressed content of same type
        """
        if isinstance(content, str):
            result = self.compress_text(content, target_tokens=target_tokens)
            return result.compressed_text
        
        elif isinstance(content, list) and all(isinstance(m, Message) for m in content):
            return self.compress_messages(content, target_tokens)
        
        elif isinstance(content, list) and all(isinstance(m, dict) for m in content):
            # Convert dicts to Messages
            messages = [
                Message(
                    role=m.get("role", "user"),
                    content=m.get("content", ""),
                    name=m.get("name"),
                )
                for m in content
            ]
            compressed = self.compress_messages(messages, target_tokens)
            return [
                {"role": m.role, "content": m.content, **({"name": m.name} if m.name else {})}
                for m in compressed
            ]
        
        return content


# Utility functions

def compress_text(
    text: str,
    level: CompressionLevel = CompressionLevel.MODERATE,
    techniques: Optional[List[str]] = None,
) -> str:
    """
    Compress text using specified level and techniques.
    
    Args:
        text: Text to compress
        level: Compression level
        techniques: List of technique names
    
    Returns:
        Compressed text
    """
    compressor = ContextCompressor()
    result = compressor.compress_text(text, level=level, techniques=techniques)
    return result.compressed_text


def compress_to_tokens(
    text: str,
    target_tokens: int,
    model: str = "gpt-4",
) -> str:
    """
    Compress text to fit within token budget.
    
    Args:
        text: Text to compress
        target_tokens: Maximum tokens
        model: Model for token counting
    
    Returns:
        Compressed text
    """
    counter = TokenCounter(model)
    compressor = ContextCompressor(token_counter=counter)
    result = compressor.compress_text(text, target_tokens=target_tokens)
    return result.compressed_text


def estimate_compression(
    text: str,
    level: CompressionLevel = CompressionLevel.MODERATE,
) -> Dict[str, Any]:
    """
    Estimate compression results without modifying text.
    
    Args:
        text: Text to analyze
        level: Compression level
    
    Returns:
        Estimation dictionary
    """
    compressor = ContextCompressor()
    result = compressor.compress_text(text, level=level)
    
    return {
        "original_tokens": result.original_tokens,
        "estimated_tokens": result.compressed_tokens,
        "savings": result.tokens_saved,
        "savings_percentage": result.savings_percentage,
        "compression_ratio": result.compression_ratio,
    }


__all__ = [
    "CompressionLevel",
    "CompressionResult",
    "Message",
    "TokenCounter",
    "Compressor",
    "WhitespaceCompressor",
    "PunctuationCompressor",
    "StopwordCompressor",
    "SentenceCompressor",
    "CodeCompressor",
    "SemanticCompressor",
    "MessageCompressor",
    "ContextCompressor",
    "compress_text",
    "compress_to_tokens",
    "estimate_compression",
]
