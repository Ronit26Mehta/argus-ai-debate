"""
AST-aware code chunker for fsociety.

Two-level chunking: parent chunks (800-1500 tokens) for full context,
child chunks (100-300 tokens) for precision search.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Supported file extensions for code analysis
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".java", ".php", ".rb",
    ".c", ".cpp", ".cs", ".rs", ".kt", ".swift", ".yaml", ".yml", ".json",
    ".env", ".toml", ".dockerfile", ".tf", ".hcl", ".xml", ".sql", ".sh",
    ".bash", ".ps1", ".md", ".html", ".css",
}

# Config file patterns (always ingest even if not "code")
CONFIG_PATTERNS = {
    "package.json", "requirements.txt", "go.mod", "Cargo.toml",
    "Gemfile", "pom.xml", "build.gradle", "docker-compose.yml",
    "Dockerfile", ".env", ".env.example", ".gitignore",
    "nginx.conf", "apache.conf", "Makefile",
}


@dataclass
class CodeChunk:
    """A chunk of code with metadata."""
    text: str
    filename: str
    language: str
    line_start: int
    line_end: int
    chunk_type: str = "parent"    # "parent" or "child"
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """Rough token estimate (4 chars per token)."""
        return len(self.text) // 4


class CodeChunker:
    """
    AST-aware code chunker with parent-child architecture.

    Parent chunks (800-1500 tokens): one per function/class/logical block.
    Child chunks (100-300 tokens): finer-grained sub-units.
    """

    def __init__(
        self,
        parent_max_tokens: int = 1500,
        parent_min_tokens: int = 100,
        child_max_tokens: int = 300,
    ):
        self.parent_max_tokens = parent_max_tokens
        self.parent_min_tokens = parent_min_tokens
        self.child_max_tokens = child_max_tokens

    def chunk_directory(self, directory: str | Path) -> list[CodeChunk]:
        """Recursively chunk all supported files in a directory."""
        directory = Path(directory)
        chunks: list[CodeChunk] = []

        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return chunks

        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix not in SUPPORTED_EXTENSIONS and file_path.name not in CONFIG_PATTERNS:
                continue
            # Skip common vendor / build directories
            parts = file_path.parts
            if any(p in (".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build") for p in parts):
                continue

            try:
                file_chunks = self.chunk_file(file_path)
                chunks.extend(file_chunks)
            except Exception as e:
                logger.warning(f"Could not chunk {file_path}: {e}")

        logger.info(f"Chunked {len(chunks)} pieces from {directory}")
        return chunks

    def chunk_file(self, file_path: str | Path) -> list[CodeChunk]:
        """Chunk a single file into parent and child chunks."""
        file_path = Path(file_path)
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Cannot read {file_path}: {e}")
            return []

        language = self._detect_language(file_path)
        lines = text.splitlines(keepends=True)

        # Try AST-based chunking for Python
        if language == "python":
            chunks = self._chunk_python_ast(text, str(file_path), lines)
            if chunks:
                return chunks

        # Fallback: line-based chunking
        return self._chunk_by_lines(text, str(file_path), language, lines)

    def _chunk_python_ast(self, text: str, filename: str, lines: list[str]) -> list[CodeChunk]:
        """AST-based chunking for Python files."""
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []

        chunks: list[CodeChunk] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 10
                block_text = "".join(lines[start:end])

                class_name = None
                func_name = None
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                else:
                    func_name = node.name

                parent = CodeChunk(
                    text=block_text,
                    filename=filename,
                    language="python",
                    line_start=start + 1,
                    line_end=end,
                    chunk_type="parent",
                    function_name=func_name,
                    class_name=class_name,
                )
                chunks.append(parent)

                # Generate child chunks from the parent
                children = self._split_into_children(block_text, filename, "python", start + 1)
                chunks.extend(children)

        # If no AST nodes found, fall back to line chunking
        if not chunks:
            return []

        return chunks

    def _chunk_by_lines(
        self, text: str, filename: str, language: str, lines: list[str]
    ) -> list[CodeChunk]:
        """Simple line-based chunking with overlap."""
        chunks: list[CodeChunk] = []
        chars_per_token = 4
        max_chars = self.parent_max_tokens * chars_per_token

        current: list[str] = []
        current_start = 1
        current_len = 0

        for i, line in enumerate(lines, 1):
            current.append(line)
            current_len += len(line)

            if current_len >= max_chars:
                chunk_text = "".join(current)
                parent = CodeChunk(
                    text=chunk_text,
                    filename=filename,
                    language=language,
                    line_start=current_start,
                    line_end=i,
                    chunk_type="parent",
                )
                chunks.append(parent)

                children = self._split_into_children(chunk_text, filename, language, current_start)
                chunks.extend(children)

                current = []
                current_start = i + 1
                current_len = 0

        # Last chunk
        if current:
            chunk_text = "".join(current)
            if len(chunk_text.strip()) > 0:
                parent = CodeChunk(
                    text=chunk_text,
                    filename=filename,
                    language=language,
                    line_start=current_start,
                    line_end=len(lines),
                    chunk_type="parent",
                )
                chunks.append(parent)

        return chunks

    def _split_into_children(
        self, text: str, filename: str, language: str, parent_start: int
    ) -> list[CodeChunk]:
        """Split a parent chunk into child chunks."""
        children: list[CodeChunk] = []
        chars_per_token = 4
        max_chars = self.child_max_tokens * chars_per_token
        lines = text.splitlines(keepends=True)

        current: list[str] = []
        current_start = parent_start
        current_len = 0

        for i, line in enumerate(lines):
            current.append(line)
            current_len += len(line)

            if current_len >= max_chars:
                child_text = "".join(current)
                children.append(CodeChunk(
                    text=child_text,
                    filename=filename,
                    language=language,
                    line_start=current_start,
                    line_end=parent_start + i,
                    chunk_type="child",
                ))
                current = []
                current_start = parent_start + i + 1
                current_len = 0

        if current and len("".join(current).strip()) > 20:
            children.append(CodeChunk(
                text="".join(current),
                filename=filename,
                language=language,
                line_start=current_start,
                line_end=parent_start + len(lines) - 1,
                chunk_type="child",
            ))

        return children

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        ext_map = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".jsx": "javascript", ".tsx": "typescript", ".go": "go",
            ".java": "java", ".php": "php", ".rb": "ruby",
            ".c": "c", ".cpp": "cpp", ".cs": "csharp",
            ".rs": "rust", ".kt": "kotlin", ".swift": "swift",
            ".yaml": "yaml", ".yml": "yaml", ".json": "json",
            ".toml": "toml", ".xml": "xml", ".sql": "sql",
            ".sh": "bash", ".bash": "bash", ".ps1": "powershell",
            ".md": "markdown", ".html": "html", ".css": "css",
            ".dockerfile": "dockerfile", ".tf": "terraform",
        }
        return ext_map.get(file_path.suffix.lower(), "text")
