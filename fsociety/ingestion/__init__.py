"""fsociety ingestion engine — Code → Embeddings → Agent Context."""
from fsociety.ingestion.code_chunker import CodeChunker
from fsociety.ingestion.dependency_scanner import DependencyScanner
from fsociety.ingestion.git_analyzer import GitAnalyzer

__all__ = ["CodeChunker", "DependencyScanner", "GitAnalyzer"]
