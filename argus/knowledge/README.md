# ARGUS Knowledge Module

## Overview

The `knowledge/` module handles document processing, embedding generation, and semantic retrieval.

## Components

| Component | Description |
|-----------|-------------|
| `DocumentLoader` | Load documents from various sources |
| `Chunker` | Split documents into chunks |
| `EmbeddingGenerator` | Generate embeddings for chunks |
| `KnowledgeBase` | Store and search documents |
| `HybridRetriever` | BM25 + semantic search |

## Quick Start

```python
from argus.knowledge import (
    DocumentLoader,
    EmbeddingGenerator,
    KnowledgeBase,
)

# Load documents
loader = DocumentLoader()
docs = loader.load_pdf("research_paper.pdf")
docs += loader.load_directory("./papers/", pattern="*.pdf")
docs += loader.load_url("https://arxiv.org/abs/2301.00001")

# Chunk documents
chunks = loader.chunk_documents(docs, chunk_size=500, overlap=50)
print(f"Created {len(chunks)} chunks")

# Generate embeddings
generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
embeddings = generator.embed_chunks(chunks)

# Build knowledge base
kb = KnowledgeBase()
kb.add_documents(chunks, embeddings)

# Search
results = kb.search("What is attention mechanism?", top_k=5)
for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Text: {result.text[:200]}...")
    print()
```

## Document Loading

```python
from argus.knowledge import DocumentLoader

loader = DocumentLoader()

# PDF files
docs = loader.load_pdf("document.pdf")

# Multiple PDFs
docs = loader.load_directory("./papers/", pattern="*.pdf")

# Web pages
docs = loader.load_url("https://example.com/article")

# Multiple URLs
urls = ["https://...", "https://..."]
docs = loader.load_urls(urls)

# Plain text
docs = loader.load_text("path/to/file.txt")

# Markdown
docs = loader.load_markdown("README.md")

# HTML
docs = loader.load_html("page.html")
```

## Chunking

```python
from argus.knowledge import Chunker, ChunkConfig

# Configure chunking
config = ChunkConfig(
    chunk_size=500,       # Characters per chunk
    chunk_overlap=50,     # Overlap between chunks
    separator="\n\n",     # Split on paragraphs
    length_function=len,  # How to measure length
)

chunker = Chunker(config)
chunks = chunker.chunk_documents(docs)

# Or use DocumentLoader
chunks = loader.chunk_documents(docs, chunk_size=500, overlap=50)
```

## Embedding Generation

```python
from argus.knowledge import EmbeddingGenerator

# Local model (free)
generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")

# Different models
generator = EmbeddingGenerator(model_name="all-mpnet-base-v2")
generator = EmbeddingGenerator(model_name="multi-qa-MiniLM-L6-cos-v1")

# Generate embeddings
embeddings = generator.embed_chunks(chunks)
print(f"Generated {len(embeddings)} embeddings")
print(f"Dimension: {embeddings[0].dimension}")

# Embed single text
vector = generator.embed_query("What is machine learning?")
```

## Knowledge Base

```python
from argus.knowledge import KnowledgeBase

# Create knowledge base
kb = KnowledgeBase()

# Add documents
kb.add_documents(chunks, embeddings)

# Semantic search
results = kb.search("attention mechanism", top_k=10)

# Hybrid search (BM25 + semantic)
results = kb.hybrid_search(
    query="transformer architecture",
    top_k=10,
    alpha=0.7,  # 0=BM25 only, 1=semantic only
)

# Filter search
results = kb.search(
    "neural networks",
    top_k=10,
    filters={"source": "arxiv", "year": 2024},
)

# Save and load
kb.save("knowledge_base.pkl")
kb = KnowledgeBase.load("knowledge_base.pkl")
```

## Hybrid Retrieval

```python
from argus.knowledge import HybridRetriever

retriever = HybridRetriever(
    knowledge_base=kb,
    bm25_weight=0.3,
    semantic_weight=0.7,
    rerank=True,
    rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
)

# Retrieve with reranking
results = retriever.retrieve(
    query="What are the benefits of attention?",
    top_k=5,
    initial_k=50,  # Retrieve 50, rerank to top 5
)
```

## Similarity Functions

```python
from argus.knowledge import cosine_similarity, batch_cosine_similarity

# Single comparison
v1 = [0.1, 0.2, 0.3]
v2 = [0.2, 0.3, 0.4]
similarity = cosine_similarity(v1, v2)
print(f"Similarity: {similarity:.3f}")

# Batch comparison
query = [0.1, 0.2, 0.3]
corpus = [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.1, 0.1, 0.1]]
similarities = batch_cosine_similarity(query, corpus)
print(f"Similarities: {similarities}")
```

## Complete RAG Pipeline

```python
from argus.knowledge import DocumentLoader, EmbeddingGenerator, KnowledgeBase
from argus.core.llm import get_llm

# 1. Load and process documents
loader = DocumentLoader()
docs = loader.load_directory("./data/")
chunks = loader.chunk_documents(docs, chunk_size=500)

# 2. Generate embeddings
generator = EmbeddingGenerator()
embeddings = generator.embed_chunks(chunks)

# 3. Build knowledge base
kb = KnowledgeBase()
kb.add_documents(chunks, embeddings)

# 4. Query function
def rag_query(question: str, llm, kb, top_k: int = 5):
    # Retrieve context
    results = kb.hybrid_search(question, top_k=top_k)
    context = "\n\n".join([r.text for r in results])
    
    # Generate answer
    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
    
    response = llm.generate(prompt)
    return response.content, results

# 5. Use
llm = get_llm("openai", model="gpt-4o")
answer, sources = rag_query("What is attention?", llm, kb)
print(f"Answer: {answer}")
print(f"Sources: {len(sources)} chunks")
```
