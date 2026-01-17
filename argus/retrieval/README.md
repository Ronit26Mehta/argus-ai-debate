# ARGUS Retrieval Module

## Overview

The `retrieval/` module provides **hybrid retrieval** combining BM25 sparse search with dense vector search and cross-encoder reranking.

## Components

| File | Description |
|------|-------------|
| `hybrid.py` | Hybrid BM25 + dense retrieval |
| `reranker.py` | Cross-encoder reranking |
| `cite_critique.py` | Citation quality assessment |

## Quick Start

```python
from argus.retrieval import HybridRetriever

# Create retriever
retriever = HybridRetriever(
    embedding_model="all-MiniLM-L6-v2",
    lambda_param=0.7,  # 0=BM25 only, 1=dense only
    use_reranker=True,
)

# Index documents
retriever.index_chunks(chunks)

# Search
results = retriever.retrieve("attention mechanism", top_k=10)
for r in results:
    print(f"[{r.rank}] Score: {r.score:.3f} - {r.text[:100]}...")
```

## Hybrid Retrieval

```python
from argus.retrieval import HybridRetriever, RetrievalConfig

config = RetrievalConfig(
    embedding_model="all-MiniLM-L6-v2",
    lambda_param=0.7,    # Dense weight (1-lambda for BM25)
    top_k=10,
    initial_k=100,       # Candidates before reranking
    use_reranker=True,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
)

retriever = HybridRetriever(config)

# Index
retriever.index_chunks(chunks)

# Retrieve with different fusion methods
results = retriever.retrieve(query, fusion="rrf")  # Reciprocal Rank Fusion
results = retriever.retrieve(query, fusion="weighted")  # Weighted combination
```

## Reranking

```python
from argus.retrieval import CrossEncoderReranker

reranker = CrossEncoderReranker(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Rerank results
query = "What is attention?"
candidates = [chunk1, chunk2, chunk3, ...]

reranked = reranker.rerank(query, candidates, top_k=5)
for r in reranked:
    print(f"Score: {r.score:.3f} - {r.text[:50]}...")
```

## Citation Critique

```python
from argus.retrieval import CiteCritique

critic = CiteCritique(llm=llm)

# Assess source quality
assessment = critic.assess(
    claim="AI will transform healthcare",
    source="FDA report on AI medical devices",
    evidence="521 AI devices approved in 2023",
)

print(f"Relevance: {assessment.relevance:.2f}")
print(f"Credibility: {assessment.credibility:.2f}")
print(f"Recency: {assessment.recency:.2f}")
print(f"Overall: {assessment.overall:.2f}")
print(f"Critique: {assessment.critique}")
```

## Fusion Methods

### Reciprocal Rank Fusion (RRF)
```python
# RRF combines rankings, not scores
# Score = sum(1 / (k + rank))
results = retriever.retrieve(query, fusion="rrf", k=60)
```

### Weighted Combination
```python
# Weighted = lambda * dense_score + (1-lambda) * bm25_score
results = retriever.retrieve(query, fusion="weighted", lambda_param=0.7)
```

## Performance Tips

1. **Initial retrieval**: Set `initial_k` high (50-100) for reranking
2. **Lambda tuning**: Start at 0.7, tune based on query types
3. **Reranker**: Adds latency but improves precision significantly
4. **Batch indexing**: Index documents in batches for speed

```python
# Batch indexing
retriever.index_chunks(chunks, batch_size=100)

# Pre-compute BM25 index
retriever.build_bm25_index(chunks)
```
