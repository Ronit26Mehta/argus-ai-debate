# ARGUS Embeddings Module

## Overview

The `embeddings/` module provides 16+ embedding model integrations for semantic search, RAG, and similarity computations.

## Structure

```
embeddings/
├── __init__.py
├── base.py               # BaseEmbedding interface
├── registry.py           # Provider registry
├── sentence_transformers.py  # Local (free)
├── fastembed.py          # Local (fast)
├── ollama.py             # Local
├── openai.py             # OpenAI
├── cohere.py             # Cohere
├── huggingface.py        # HuggingFace API
├── voyage.py             # Voyage AI
├── mistral.py            # Mistral
├── google.py             # Google/Gemini
├── azure.py              # Azure OpenAI
├── together.py           # Together AI
├── nvidia.py             # NVIDIA NIM
├── jina.py               # Jina AI
├── nomic.py              # Nomic
├── bedrock.py            # AWS Bedrock
└── fireworks.py          # Fireworks
```

---

## Available Providers (16)

### Local (Free, No API Key)

| Provider | Models | Dimension |
|----------|--------|-----------|
| `sentence_transformers` | all-MiniLM-L6-v2, all-mpnet-base-v2 | 384-768 |
| `fastembed` | BAAI/bge-small-en, bge-base-en | 384-768 |
| `ollama` | nomic-embed-text, mxbai-embed-large | 768-1024 |

### Cloud APIs

| Provider | Models | Dimension | API Key Env |
|----------|--------|-----------|-------------|
| `openai` | text-embedding-3-small/large | 1536-3072 | `OPENAI_API_KEY` |
| `cohere` | embed-english-v3.0 | 1024 | `COHERE_API_KEY` |
| `huggingface` | BAAI/bge-*, intfloat/e5-* | 384-1024 | `HF_TOKEN` |
| `voyage` | voyage-3, voyage-code-3 | 512-1024 | `VOYAGE_API_KEY` |
| `mistral` | mistral-embed | 1024 | `MISTRAL_API_KEY` |
| `google` | text-embedding-004 | 768 | `GOOGLE_API_KEY` |
| `azure` | text-embedding-ada-002 | 1536 | `AZURE_OPENAI_API_KEY` |
| `together` | BAAI/bge-base-en-v1.5 | 768 | `TOGETHER_API_KEY` |
| `nvidia` | nvidia/nv-embedqa-e5-v5 | 1024 | `NVIDIA_API_KEY` |
| `jina` | jina-embeddings-v3 | 1024 | `JINA_API_KEY` |
| `nomic` | nomic-embed-text-v1.5 | 768 | `NOMIC_API_KEY` |
| `bedrock` | amazon.titan-embed-text-v2 | 1024 | AWS credentials |
| `fireworks` | nomic-ai/nomic-embed-text | 768 | `FIREWORKS_API_KEY` |

---

## Usage Examples

### Basic Usage

```python
from argus.embeddings import get_embedding, list_embedding_providers

# List all providers
providers = list_embedding_providers()
print(providers)
# ['sentence_transformers', 'fastembed', 'ollama', 'openai', 'cohere', ...]

# Get embedding model
embedder = get_embedding("sentence_transformers", model="all-MiniLM-L6-v2")

# Embed documents
texts = ["Hello world", "Machine learning is cool", "Python programming"]
vectors = embedder.embed_documents(texts)

print(f"Number of vectors: {len(vectors)}")
print(f"Vector dimension: {len(vectors[0])}")

# Embed single query
query_vector = embedder.embed_query("What is AI?")
```

### Local Embeddings (Free)

```python
from argus.embeddings import (
    SentenceTransformersEmbedding,
    FastEmbedEmbedding,
    OllamaEmbedding,
)

# Sentence Transformers (most popular)
embedder = SentenceTransformersEmbedding(model="all-MiniLM-L6-v2")
vectors = embedder.embed_documents(["Hello", "World"])

# With different models
embedder = SentenceTransformersEmbedding(model="all-mpnet-base-v2")  # 768 dim
embedder = SentenceTransformersEmbedding(model="paraphrase-multilingual-MiniLM-L12-v2")

# FastEmbed (faster, smaller memory)
embedder = FastEmbedEmbedding(model="BAAI/bge-small-en-v1.5")
vectors = embedder.embed_documents(["Fast", "Embedding"])

# Ollama (local server)
embedder = OllamaEmbedding(model="nomic-embed-text")
vectors = embedder.embed_documents(["Ollama", "Embeddings"])
```

### Cloud Embeddings

```python
from argus.embeddings import (
    OpenAIEmbedding,
    CohereEmbedding,
    VoyageEmbedding,
    GoogleEmbedding,
)

# OpenAI
embedder = OpenAIEmbedding(model="text-embedding-3-small")
vectors = embedder.embed_documents(["OpenAI embeddings"])

# With custom dimensions (text-embedding-3 only)
embedder = OpenAIEmbedding(model="text-embedding-3-large", dimensions=1024)

# Cohere
embedder = CohereEmbedding(model="embed-english-v3.0")
# Different input types for documents vs queries
doc_vectors = embedder.embed_documents(["Document text"])
query_vector = embedder.embed_query("Search query")

# Voyage AI (RAG-optimized)
embedder = VoyageEmbedding(model="voyage-3")
vectors = embedder.embed_documents(["Voyage embeddings"])

# Google Gemini
embedder = GoogleEmbedding(model="text-embedding-004")
vectors = embedder.embed_documents(["Google embeddings"])
```

### Batch Processing

```python
from argus.embeddings import get_embedding

embedder = get_embedding("sentence_transformers", model="all-MiniLM-L6-v2")

# Large document list
documents = [f"Document {i}" for i in range(1000)]

# Batch processing (automatic chunking)
vectors = embedder.embed_batch(documents, batch_size=64)
print(f"Embedded {len(vectors)} documents")
```

### Similarity Search

```python
import numpy as np
from argus.embeddings import get_embedding

embedder = get_embedding("sentence_transformers")

# Corpus
documents = [
    "Python is a programming language",
    "Machine learning uses algorithms",
    "The weather is sunny today",
    "Deep learning is a subset of ML",
    "I love coffee in the morning",
]

# Embed corpus
corpus_vectors = embedder.embed_documents(documents)

# Query
query = "What is artificial intelligence?"
query_vector = embedder.embed_query(query)

# Compute similarities
similarities = [
    embedder.similarity(query_vector, doc_vec)
    for doc_vec in corpus_vectors
]

# Rank results
ranked = sorted(
    zip(documents, similarities),
    key=lambda x: x[1],
    reverse=True,
)

print(f"Query: {query}")
for doc, score in ranked[:3]:
    print(f"  {score:.3f}: {doc}")
```

### Using with Knowledge Base

```python
from argus.embeddings import get_embedding
from argus.knowledge import KnowledgeBase

# Initialize embedder
embedder = get_embedding("openai", model="text-embedding-3-small")

# Create knowledge base
kb = KnowledgeBase()

# Add documents with embeddings
documents = ["Doc 1 content", "Doc 2 content", "Doc 3 content"]
vectors = embedder.embed_documents(documents)

for doc, vec in zip(documents, vectors):
    kb.add(doc, vec)

# Search
query_vec = embedder.embed_query("search term")
results = kb.search(query_vec, top_k=5)
```

---

## Configuration

### Environment Variables

```env
# OpenAI
OPENAI_API_KEY=sk-...

# Cohere
COHERE_API_KEY=...

# HuggingFace
HF_TOKEN=hf_...

# Voyage
VOYAGE_API_KEY=pa-...

# Jina
JINA_API_KEY=jina_...

# Google
GOOGLE_API_KEY=...

# Azure
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com/

# Others
MISTRAL_API_KEY=...
NVIDIA_API_KEY=nvapi-...
TOGETHER_API_KEY=...
FIREWORKS_API_KEY=fw_...
NOMIC_API_KEY=...

# Local
OLLAMA_HOST=http://localhost:11434
```

### Programmatic Configuration

```python
from argus.core.config import EmbeddingProviderConfig

config = EmbeddingProviderConfig(
    default_embedding_provider="openai",
    default_embedding_model="text-embedding-3-small",
    embedding_batch_size=64,
    embedding_normalize=True,
)

print(config.get_available_providers())
```

---

## Creating Custom Embeddings

```python
from argus.embeddings import BaseEmbedding, EmbeddingRegistry

class MyCustomEmbedding(BaseEmbedding):
    """Custom embedding provider."""
    
    name = "my_embedding"
    
    def __init__(self, model: str = "my-model", **kwargs):
        super().__init__(model=model, **kwargs)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Your implementation
        return [[0.1, 0.2, ...] for _ in texts]
    
    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

# Register
EmbeddingRegistry.register("my_embedding", MyCustomEmbedding)

# Use
from argus.embeddings import get_embedding
embedder = get_embedding("my_embedding")
```

---

## Performance Tips

1. **Use local models for development** - SentenceTransformers and FastEmbed are free
2. **Batch your requests** - Use `embed_batch()` for large document sets
3. **Cache embeddings** - Store computed embeddings to avoid recomputation
4. **Match dimensions** - Ensure query and document embeddings use same model
5. **Consider dimensionality** - Smaller dimensions = faster, larger = more accurate
