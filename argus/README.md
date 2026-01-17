# ARGUS Framework Documentation

## Overview

The `argus/` directory contains the complete ARGUS framework implementation. This documentation provides comprehensive guides for each module with code examples.

## Module Structure

```
argus/
├── core/              # Core configurations and utilities
│   └── llm/           # 27+ LLM provider integrations
├── agents/            # Multi-agent debate system
├── debate/            # Debate orchestration
├── knowledge/         # Knowledge base and retrieval
├── tools/             # 19+ tool integrations
│   └── integrations/  # Pre-built tools
├── embeddings/        # 16+ embedding models
├── evaluation/        # Benchmarking and evaluation
└── cli.py             # Command-line interface
```

---

## Quick Start Examples

### 1. LLM Providers

ARGUS supports 27+ LLM providers with a unified interface:

```python
from argus.core.llm import get_llm, list_providers

# List all available providers
print(list_providers())
# ['openai', 'anthropic', 'gemini', 'ollama', 'cohere', 'mistral', 
#  'groq', 'deepseek', 'xai', 'perplexity', 'nvidia', 'together', ...]

# Use OpenAI
llm = get_llm("openai", model="gpt-4o")
response = llm.generate("Explain quantum computing in simple terms")
print(response.content)

# Use Anthropic Claude
llm = get_llm("anthropic", model="claude-3-5-sonnet-20241022")
response = llm.generate("What are the benefits of renewable energy?")

# Use local Ollama (free, no API key)
llm = get_llm("ollama", model="llama3.2")
response = llm.generate("Write a haiku about AI")

# Streaming responses
for chunk in llm.stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

### 2. Embedding Models

16+ embedding providers for semantic search and RAG:

```python
from argus.embeddings import get_embedding, list_embedding_providers

# List available embedding providers
print(list_embedding_providers())
# ['sentence_transformers', 'openai', 'cohere', 'huggingface', 
#  'voyage', 'mistral', 'ollama', 'google', 'azure', ...]

# Local embedding (free, no API key)
embedder = get_embedding("sentence_transformers", model="all-MiniLM-L6-v2")
vectors = embedder.embed_documents(["Hello world", "AI is amazing", "Python programming"])
print(f"Dimension: {len(vectors[0])}")  # 384

# Query embedding for search
query_vec = embedder.embed_query("What is machine learning?")

# OpenAI embeddings (requires API key)
embedder = get_embedding("openai", model="text-embedding-3-small")
vectors = embedder.embed_documents(["Document 1", "Document 2"])

# Cohere embeddings
embedder = get_embedding("cohere", model="embed-english-v3.0")
vectors = embedder.embed_documents(["Hello", "World"])
```

### 3. Tools

19+ pre-built tools for agents:

```python
from argus.tools.integrations import (
    DuckDuckGoTool, WikipediaTool, ArxivTool,
    PythonReplTool, ShellTool, FileSystemTool,
    YahooFinanceTool, WebScraperTool
)

# Free web search (no API key needed)
search = DuckDuckGoTool()
result = search(query="latest AI research 2024", max_results=5)
for r in result.data["results"]:
    print(f"- {r['title']}: {r['url']}")

# Wikipedia lookup
wiki = WikipediaTool()
result = wiki(query="Artificial Intelligence", action="summary", sentences=3)
print(result.data["summary"])

# ArXiv paper search
arxiv = ArxivTool()
result = arxiv(query="transformer neural networks", max_results=5)
for paper in result.data["results"]:
    print(f"📄 {paper['title']} - {paper['pdf_url']}")

# Python code execution
repl = PythonReplTool()
result = repl(code="import math; print(math.pi * 2)")
print(result.data["output"])  # 6.283185307179586

# Stock data
finance = YahooFinanceTool()
result = finance(symbol="AAPL", action="quote")
print(f"Apple stock: ${result.data['price']}")

# Web scraping
scraper = WebScraperTool()
result = scraper(url="https://example.com", extract="text")
print(result.data["content"][:500])
```

### 4. Debate System

Multi-agent structured argumentation:

```python
from argus.debate import DebateOrchestrator, DebateConfig
from argus.agents import ProponentAgent, OpponentAgent, JudgeAgent

# Configure debate
config = DebateConfig(
    topic="Should AI systems be required to explain their decisions?",
    num_rounds=3,
    judge_model="gpt-4o",
    agent_model="gpt-4o-mini",
)

# Initialize orchestrator
orchestrator = DebateOrchestrator(config)

# Run debate
result = orchestrator.run()

# Access results
print(f"Winner: {result.winner}")
print(f"Final score: {result.final_score}")
print(f"Summary: {result.summary}")

# Iterate through rounds
for round in result.rounds:
    print(f"\n=== Round {round.number} ===")
    print(f"Proponent: {round.proponent_argument[:200]}...")
    print(f"Opponent: {round.opponent_argument[:200]}...")
```

### 5. Knowledge Base & Retrieval

Build and query knowledge bases:

```python
from argus.knowledge import KnowledgeBase, DocumentLoader, EmbeddingGenerator

# Load documents
loader = DocumentLoader()
docs = loader.load_directory("./papers/")
docs += loader.load_pdf("./report.pdf")
docs += loader.load_url("https://arxiv.org/abs/2401.00001")

# Create embeddings
generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
chunks = loader.chunk_documents(docs, chunk_size=500, overlap=50)
embeddings = generator.embed_chunks(chunks)

# Build knowledge base
kb = KnowledgeBase()
kb.add_documents(chunks, embeddings)

# Semantic search
results = kb.search("What is the attention mechanism?", top_k=5)
for result in results:
    print(f"Score: {result.score:.3f} - {result.text[:100]}...")

# Hybrid search (BM25 + semantic)
results = kb.hybrid_search("transformer architecture", top_k=10, alpha=0.7)
```

---

## Module Documentation

| Module | Documentation | Description |
|--------|---------------|-------------|
| [core/](core/README.md) | Configuration & LLM | Core settings and 27+ LLM providers |
| [agents/](agents/README.md) | Agent System | Multi-agent debate agents |
| [debate/](debate/README.md) | Debate Framework | Orchestration and scoring |
| [knowledge/](knowledge/README.md) | Knowledge Base | Document processing and retrieval |
| [tools/](tools/README.md) | Tools | 19+ pre-built tool integrations |
| [embeddings/](embeddings/README.md) | Embeddings | 16+ embedding model integrations |
| [evaluation/](evaluation/README.md) | Evaluation | Benchmarking and metrics |

---

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```env
# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
COHERE_API_KEY=...
MISTRAL_API_KEY=...
GROQ_API_KEY=gsk_...

# Embedding Providers
VOYAGE_API_KEY=...
JINA_API_KEY=...

# Tool APIs
TAVILY_API_KEY=tvly-...
BRAVE_API_KEY=BSA...
GITHUB_TOKEN=ghp_...

# Local Options
OLLAMA_HOST=http://localhost:11434
```

### Programmatic Configuration

```python
from argus.core.config import ArgusConfig

config = ArgusConfig(
    default_provider="openai",
    default_model="gpt-4o",
    temperature=0.7,
    max_tokens=4096,
)
```

---

## CLI Usage

```bash
# Run a debate
argus debate "Is remote work better than office work?" --rounds 3

# List LLM providers
argus providers

# List tools
argus tools

# Run benchmark
argus benchmark --dataset mmlu --samples 100

# Generate report
argus report debate_results.json --output report.html
```

---

## Version

**ARGUS v1.4.1** - Production Release
