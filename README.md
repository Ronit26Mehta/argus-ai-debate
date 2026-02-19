# ARGUS

**Agentic Research & Governance Unified System**

*A debate-native, multi-agent AI framework for evidence-based reasoning with structured argumentation, decision-theoretic planning, and full provenance tracking.*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/argus-debate-ai.svg)](https://pypi.org/project/argus-debate-ai/2.5/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue.svg)](https://mypy.readthedocs.io/)

---

## Table of Contents

- [Overview](#overview)
- [Key Innovations](#key-innovations)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [LLM Providers](#llm-providers)
- [External Connectors](#external-connectors)
- [Visualization & Plotting](#visualization--plotting)
- [Argus Terminal (TUI)](#argus-terminal-tui)
- [Argus-Viz (Streamlit Sandbox)](#argus-viz-streamlit-sandbox)
- [CRUX Protocol](#crux-protocol)
- [Command Line Interface](#command-line-interface)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Core Components](#core-components)
- [Algorithms](#algorithms)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

ARGUS implements **Research Debate Chain (RDC)** - a novel approach to AI reasoning that structures knowledge evaluation as multi-agent debates. Instead of single-pass inference, ARGUS orchestrates specialist agents that gather evidence, generate rebuttals, and render verdicts through Bayesian aggregation.

### Why ARGUS?

Traditional LLM applications suffer from:
- **Hallucination**: Models generate plausible but incorrect information
- **Overconfidence**: No calibrated uncertainty estimates
- **Opacity**: Black-box reasoning with no audit trail
- **Single-Point Failure**: One model, one perspective

ARGUS addresses these through:
- **Adversarial Debate**: Multiple agents challenge claims with evidence
- **Bayesian Aggregation**: Calibrated confidence through probability theory
- **Full Provenance**: Every claim traced to its source
- **Multi-Model Support**: Use different LLMs for different roles

---

## Key Innovations

### Conceptual Debate Graph (C-DAG)
A directed graph structure where propositions, evidence, and rebuttals are nodes with signed edges representing support/attack relationships. The graph enables:
- Structured argument representation
- Influence propagation via Bayesian updating
- Conflict detection and resolution
- Visual debugging and analysis

### Evidence-Directed Debate Orchestration (EDDO)
Algorithm for managing multi-round debates with configurable stopping criteria:
- Convergence detection (posterior stability)
- Maximum rounds enforcement
- Budget-based termination
- Information gain thresholds

### Value of Information Planning
Decision-theoretic experiment selection using Expected Information Gain (EIG):
- Prioritize high-value evidence gathering
- Optimal resource allocation under constraints
- Monte Carlo estimation of information value

### Full Provenance Tracking
PROV-O compatible ledger with hash-chain integrity:
- W3C standard compliance
- Cryptographic attestations
- Complete audit trails
- Tamper detection

---

## Features

### Multi-Agent Debate System

| Agent | Role | Capabilities |
|-------|------|--------------|
| **Moderator** | Orchestration | Creates debate agendas, manages rounds, evaluates stopping criteria, breaks ties |
| **Specialist** | Evidence Gathering | Domain-specific research, hybrid retrieval, source quality assessment |
| **Refuter** | Challenge Generation | Counter-evidence, methodological critiques, logical fallacy detection |
| **Jury** | Verdict Rendering | Bayesian aggregation, confidence calibration, label assignment |

### Conceptual Debate Graph (C-DAG)

**Node Types:**
| Type | Description | Attributes |
|------|-------------|------------|
| `Proposition` | Main claims under evaluation | text, prior, domain, status |
| `Evidence` | Supporting/attacking information | polarity, confidence, source, type |
| `Rebuttal` | Challenges to evidence | target_id, strength, rebuttal_type |
| `Finding` | Intermediate conclusions | derived_from, confidence |
| `Assumption` | Underlying premises | explicit, challenged |

**Edge Types:**
| Type | Polarity | Description |
|------|----------|-------------|
| `SUPPORTS` | +1 | Evidence supporting a proposition |
| `ATTACKS` | -1 | Evidence challenging a proposition |
| `REBUTS` | -1 | Rebuttal targeting evidence |
| `REFINES` | 0 | Clarification or specification |

**Propagation:** Log-odds Bayesian belief updating across the graph with configurable decay and damping.

### Hybrid Retrieval System

```
┌─────────────────────────────────────────────────────────────┐
│                    Hybrid Retriever                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ BM25 Sparse │    │ FAISS Dense │    │ Cross-Encoder│     │
│  │  Retrieval  │ -> │  Retrieval  │ -> │  Reranking   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│        │                   │                  │              │
│        v                   v                  v              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Reciprocal Rank Fusion (RRF)                │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Components:**
- **BM25 Sparse Retrieval**: Traditional keyword-based retrieval with TF-IDF scoring
- **FAISS Dense Retrieval**: Semantic vector search using sentence-transformers
- **Fusion Methods**: Weighted combination or Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Reranking**: Neural reranking for precision (optional)

### Decision-Theoretic Planning

**Expected Information Gain (EIG):**
```python
# Estimate value of an experiment
planner = VoIPlanner(llm=llm, n_samples=1000)
ranked_actions = planner.rank_by_eig(experiments, current_belief)

# Select optimal action set under budget constraint
optimal_set = planner.select_under_budget(experiments, budget=100)
```

**Calibration:**
- Brier Score assessment
- Expected Calibration Error (ECE)
- Temperature scaling for confidence adjustment
- Histogram binning for reliability diagrams

### Provenance & Governance

**Event Types:**
| Event | Description |
|-------|-------------|
| `SESSION_START` | Debate session initialization |
| `PROPOSITION_ADDED` | New proposition registered |
| `EVIDENCE_ADDED` | Evidence attached to proposition |
| `REBUTTAL_ADDED` | Rebuttal targeting evidence |
| `VERDICT_RENDERED` | Jury verdict recorded |
| `SESSION_END` | Session completion |

**Integrity Features:**
- SHA-256 hash chain for tamper detection
- PROV-O compatible event model
- Cryptographic attestations for content
- Query API for filtering and analysis

---

## Installation

### From PyPI (Recommended)

```bash
pip install argus-debate-ai
```

### From Source (Development)

```bash
git clone https://github.com/argus-ai/argus.git
cd argus
pip install -e ".[dev]"
```

### Optional Dependencies

```bash
# All features including development tools
pip install argus-debate-ai[all]

# Individual extras
pip install argus-debate-ai[ollama]   # Ollama local LLM support
pip install argus-debate-ai[cohere]   # Cohere integration
pip install argus-debate-ai[mistral]  # Mistral integration
pip install argus-debate-ai[groq]     # Groq LPU inference
pip install argus-debate-ai[arxiv]    # arXiv connector
```

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.11+ | 3.12+ |
| RAM | 4 GB | 16 GB |
| Storage | 1 GB | 10 GB (with embeddings) |
| GPU | None | CUDA-compatible (for local embeddings) |

---

## Quick Start

### Basic Usage

```python
from argus import RDCOrchestrator, get_llm

# Initialize with any supported LLM
llm = get_llm("openai", model="gpt-4o")

# Run a debate on a proposition
orchestrator = RDCOrchestrator(llm=llm, max_rounds=5)
result = orchestrator.debate(
    "The new treatment reduces symptoms by more than 20%",
    prior=0.5,  # Start with 50/50 uncertainty
)

print(f"Verdict: {result.verdict.label}")
print(f"Posterior: {result.verdict.posterior:.3f}")
print(f"Evidence: {result.num_evidence} items")
print(f"Reasoning: {result.verdict.reasoning}")
```

### Building a Debate Graph Manually

```python
from argus import CDAG, Proposition, Evidence, EdgeType
from argus.cdag.nodes import EvidenceType
from argus.cdag.propagation import compute_posterior

# Create the graph
graph = CDAG(name="drug_efficacy_debate")

# Add the proposition to evaluate
prop = Proposition(
    text="Drug X is effective for treating condition Y",
    prior=0.5,
    domain="clinical",
)
graph.add_proposition(prop)

# Add supporting evidence
trial_evidence = Evidence(
    text="Phase 3 RCT showed 35% symptom reduction (n=500, p<0.001)",
    evidence_type=EvidenceType.EMPIRICAL,
    polarity=1,  # Supports
    confidence=0.9,
    relevance=0.95,
    quality=0.85,
)
graph.add_evidence(trial_evidence, prop.id, EdgeType.SUPPORTS)

# Add challenging evidence
side_effect = Evidence(
    text="15% of patients experienced adverse events",
    evidence_type=EvidenceType.EMPIRICAL,
    polarity=-1,  # Attacks
    confidence=0.8,
    relevance=0.7,
)
graph.add_evidence(side_effect, prop.id, EdgeType.ATTACKS)

# Add rebuttal to the challenge
rebuttal = Rebuttal(
    text="Adverse events were mild and resolved without intervention",
    target_id=side_effect.id,
    rebuttal_type="clarification",
    strength=0.7,
    confidence=0.85,
)
graph.add_rebuttal(rebuttal, side_effect.id)

# Compute Bayesian posterior
posterior = compute_posterior(graph, prop.id)
print(f"Posterior probability: {posterior:.3f}")
```

### Document Ingestion & Retrieval

```python
from argus import DocumentLoader, Chunker, EmbeddingGenerator
from argus.retrieval import HybridRetriever

# Load documents (supports PDF, TXT, HTML, Markdown, JSON)
loader = DocumentLoader()
doc = loader.load("research_paper.pdf")

# Chunk with overlap for context preservation
chunker = Chunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk(doc)

# Create hybrid retriever
retriever = HybridRetriever(
    embedding_model="all-MiniLM-L6-v2",
    lambda_param=0.7,  # Weight toward dense retrieval
    use_reranker=True,
)
retriever.index_chunks(chunks)

# Search with hybrid scoring
results = retriever.retrieve("treatment efficacy results", top_k=10)
for r in results:
    print(f"[{r.rank}] Score: {r.score:.3f} - {r.chunk.text[:100]}...")
```

### Multi-Agent Debate

```python
from argus import get_llm
from argus.agents import Moderator, Specialist, Refuter, Jury
from argus import CDAG, Proposition

# Initialize LLM (can use different models for different agents)
llm = get_llm("anthropic", model="claude-3-5-sonnet-20241022")

# Initialize agents
moderator = Moderator(llm)
specialist = Specialist(llm, domain="clinical")
refuter = Refuter(llm)
jury = Jury(llm)

# Create debate graph
graph = CDAG()
prop = Proposition(text="The intervention is cost-effective", prior=0.5)
graph.add_proposition(prop)

# Moderator creates agenda
agenda = moderator.create_agenda(graph, prop.id)

# Specialists gather evidence
evidence = specialist.gather_evidence(graph, prop.id)

# Refuter challenges evidence
rebuttals = refuter.generate_rebuttals(graph, prop.id)

# Jury renders verdict
verdict = jury.evaluate(graph, prop.id)
print(f"Verdict: {verdict.label} (posterior={verdict.posterior:.3f})")
print(f"Reasoning: {verdict.reasoning}")
```

---

## LLM Providers (27+)

ARGUS v1.4 supports **27+ LLM providers** through a unified interface. All providers implement the same `BaseLLM` interface for seamless interchangeability.

### Supported Providers

| Provider | Models | Features | API Key Env Variable |
|----------|--------|----------|---------------------|
| **OpenAI** | GPT-4o, GPT-4, o1 | Generate, Stream, Embed | `OPENAI_API_KEY` |
| **Anthropic** | Claude 3.5 Sonnet, Opus | Generate, Stream | `ANTHROPIC_API_KEY` |
| **Google** | Gemini 1.5 Pro/Flash | Generate, Stream, Embed | `GOOGLE_API_KEY` |
| **Ollama** | Llama 3.2, Mistral, Phi | Local deployment | N/A (local) |
| **Cohere** | Command R, R+ | Generate, Stream, Embed | `COHERE_API_KEY` |
| **Mistral** | Large, Small, Codestral | Generate, Stream, Embed | `MISTRAL_API_KEY` |
| **Groq** | Llama 3.1 70B (ultra-fast) | Generate, Stream | `GROQ_API_KEY` |
| **DeepSeek** | DeepSeek Chat, Coder | Generate, Stream | `DEEPSEEK_API_KEY` |
| **xAI** | Grok-beta | Generate, Stream | `XAI_API_KEY` |
| **Perplexity** | Sonar (search-grounded) | Generate, Stream | `PERPLEXITY_API_KEY` |
| **Together** | 100+ open models | Generate, Stream, Embed | `TOGETHER_API_KEY` |
| **Fireworks** | Fast inference | Generate, Stream | `FIREWORKS_API_KEY` |
| **NVIDIA** | NIM endpoints | Generate, Stream | `NVIDIA_API_KEY` |
| **Azure OpenAI** | GPT-4 on Azure | Generate, Stream, Embed | `AZURE_OPENAI_API_KEY` |
| **AWS Bedrock** | Claude, Llama on AWS | Generate, Stream | AWS credentials |
| **Vertex AI** | Gemini on GCP | Generate, Stream | GCP credentials |
| **+ 10 more** | See docs | Various | Various |


### Usage Examples

#### OpenAI
```python
from argus.core.llm import OpenAILLM

llm = OpenAILLM(model="gpt-4o")
response = llm.generate("Explain quantum computing")
print(response.content)
```

#### Anthropic
```python
from argus.core.llm import AnthropicLLM

llm = AnthropicLLM(model="claude-3-5-sonnet-20241022")
response = llm.generate(
    "Analyze this research methodology",
    system_prompt="You are a research methodology expert."
)
```

#### Google Gemini
```python
from argus.core.llm import GeminiLLM

llm = GeminiLLM(model="gemini-1.5-pro")
response = llm.generate("Summarize the key findings")

# Also supports embeddings
embeddings = llm.embed(["text to embed"])
```

#### Ollama (Local)
```python
from argus.core.llm import OllamaLLM

llm = OllamaLLM(model="llama3.1", host="http://localhost:11434")
response = llm.generate("What is the capital of France?")
```

#### Cohere
```python
from argus.core.llm import CohereLLM

llm = CohereLLM(model="command-r-plus")
response = llm.generate("Explain machine learning")

# Cohere embeddings with input types
embeddings = llm.embed(
    ["search query"],
    input_type="search_query"  # or "search_document"
)
```

#### Mistral
```python
from argus.core.llm import MistralLLM

llm = MistralLLM(model="mistral-large-latest")
response = llm.generate(
    "Write a Python function",
    temperature=0.3
)

# Streaming
for chunk in llm.stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

#### Groq (Ultra-Fast Inference)
```python
from argus.core.llm import GroqLLM

llm = GroqLLM(model="llama-3.1-70b-versatile")
response = llm.generate("Explain photosynthesis")

# Groq also supports audio transcription
transcript = llm.transcribe("audio.wav")
```

### Provider Registry

```python
from argus.core.llm import get_llm, list_providers, register_provider

# List available providers
print(list_providers())
# ['openai', 'anthropic', 'gemini', 'ollama', 'cohere', 'mistral', 'groq']

# Get LLM by provider name
llm = get_llm("groq", model="llama-3.1-70b-versatile")

# Register custom provider
class MyCustomLLM(BaseLLM):
    # ... implementation
    pass

register_provider("custom", MyCustomLLM)
```

---

## Embedding Models (v2.0.0)

ARGUS v1.4 includes 16 embedding providers for semantic search and RAG applications.

### Available Providers

| Type | Providers |
|------|-----------|
| **Local (Free)** | SentenceTransformers, FastEmbed, Ollama |
| **Cloud APIs** | OpenAI, Cohere, HuggingFace, Voyage, Mistral, Google, Azure, Together, NVIDIA, Jina, Nomic, Bedrock, Fireworks |

### Quick Examples

```python
from argus.embeddings import get_embedding, list_embedding_providers

# List all 16 providers
print(list_embedding_providers())

# Local embedding (free, no API key)
embedder = get_embedding("sentence_transformers", model="all-MiniLM-L6-v2")
vectors = embedder.embed_documents(["Hello world", "Machine learning"])
print(f"Dimension: {len(vectors[0])}")  # 384

# Query embedding for search
query_vec = embedder.embed_query("What is AI?")

# OpenAI embeddings
embedder = get_embedding("openai", model="text-embedding-3-small")
vectors = embedder.embed_documents(["Doc 1", "Doc 2"])

# Cohere embeddings
embedder = get_embedding("cohere", model="embed-english-v3.0")
query_vec = embedder.embed_query("search query")  # Uses search_query input type
```

---

## Tool Integrations (v2.0.0)

ARGUS v1.4 includes 19 pre-built tools for search, web access, productivity, and finance.

### Available Tools

| Category | Tools | API Key Required |
|----------|-------|------------------|
| **Search** | DuckDuckGo, Wikipedia, ArXiv, Tavily, Brave, Exa | Some optional |
| **Web** | HTTP Requests, Web Scraper, Jina Reader, YouTube | Some optional |
| **Productivity** | FileSystem, Python REPL, Shell, GitHub, JSON | GitHub optional |
| **Database** | SQL, Pandas DataFrame | None |
| **Finance** | Yahoo Finance, Weather | Weather optional |

### Quick Examples

```python
from argus.tools.integrations import (
    DuckDuckGoTool, WikipediaTool, ArxivTool,
    PythonReplTool, YahooFinanceTool
)

# Free web search
search = DuckDuckGoTool()
result = search(query="latest AI research 2024", max_results=5)
for r in result.data["results"]:
    print(f"- {r['title']}: {r['url']}")

# Wikipedia lookup
wiki = WikipediaTool()
result = wiki(query="Machine Learning", action="summary", sentences=3)
print(result.data["summary"])

# ArXiv paper search
arxiv = ArxivTool()
result = arxiv(query="transformer attention", max_results=5)
for paper in result.data["results"]:
    print(f"📄 {paper['title']}")

# Execute Python code
repl = PythonReplTool()
result = repl(code="print(sum([1,2,3,4,5]))")
print(result.data["output"])  # 15

# Stock quotes
finance = YahooFinanceTool()
result = finance(symbol="AAPL", action="quote")
print(f"Apple: ${result.data['price']}")
```

---

## External Connectors

ARGUS provides connectors for fetching data from external sources. All connectors implement the `BaseConnector` interface.


### Web Connector (with robots.txt compliance)

Fetch web content while respecting robots.txt rules:

```python
from argus.knowledge.connectors import WebConnector, WebConnectorConfig

config = WebConnectorConfig(
    respect_robots_txt=True,  # Check robots.txt before fetching
    user_agent="ARGUS-Bot/1.0",
    timeout=30,
    max_content_length=10_000_000,  # 10MB
    robots_cache_ttl=3600,  # Cache robots.txt for 1 hour
)

connector = WebConnector(config=config)
result = connector.fetch("https://example.com/article")

if result.success:
    doc = result.documents[0]
    print(f"Title: {doc.title}")
    print(f"Content: {doc.content[:500]}...")
else:
    print(f"Error: {result.error}")
```

**Features:**
- Full robots.txt parsing and compliance
- Crawl-delay support
- Sitemap extraction
- Automatic content type detection
- Link extraction (optional)
- Beautiful Soup HTML parsing

### arXiv Connector

Fetch academic papers from arXiv:

```python
from argus.knowledge.connectors import ArxivConnector, ArxivConnectorConfig

config = ArxivConnectorConfig(
    sort_by="submittedDate",  # relevance, lastUpdatedDate, submittedDate
    sort_order="descending",
    include_abstract=True,
)

connector = ArxivConnector(config=config)

# Search by query
result = connector.fetch(
    "machine learning transformers",
    max_results=10,
    categories=["cs.AI", "cs.LG"],
)

for doc in result.documents:
    print(f"Title: {doc.title}")
    print(f"Authors: {doc.metadata['authors']}")
    print(f"arXiv ID: {doc.metadata['arxiv_id']}")
    print(f"PDF: {doc.metadata['pdf_url']}")
    print("---")

# Fetch specific paper by ID
result = connector.fetch_by_id("2103.14030")

# Fetch by category
result = connector.fetch_by_category(
    categories=["cs.AI", "cs.CL"],
    max_results=20,
)
```

**Query Syntax:**
- Full-text: `"machine learning"`
- Author: `au:Einstein`
- Title: `ti:quantum computing`
- Abstract: `abs:neural network`
- Category: `cat:cs.AI`
- Combined: `au:LeCun AND cat:cs.LG`

### CrossRef Connector

Fetch citation metadata from CrossRef:

```python
from argus.knowledge.connectors import CrossRefConnector, CrossRefConnectorConfig

config = CrossRefConnectorConfig(
    mailto="your@email.com",  # For polite pool (faster rate limits)
    sort="score",  # score, relevance, published, updated
    order="desc",
)

connector = CrossRefConnector(config=config)

# Lookup by DOI
result = connector.fetch_by_doi("10.1038/nature12373")
if result.success:
    doc = result.documents[0]
    print(f"Title: {doc.title}")
    print(f"Authors: {doc.metadata['author_names']}")
    print(f"Journal: {doc.metadata['container_title']}")
    print(f"Cited by: {doc.metadata['cited_by_count']}")

# Search by bibliographic query
result = connector.fetch(
    "attention is all you need transformers",
    max_results=5,
)

# Fetch references for a paper
result = connector.fetch_references("10.1038/nature12373")

# Find papers citing a DOI
result = connector.fetch_citing_works("10.1038/nature12373")
```

### Connector Registry

```python
from argus.knowledge.connectors import (
    ConnectorRegistry,
    get_default_registry,
    register_connector,
)

# Get default registry
registry = get_default_registry()

# Register connectors
from argus.knowledge.connectors import WebConnector, ArxivConnector

registry.register(WebConnector())
registry.register(ArxivConnector())

# Fetch from all registered connectors
results = registry.fetch_from_all(
    "machine learning",
    max_results_per_connector=5,
)

for name, result in results.items():
    print(f"{name}: {len(result.documents)} documents")

# Custom connector
from argus.knowledge.connectors import BaseConnector, ConnectorResult

class MyAPIConnector(BaseConnector):
    name = "my_api"
    description = "Custom API connector"
    
    def fetch(self, query: str, max_results: int = 10, **kwargs):
        # Your implementation here
        return ConnectorResult(success=True, documents=[...])

register_connector(MyAPIConnector())
```

---

## Visualization & Plotting

ARGUS provides publication-quality visualization capabilities for debate results, including static plots for research papers and interactive dashboards for exploration.

### Installation

```bash
# Core plotting dependencies (matplotlib, seaborn)
pip install argus-debate-ai[plotting]

# Interactive plots (adds Plotly)
pip install argus-debate-ai[interactive]

# Or install all visualization dependencies
pip install matplotlib seaborn plotly networkx
```

### Quick Start

```python
from argus.outputs import DebatePlotter, PlotConfig

# Configure plot settings
config = PlotConfig(
    output_dir="./plots",
    dpi=300,                # Publication quality
    format="png",           # png, pdf, svg
    theme="publication",    # publication, dark, light, minimal
)

# Generate all plots for a debate result
plotter = DebatePlotter(config)
paths = plotter.generate_all_plots(debate_result)
print(f"Generated {len(paths)} plots")
```

### Available Plot Types

#### Static Plots (Matplotlib/Seaborn)

| Plot Type | Method | Description |
|-----------|--------|-------------|
| **Posterior Evolution** | `plot_posterior_evolution()` | Line chart showing probability changes across rounds |
| **Evidence Distribution** | `plot_evidence_distribution()` | Donut and bar charts of support vs attack evidence |
| **Specialist Contributions** | `plot_specialist_contributions()` | Stacked bar chart by specialist and polarity |
| **Confidence Distribution** | `plot_confidence_distribution()` | Histogram, KDE, and box plot of evidence confidence |
| **Round Heatmap** | `plot_round_heatmap()` | Evidence count matrix by specialist and round |
| **CDAG Network** | `plot_cdag_network()` | NetworkX graph visualization with color-coded nodes |
| **Multi-Stock Comparison** | `plot_multi_stock_comparison()` | 4-panel dashboard comparing multiple debates |
| **Summary Radar** | `plot_summary_radar()` | Radar chart for multi-metric comparison |

#### Interactive Plots (Plotly)

| Plot Type | Method | Description |
|-----------|--------|-------------|
| **Interactive Posterior** | `plot_interactive_posterior()` | Zoomable, hoverable timeline chart |
| **Interactive Network** | `plot_interactive_network()` | Force-directed graph with tooltips |
| **Combined Dashboard** | `plot_dashboard()` | Multi-plot HTML dashboard |

### Usage Examples

#### Posterior Evolution Plot

```python
from argus.outputs import DebatePlotter, PlotConfig

plotter = DebatePlotter(PlotConfig(output_dir="./plots"))
path = plotter.plot_posterior_evolution(debate_result)
print(f"Saved to: {path}")
```

#### CDAG Network Visualization

```python
# Visualize the conceptual debate graph
path = plotter.plot_cdag_network(debate_result)
# Nodes colored by type: Proposition (blue), Evidence Support (green),
# Evidence Attack (red), Rebuttal (orange)
```

#### Multi-Stock Comparison Dashboard

```python
# Compare multiple debate results
all_results = [aapl_result, msft_result, googl_result, tsla_result]
path = plotter.plot_multi_stock_comparison(all_results)
# Creates 4-panel dashboard: posteriors, evidence counts, 
# verdict distribution, duration comparison
```

#### Interactive Dashboard

```python
from argus.outputs import InteractivePlotter

interactive = InteractivePlotter(PlotConfig(output_dir="./plots"))
path = interactive.plot_dashboard(all_results)
# Open {path} in browser for interactive exploration
```

### Plot Configuration

```python
from argus.outputs import PlotConfig, PlotTheme

config = PlotConfig(
    output_dir="./plots",           # Output directory
    dpi=300,                         # Resolution (300 for print)
    format="png",                    # Export format
    theme=PlotTheme.PUBLICATION,     # Visual theme
    interactive=True,                # Enable interactive plots
    figsize=(12, 8),                 # Default figure size
    title_fontsize=16,               # Title font size
    label_fontsize=12,               # Axis label font size
)
```

### Themes

| Theme | Description |
|-------|-------------|
| `publication` | Professional style for academic papers (default) |
| `dark` | Dark background with light elements |
| `light` | Clean, minimal light theme |
| `minimal` | Reduced chrome, focus on data |

### Color Palettes

ARGUS uses colorblind-friendly palettes:

```python
from argus.outputs import COLORS, SPECIALIST_COLORS

# Main palette
COLORS = {
    "primary": "#2E86AB",      # Blue
    "secondary": "#A23B72",    # Magenta
    "success": "#F18F01",      # Orange
    "danger": "#C73E1D",       # Red
    "warning": "#FFE66D",      # Yellow
    "support": "#2E8B57",      # Green
    "attack": "#DC143C",       # Crimson
    "neutral": "#708090",      # Slate gray
}

# Specialist colors
SPECIALIST_COLORS = {
    "Bull Analyst": "#2E8B57",
    "Bear Analyst": "#DC143C",
    "Technical Analyst": "#4169E1",
    "SEC Filing Analyst": "#9932CC",
}
```

### Integration with SEC Debate Workflow

The plotting module is automatically integrated with the SEC enhanced debate workflow:

```python
# Run SEC debate with automatic plot generation
python -m testing.workflows.sec_enhanced_debate

# Generates:
# - Individual plots for each stock (posterior, evidence, network, etc.)
# - Comparison plots across all stocks
# - Interactive dashboard
# 
# All saved to: testing/results/plots/
```

### Export Formats

| Format | Use Case |
|--------|----------|
| `png` | Web, presentations (raster, 300 DPI default) |
| `pdf` | Academic papers, print (vector graphics) |
| `svg` | Web scalable graphics (vector) |
| `html` | Interactive plots (Plotly only) |

---

---

## Argus Terminal (TUI)

Argus includes a Bloomberg-style Terminal User Interface (TUI) for interactive debates and research.

### Features
- **Retro Aesthetics**: Choose between 1980s Amber (financial) and 1970s Green (CRT) themes.
- **Real-time Debate**: Watch agents debate, cite evidence, and reach verdicts live.
- **System Monitoring**: Track token usage, costs, and agent states.
- **Interactive Tools**: Browser-like tool execution within the terminal.

### Quick Start
Run the terminal directly from your command line:
```bash
argus-terminal
```

### Controls
- **1-8**: Switch screens (Dashboard, Debate, Providers, Tools, etc.)
- **Tab/Enter**: Navigate and select
- **q**: Quit

---

## Argus-Viz (Streamlit Sandbox)

ARGUS v2.5 includes **Argus-Viz**, an interactive Streamlit web application for experimenting with and visualizing AI debates in real-time.

### Features

| Feature | Description |
|---------|-------------|
| **Live Debate Arena** | Run debates with real-time streaming — watch posterior probability and debate flow graph update incrementally each round |
| **10 Interactive Charts** | Posterior evolution, evidence waterfall, CDAG network, specialist radar, confidence histogram, debate timeline, polarity donut, round heatmap, and full lifecycle DAG |
| **Debate Flow Explainer** | Sankey pipeline diagram, step-by-step explanations, Bayesian algorithm visualization with LaTeX formulas |
| **Configurable Sidebar** | Pick LLM provider/model, set API key, adjust rounds, prior, jury threshold, toggle refuter, customize specialists |
| **Raw Data Export** | Download full debate results as JSON |

### Quick Start

```bash
# Install viz dependencies
pip install argus-debate-ai[viz]

# Launch (any of these work)
argus-viz
python -m argus_viz
streamlit run argus_viz/app.py
```

### Tabs

| Tab | What It Shows |
|-----|---------------|
| **⚔️ Debate Arena** | Live posterior chart + debate flow DAG updating each round, round logs, verdict card, evidence cards |
| **📊 Analysis Dashboard** | All 10 Plotly charts rendered in a grid layout |
| **🗺️ Debate Flow** | ARGUS pipeline Sankey diagram, step explanations, Bayesian formula, data overlay |
| **📋 Raw Data** | JSON result viewer, graph summary, download button |

### Live Visualization

During a debate, two charts update side-by-side in real-time:
- **Left**: Posterior probability evolution (line chart with confidence band)
- **Right**: Debate flow DAG — nodes and edges grow each round (Proposition → Specialists → Evidence → Rebuttals → Bayesian Updates → Verdict)

---

## CRUX Protocol

**Claim-Routed Uncertainty eXchange (CRUX)** is a novel inter-agent communication protocol that extends ARGUS with first-class epistemic state management. CRUX treats beliefs, uncertainty distributions, argument lineage, and credibility as core primitives of agent communication.

### Overview

Traditional multi-agent systems pass messages without explicit epistemic context. CRUX addresses this by:

- **Explicit Uncertainty**: Every claim carries a Beta distribution over confidence
- **Credibility Tracking**: Agents build statistical trust records based on prediction accuracy
- **Adversarial Routing**: Claims are routed to agents most likely to challenge them
- **Belief Reconciliation**: Contradicting claims are merged using Bayesian inference
- **Offline Support**: Agents can disconnect and reconnect without losing epistemic state

### Installation

```python
# CRUX is included with argus-debate-ai
from argus.crux import (
    CRUXOrchestrator,
    ClaimBundle,
    CredibilityLedger,
    EpistemicAgentCard,
)
```

### Seven Core Primitives

| Primitive | Module | Description |
|-----------|--------|-------------|
| **Epistemic Agent Card (EAC)** | `agent_card.py` | Agent identity with calibration metadata, domain expertise, and capability flags |
| **Claim Bundle (CB)** | `claim_bundle.py` | Atomic epistemic unit with uncertainty distribution (Beta), lineage, and supporting evidence |
| **Dialectical Routing (DR)** | `routing.py` | Adversarial-aware agent selection using Dialectical Fitness Scores (DFS) |
| **Belief Reconciliation Protocol (BRP)** | `brp.py` | Merging contradicting claims via Bayesian inference with proof certificates |
| **Credibility Ledger (CL)** | `ledger.py` | Hash-chained statistical trust layer with ELO-style updates |
| **Epistemic Dead Reckoning (EDR)** | `edr.py` | Reconnection sync protocol for offline agents |
| **Challenger Auction (CA)** | `auction.py` | Best challenger selection via competitive bidding |

### Quick Start

```python
from argus import RDCOrchestrator, get_llm
from argus.crux import (
    CRUXOrchestrator,
    CRUXConfig,
    ClaimBundle,
    BetaDistribution,
)

# Create base ARGUS orchestrator
llm = get_llm("openai", model="gpt-4o")
base = RDCOrchestrator(llm=llm, max_rounds=5)

# Wrap with CRUX for enhanced epistemic tracking
config = CRUXConfig(
    contradiction_threshold=0.20,
    enable_edr=True,
    enable_auction=True,
)
crux = CRUXOrchestrator(base=base, config=config)

# Run a CRUX-enabled debate
result = crux.debate(
    "Treatment X reduces symptoms by more than 20%",
    prior=0.5,
)

# Access CRUX-specific results
print(f"Verdict: {result.verdict.label}")
print(f"Reconciled Posterior: {result.reconciled_cb.posterior:.3f}")
print(f"Final Credibility Scores: {result.credibility_snapshot}")
```

### Epistemic Agent Card (EAC)

Every agent in CRUX declares its epistemic capabilities through an Agent Card:

```python
from argus.crux import EpistemicAgentCard, AgentCalibration, AgentCapabilities

card = EpistemicAgentCard(
    agent_id="specialist-clinical-001",
    agent_type="specialist",
    display_name="Clinical Trial Specialist",
    calibration=AgentCalibration(
        brier_score=0.12,
        ece=0.08,
        n_predictions=500,
        last_updated="2024-01-15T10:00:00Z",
    ),
    capabilities=AgentCapabilities(
        domains=["clinical", "pharmacology", "epidemiology"],
        evidence_types=["empirical", "statistical"],
        can_refute=True,
        can_synthesize=True,
    ),
    llm_provider="anthropic",
    llm_model="claude-3-5-sonnet-20241022",
)

# Register with the orchestrator
crux.register_agent_card(card)
```

### Claim Bundle

Claim Bundles are the atomic unit of epistemic exchange:

```python
from argus.crux import ClaimBundle, ClaimBundleFactory, BetaDistribution

# Create a claim with confidence distribution
bundle = ClaimBundle(
    claim_id="claim-001",
    text="The intervention reduces mortality by 15%",
    source_agent="specialist-clinical-001",
    confidence_distribution=BetaDistribution(alpha=8.0, beta=2.0),
    lineage=["evidence-001", "evidence-002"],
    timestamp="2024-01-15T10:30:00Z",
)

# Access derived properties
print(f"Posterior: {bundle.posterior:.3f}")           # Mean of Beta: α/(α+β)
print(f"Uncertainty: {bundle.uncertainty:.3f}")       # Variance of Beta
print(f"95% CI: {bundle.credible_interval(0.95)}")    # Bayesian credible interval

# Factory for creating bundles from debate evidence
factory = ClaimBundleFactory()
bundle = factory.from_evidence(
    evidence=evidence_node,
    source_agent="specialist-001",
)
```

### Dialectical Fitness Score (DFS)

DFS determines which agent should handle a claim based on adversarial potential:

```python
from argus.crux import DialecticalRouter, compute_dfs

# Initialize router with agent cards
router = DialecticalRouter(
    registry=crux.agent_registry,
    ledger=crux.credibility_ledger,
)

# Compute DFS for all agents on a claim
scores = router.compute_all_dfs(claim_bundle)
for agent_id, score in scores.items():
    print(f"{agent_id}: DFS={score.total:.3f}")
    print(f"  Domain Match: {score.domain_match:.2f}")
    print(f"  Adversarial Potential: {score.adversarial_potential:.2f}")
    print(f"  Credibility: {score.credibility:.2f}")
    print(f"  Recency: {score.recency:.2f}")

# Route to best challenger
best_agent = router.select_best_challenger(claim_bundle)
print(f"Routed to: {best_agent}")
```

**DFS Formula:**
```
DFS(agent, claim) = w₁·domain_match + w₂·adversarial_potential + w₃·credibility + w₄·recency
```

### Belief Reconciliation Protocol (BRP)

When agents produce contradicting claims, BRP merges them:

```python
from argus.crux import BeliefReconciliationProtocol, BRPSession

brp = BeliefReconciliationProtocol(
    contradiction_threshold=0.20,  # Claims >20% apart are contradictions
)

# Detect contradictions
contradictions = brp.detect_contradictions([bundle1, bundle2, bundle3])

for contradiction in contradictions:
    print(f"Contradiction: {contradiction.bundle_a.claim_id} vs {contradiction.bundle_b.claim_id}")
    print(f"  Gap: {contradiction.gap:.2%}")
    
    # Reconcile using Bayesian merging
    result = brp.reconcile(contradiction)
    print(f"  Merged Posterior: {result.merged_bundle.posterior:.3f}")
    print(f"  Method: {result.method}")  # bayesian_merge, credibility_weighted, etc.
    print(f"  Proof: {result.proof_certificate}")
```

**Reconciliation Methods:**
- **Bayesian Merge**: Combine Beta distributions via parameter addition
- **Credibility-Weighted**: Weight by agent credibility scores
- **Evidence Quality**: Weight by underlying evidence quality metrics
- **Dominance**: Higher-credibility agent's claim dominates

### Credibility Ledger

The Credibility Ledger maintains a hash-chained record of agent performance:

```python
from argus.crux import CredibilityLedger, CredibilityUpdate

ledger = CredibilityLedger()

# Record a prediction outcome
ledger.record_update(
    agent_id="specialist-001",
    update=CredibilityUpdate(
        claim_id="claim-001",
        predicted_probability=0.75,
        actual_outcome=True,  # Claim was verified
        timestamp="2024-01-15T12:00:00Z",
    )
)

# Get current credibility
cred = ledger.get_credibility("specialist-001")
print(f"Credibility: {cred.score:.3f}")
print(f"Brier Score: {cred.brier_score:.3f}")
print(f"N Predictions: {cred.n_predictions}")

# Verify ledger integrity
assert ledger.verify_chain(), "Ledger tampered!"

# Get full history for visualization
history = ledger.get_credibility_history("specialist-001")
```

**Hash Chain:**
```
entry_hash = SHA256(prev_hash || agent_id || update_data || timestamp)
```

### Epistemic Dead Reckoning (EDR)

EDR enables agents to disconnect and reconnect without losing state:

```python
from argus.crux import EpistemicDeadReckoning, EDRSynchronizer

edr = EpistemicDeadReckoning(session=crux_session)

# Checkpoint before agent disconnects
checkpoint = edr.create_checkpoint("specialist-001")
print(f"Checkpoint ID: {checkpoint.checkpoint_id}")
print(f"Belief State: {len(checkpoint.belief_state)} claims")

# ... agent is offline ...

# Sync when agent reconnects
sync_result = edr.synchronize(
    agent_id="specialist-001",
    checkpoint_id=checkpoint.checkpoint_id,
)

print(f"Deltas Applied: {len(sync_result.deltas)}")
print(f"New Claims: {sync_result.new_claims}")
print(f"Updated Claims: {sync_result.updated_claims}")
print(f"Conflicts Resolved: {sync_result.conflicts_resolved}")
```

### Challenger Auction

For high-stakes claims, CRUX runs an auction to select the best challenger:

```python
from argus.crux import ChallengerAuction, ChallengerBid

auction = ChallengerAuction(
    claim=claim_bundle,
    timeout_seconds=30,
)

# Agents submit bids
auction.submit_bid(ChallengerBid(
    agent_id="refuter-001",
    confidence=0.85,
    evidence_preview=["Counter-evidence from meta-analysis..."],
    stake=0.10,  # Credibility stake
))

auction.submit_bid(ChallengerBid(
    agent_id="refuter-002",
    confidence=0.72,
    evidence_preview=["Methodological concerns..."],
    stake=0.08,
))

# Close auction and select winner
result = auction.close()
print(f"Winner: {result.winner_agent_id}")
print(f"Winning Bid DFS: {result.winning_dfs:.3f}")
print(f"All Bids Evaluated: {len(result.all_bids)}")
```

### Visualization

CRUX includes comprehensive visualization for debates:

```python
from argus.crux import (
    plot_crux_debate_flow,
    plot_credibility_evolution,
    plot_brp_merge,
    plot_dfs_heatmap,
    plot_auction_results,
    create_crux_dashboard,
    export_debate_static,
)

# Interactive debate flow (Plotly)
fig = plot_crux_debate_flow(crux_result)
fig.show()

# Credibility evolution over time
fig = plot_credibility_evolution(crux_result)
fig.write_html("credibility.html")

# BRP merge visualization
fig = plot_brp_merge(reconciliation_result)
fig.show()

# DFS heatmap for routing decisions
fig = plot_dfs_heatmap(routing_history)
fig.write_image("dfs_heatmap.png")

# Auction results
fig = plot_auction_results(auction_result)
fig.show()

# Complete dashboard
fig = create_crux_dashboard(crux_result)
fig.write_html("crux_dashboard.html")

# Static export for papers
export_debate_static(
    crux_result,
    output_dir="./figures",
    format="pdf",  # pdf, png, svg
    dpi=300,
)
```

### Module Structure

```
argus/crux/
├── __init__.py          # Public exports
├── models.py            # Core data structures (BetaDistribution, etc.)
├── agent_card.py        # Epistemic Agent Card
├── claim_bundle.py      # Claim Bundle
├── routing.py           # Dialectical Routing & DFS
├── brp.py               # Belief Reconciliation Protocol
├── ledger.py            # Credibility Ledger (hash-chained)
├── edr.py               # Epistemic Dead Reckoning
├── auction.py           # Challenger Auction
├── orchestrator.py      # CRUXOrchestrator wrapper
└── visualization.py     # Plotting functions
```

### Integration with ARGUS

CRUX integrates seamlessly with existing ARGUS components:

```python
# CRUX extends the C-DAG with confidence distributions
from argus import CDAG
from argus.crux import ClaimBundleFactory

cdag = CDAG(name="crux_enabled_debate")
factory = ClaimBundleFactory()

# Convert Evidence nodes to Claim Bundles
for evidence in cdag.get_all_evidence():
    bundle = factory.from_evidence(evidence, source_agent="specialist-001")
    crux_session.add_claim(bundle)

# CRUX writes to PROV-O ledger
from argus.provenance import ProvenanceLedger

ledger = ProvenanceLedger()
crux = CRUXOrchestrator(base=orchestrator, provenance_ledger=ledger)

# All CRUX operations are recorded
result = crux.debate("proposition")
assert len(ledger.events) > 0
```

### Configuration

```python
from argus.crux import CRUXConfig

config = CRUXConfig(
    # BRP settings
    contradiction_threshold=0.20,      # Gap to trigger reconciliation
    reconciliation_method="bayesian",  # bayesian, credibility_weighted
    
    # DFS weights
    dfs_domain_weight=0.3,
    dfs_adversarial_weight=0.3,
    dfs_credibility_weight=0.25,
    dfs_recency_weight=0.15,
    
    # Features
    enable_edr=True,                   # Enable dead reckoning
    enable_auction=True,               # Enable challenger auction
    auction_timeout=30,                # Seconds
    
    # Credibility
    initial_credibility=0.5,
    credibility_update_rate=0.1,       # ELO-style K-factor
)
```

---

## Command Line Interface

ARGUS provides a full-featured CLI for common operations:


### Debate Commands

```bash
# Run a debate
argus debate "The hypothesis is supported by evidence" --prior 0.5 --rounds 3

# Quick single-call evaluation
argus evaluate "Climate change increases wildfire frequency"

# Debate with specific provider
argus debate "Query" --provider anthropic --model claude-3-5-sonnet-20241022

# Verbose output with provenance
argus debate "Claim to evaluate" --verbose --provenance
```

### Document Management

```bash
# Ingest documents into index
argus ingest ./documents --output ./index

# Ingest specific file types
argus ingest ./papers --extensions pdf,md,txt

# Show index statistics
argus index stats ./index

# Search the index
argus search "treatment efficacy" --index ./index --top-k 10
```

### Configuration

```bash
# Show current configuration
argus config

# Show specific value
argus config get default_provider

# Set value (saves to ~/.argus/config.yaml)
argus config set temperature 0.5

# Validate API keys
argus config validate
```

### Utility Commands

```bash
# List available providers
argus providers

# Check connection to provider
argus ping openai

# Version information
argus --version
```

---

## Configuration

### Environment Variables

```bash
# LLM API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
export COHERE_API_KEY="..."
export MISTRAL_API_KEY="..."
export GROQ_API_KEY="gsk_..."

# Default settings
export ARGUS_DEFAULT_PROVIDER="openai"
export ARGUS_DEFAULT_MODEL="gpt-4o"
export ARGUS_TEMPERATURE="0.7"
export ARGUS_MAX_TOKENS="4096"

# Ollama (local)
export ARGUS_OLLAMA_HOST="http://localhost:11434"

# Logging
export ARGUS_LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR
```

### Configuration File

Create `~/.argus/config.yaml`:

```yaml
# Default LLM settings
default_provider: openai
default_model: gpt-4o
temperature: 0.7
max_tokens: 4096

# LLM credentials (prefer env vars for sensitive data)
llm:
  openai_api_key: ${OPENAI_API_KEY}
  anthropic_api_key: ${ANTHROPIC_API_KEY}
  google_api_key: ${GOOGLE_API_KEY}
  ollama_host: http://localhost:11434

# Debate settings
debate:
  max_rounds: 5
  min_evidence: 3
  convergence_threshold: 0.01
  
# Retrieval settings  
retrieval:
  embedding_model: all-MiniLM-L6-v2
  lambda_param: 0.7
  use_reranker: true
  reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
  
# Chunking settings
chunking:
  chunk_size: 512
  chunk_overlap: 50
  strategy: recursive  # sentence, recursive, semantic
```

### Programmatic Configuration

```python
from argus import ArgusConfig, get_config

# Create custom config
config = ArgusConfig(
    default_provider="anthropic",
    default_model="claude-3-5-sonnet-20241022",
    temperature=0.5,
    max_tokens=4096,
)

# Or get global config (from env vars and config file)
config = get_config()

# Access nested config
print(config.chunking.chunk_size)
print(config.llm.openai_api_key)
```

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ARGUS Architecture                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        Orchestration Layer                           │    │
│  │  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐  │    │
│  │  │ Moderator │───▷│ Specialist│───▷│  Refuter  │───▷│   Jury    │  │    │
│  │  │ (Planner) │    │ (Evidence)│    │(Challenges)│    │ (Verdict) │  │    │
│  │  └─────┬─────┘    └─────┬─────┘    └─────┬─────┘    └─────┬─────┘  │    │
│  │        │                │                │                │         │    │
│  └────────┼────────────────┼────────────────┼────────────────┼─────────┘    │
│           │                │                │                │               │
│           ▼                ▼                ▼                ▼               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    C-DAG (Conceptual Debate Graph)                   │    │
│  │                                                                       │    │
│  │    ┌────────────┐      ┌────────────┐      ┌────────────┐           │    │
│  │    │Propositions│◀────▶│  Evidence  │◀────▶│  Rebuttals │           │    │
│  │    └────────────┘      └────────────┘      └────────────┘           │    │
│  │                   ▲                  │                                │    │
│  │                   └──────────────────┘                                │    │
│  │              Signed Influence Propagation                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         Decision Layer                               │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │    │
│  │  │  Bayesian   │  │     EIG     │  │ Calibration │                  │    │
│  │  │  Updating   │  │    (VoI)    │  │ (Brier/ECE) │                  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │ Knowledge Layer│  │ External Layer │  │Provenance Layer│                 │
│  │ ┌────────────┐ │  │ ┌────────────┐ │  │ ┌────────────┐ │                 │
│  │ │ Ingestion  │ │  │ │    Web     │ │  │ │PROV-O Ledger│ │                 │
│  │ │ Chunking   │ │  │ │   arXiv    │ │  │ │ Hash Chain │ │                 │
│  │ │ Embeddings │ │  │ │ CrossRef   │ │  │ │Attestations│ │                 │
│  │ │Hybrid Index│ │  │ │ (Custom)   │ │  │ │  Queries   │ │                 │
│  │ └────────────┘ │  │ └────────────┘ │  │ └────────────┘ │                 │
│  └────────────────┘  └────────────────┘  └────────────────┘                 │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         LLM Provider Layer                           │    │
│  │  ┌─────┐ ┌─────────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌───────┐ ┌────┐   │    │
│  │  │OpenAI│ │Anthropic│ │Gemini│ │Ollama│ │Cohere│ │Mistral│ │Groq│   │    │
│  │  └─────┘ └─────────┘ └──────┘ └──────┘ └──────┘ └───────┘ └────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Structure

| Module | Description | Key Classes |
|--------|-------------|-------------|
| `argus.core` | Configuration, data models, LLM abstractions | `ArgusConfig`, `Document`, `Chunk`, `BaseLLM` |
| `argus.cdag` | Conceptual Debate Graph implementation | `CDAG`, `Proposition`, `Evidence`, `Rebuttal` |
| `argus.decision` | Bayesian updating, EIG, VoI planning, calibration | `BayesianUpdater`, `VoIPlanner`, `Calibrator` |
| `argus.knowledge` | Document ingestion, chunking, embeddings, indexing | `DocumentLoader`, `Chunker`, `EmbeddingGenerator` |
| `argus.retrieval` | Hybrid retrieval, reranking | `HybridRetriever`, `BM25Retriever`, `DenseRetriever` |
| `argus.agents` | Debate agents | `Moderator`, `Specialist`, `Refuter`, `Jury` |
| `argus.provenance` | PROV-O ledger, integrity, attestations | `ProvenanceLedger`, `Event`, `Attestation` |
| `argus.orchestrator` | RDC orchestration engine | `RDCOrchestrator` |
| `argus.tools` | Extensible tool framework | `Tool`, `ToolExecutor`, `ToolRegistry` |
| `argus.outputs` | Report generation and visualization | `ReportGenerator`, `DebatePlotter`, `InteractivePlotter` |
| `argus.metrics` | Observability and tracing | `MetricsCollector`, `Tracer` |

---

## Core Components

### Document Model

```python
from argus.core.models import Document, SourceType

doc = Document(
    url="file:///path/to/doc.pdf",
    title="Research Paper Title",
    content="Full text content...",
    source_type=SourceType.PDF,
    metadata={
        "author": "Jane Doe",
        "date": "2024-01-15",
        "pages": 12,
    }
)

# Computed properties
print(doc.id)           # Auto-generated UUID
print(doc.content_hash) # SHA-256 hash
print(doc.word_count)   # Word count
```

### Chunk Model

```python
from argus.core.models import Chunk

chunk = Chunk(
    doc_id=doc.id,
    text="Chunk text content...",
    start_char=0,
    end_char=512,
    chunk_index=0,
    metadata={"section": "Abstract"}
)

# Properties
print(chunk.span)    # (0, 512)
print(chunk.length)  # 512
```

### Evidence Types

```python
from argus.cdag.nodes import EvidenceType

# Available types
EvidenceType.EMPIRICAL      # Experimental/observational data
EvidenceType.THEORETICAL    # Theoretical arguments
EvidenceType.STATISTICAL    # Statistical analysis
EvidenceType.CASE_STUDY     # Case study evidence
EvidenceType.EXPERT_OPINION # Expert testimony
EvidenceType.LITERATURE     # Literature review
EvidenceType.LOGICAL        # Logical argument
EvidenceType.METHODOLOGICAL # Methodological critique
EvidenceType.ECONOMIC       # Economic analysis
```

---

## Algorithms

### Signed Influence Propagation

The C-DAG uses log-odds space for numerically stable Bayesian belief propagation:

```
posterior = σ(log-odds(prior) + Σᵢ wᵢ · log(LRᵢ))
```

Where:
- `σ` is the logistic (sigmoid) function
- `LRᵢ` is the likelihood ratio for evidence i
- `wᵢ = polarityᵢ × confidenceᵢ × relevanceᵢ × qualityᵢ`

**Implementation:**
```python
def compute_posterior(graph, proposition_id):
    prop = graph.get_proposition(proposition_id)
    log_odds = logit(prop.prior)
    
    for evidence in graph.get_evidence_for(proposition_id):
        weight = evidence.polarity * evidence.confidence * evidence.relevance
        likelihood_ratio = compute_lr(evidence)
        log_odds += weight * log(likelihood_ratio)
    
    return sigmoid(log_odds)
```

### Expected Information Gain

For experiment planning, ARGUS computes EIG via Monte Carlo sampling:

```
EIG(a) = H(p) - 𝔼ᵧ[H(p|y)]
```

Where:
- `H(p)` is the entropy of current belief
- `𝔼ᵧ[H(p|y)]` is expected entropy after observing outcome y

**Implementation:**
```python
def compute_eig(action, current_belief, n_samples=1000):
    current_entropy = entropy(current_belief)
    
    expected_posterior_entropy = 0
    for _ in range(n_samples):
        outcome = simulate_outcome(action, current_belief)
        posterior = update_belief(current_belief, outcome)
        expected_posterior_entropy += entropy(posterior)
    
    expected_posterior_entropy /= n_samples
    return current_entropy - expected_posterior_entropy
```

### Calibration Methods

**Temperature Scaling:**
```
T* = argmin_T Σᵢ CrossEntropy(yᵢ, σ(zᵢ/T))
```

**Metrics:**
- **Brier Score**: Mean squared error of probability estimates
- **ECE**: Expected Calibration Error (binned reliability)
- **MCE**: Maximum Calibration Error

```python
from argus.decision import Calibrator

calibrator = Calibrator()
calibrator.fit(logits, labels)

calibrated_probs = calibrator.calibrate(new_logits)
brier_score = calibrator.brier_score(labels, probs)
ece = calibrator.expected_calibration_error(labels, probs)
```

---

## API Reference

### Core Classes

#### `RDCOrchestrator`

```python
class RDCOrchestrator:
    def __init__(
        self,
        llm: BaseLLM,
        max_rounds: int = 5,
        min_evidence: int = 3,
        convergence_threshold: float = 0.01,
        retriever: Optional[HybridRetriever] = None,
    ): ...
    
    def debate(
        self,
        proposition: str,
        prior: float = 0.5,
        domain: Optional[str] = None,
        documents: Optional[List[Document]] = None,
    ) -> DebateResult: ...
```

#### `CDAG`

```python
class CDAG:
    def __init__(self, name: str = ""): ...
    
    def add_proposition(self, prop: Proposition) -> str: ...
    def add_evidence(self, evidence: Evidence, target_id: str, edge_type: EdgeType) -> str: ...
    def add_rebuttal(self, rebuttal: Rebuttal, target_id: str) -> str: ...
    
    def get_proposition(self, prop_id: str) -> Optional[Proposition]: ...
    def get_evidence_for(self, prop_id: str) -> List[Evidence]: ...
    def get_rebuttals_for(self, evidence_id: str) -> List[Rebuttal]: ...
    
    def to_networkx(self) -> nx.DiGraph: ...
    def to_dict(self) -> Dict[str, Any]: ...
```

#### `BaseLLM`

```python
class BaseLLM(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str | List[Message],
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse: ...
    
    @abstractmethod
    def stream(
        self,
        prompt: str | List[Message],
        **kwargs,
    ) -> Iterator[str]: ...
    
    def embed(
        self,
        texts: str | List[str],
        **kwargs,
    ) -> List[List[float]]: ...
    
    def count_tokens(self, text: str) -> int: ...
```

#### `BaseConnector`

```python
class BaseConnector(ABC):
    name: str
    description: str
    
    @abstractmethod
    def fetch(
        self,
        query: str,
        max_results: int = 10,
        **kwargs,
    ) -> ConnectorResult: ...
    
    def test_connection(self) -> bool: ...
    def validate_config(self) -> Optional[str]: ...
```

---

## Examples

### Clinical Evidence Evaluation

```python
from argus import RDCOrchestrator, get_llm
from argus.retrieval import HybridRetriever
from argus.knowledge import DocumentLoader, Chunker

# Load clinical literature
loader = DocumentLoader()
documents = [loader.load(f) for f in clinical_papers]

# Create chunks
chunker = Chunker(chunk_size=512)
all_chunks = []
for doc in documents:
    all_chunks.extend(chunker.chunk(doc))

# Index for retrieval
retriever = HybridRetriever(use_reranker=True)
retriever.index_chunks(all_chunks)

# Evaluate treatment claim
orchestrator = RDCOrchestrator(
    llm=get_llm("openai", model="gpt-4o"),
    max_rounds=5,
)

result = orchestrator.debate(
    "Metformin reduces HbA1c by >1% in Type 2 diabetes",
    prior=0.6,  # Prior based on existing knowledge
    retriever=retriever,
    domain="clinical",
)

print(f"Verdict: {result.verdict.label}")
print(f"Posterior: {result.verdict.posterior:.3f}")
print(f"Confidence: {result.verdict.confidence:.3f}")
print(f"\nEvidence Summary:")
for e in result.evidence[:5]:
    print(f"  - [{e.polarity:+d}] {e.text[:80]}...")
```

### Research Claim Verification

```python
from argus import CDAG, Proposition, Evidence, EdgeType
from argus.cdag.nodes import EvidenceType
from argus.cdag.propagation import compute_all_posteriors
from argus.knowledge.connectors import ArxivConnector

# Fetch relevant papers
arxiv = ArxivConnector()
result = arxiv.fetch(
    "neural scaling laws emergent capabilities",
    max_results=20,
)

# Create debate graph
graph = CDAG(name="research_verification")

claim = Proposition(
    text="Neural scaling laws predict emergent capabilities",
    prior=0.5,
)
graph.add_proposition(claim)

# Add evidence from papers
for doc in result.documents:
    evidence = Evidence(
        text=f"{doc.title}: {doc.content[:200]}...",
        evidence_type=EvidenceType.LITERATURE,
        polarity=1 if "support" in doc.content.lower() else -1,
        confidence=0.7,
    )
    graph.add_evidence(
        evidence, 
        claim.id, 
        EdgeType.SUPPORTS if evidence.polarity > 0 else EdgeType.ATTACKS
    )

# Compute posteriors
posteriors = compute_all_posteriors(graph)
for prop_id, posterior in posteriors.items():
    prop = graph.get_proposition(prop_id)
    print(f"{prop.text[:50]}... : {posterior:.3f}")
```

### Custom Agent Pipeline

```python
from argus import get_llm, CDAG, Proposition
from argus.agents import Moderator, Specialist, Refuter, Jury
from argus.provenance import ProvenanceLedger, EventType

# Initialize with provenance tracking
ledger = ProvenanceLedger()
ledger.record(EventType.SESSION_START)

# Different models for different tasks
moderator_llm = get_llm("openai", model="gpt-4o")
specialist_llm = get_llm("anthropic", model="claude-3-5-sonnet-20241022")
refuter_llm = get_llm("groq", model="llama-3.1-70b-versatile")
jury_llm = get_llm("gemini", model="gemini-1.5-pro")

# Initialize agents
moderator = Moderator(moderator_llm)
specialist = Specialist(specialist_llm, domain="policy")
refuter = Refuter(refuter_llm)
jury = Jury(jury_llm)

# Create debate
graph = CDAG()
prop = Proposition(
    text="Carbon pricing is effective for reducing emissions",
    prior=0.5,
)
graph.add_proposition(prop)
ledger.record(EventType.PROPOSITION_ADDED, entity_id=prop.id)

# Run debate rounds
for round_num in range(3):
    # Gather evidence
    evidence = specialist.gather_evidence(graph, prop.id)
    for e in evidence:
        ledger.record(EventType.EVIDENCE_ADDED, entity_id=e.id)
    
    # Generate rebuttals
    rebuttals = refuter.generate_rebuttals(graph, prop.id)
    for r in rebuttals:
        ledger.record(EventType.REBUTTAL_ADDED, entity_id=r.id)
    
    # Check stopping criteria
    if moderator.should_stop(graph, prop.id):
        break

# Render verdict
verdict = jury.evaluate(graph, prop.id)
ledger.record(EventType.VERDICT_RENDERED, entity_id=prop.id)
ledger.record(EventType.SESSION_END)

print(f"Verdict: {verdict.label}")
print(f"Posterior: {verdict.posterior:.3f}")
print(f"Ledger entries: {len(ledger)}")

# Verify integrity
is_valid, errors = ledger.verify_integrity()
print(f"Integrity: {'Valid' if is_valid else 'Invalid'}")
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=argus --cov-report=html

# Run specific test modules
pytest tests/unit/test_cdag.py -v
pytest tests/unit/test_llm.py -v

# Run integration tests
pytest tests/integration/ -v

# Run only fast tests (skip slow/network tests)
pytest -m "not slow"

# Run with verbose output
pytest -v --tb=short
```

### Test Categories

| Category | Path | Description |
|----------|------|-------------|
| Unit | `tests/unit/` | Isolated component tests |
| Integration | `tests/integration/` | Multi-component tests |
| E2E | `tests/e2e/` | Full workflow tests |

### Writing Tests

```python
import pytest
from argus.cdag import CDAG, Proposition, Evidence

def test_posterior_increases_with_supporting_evidence(mock_llm):
    """Test that posterior increases with supporting evidence."""
    from argus.cdag.propagation import compute_posterior
    
    graph = CDAG()
    prop = Proposition(text="Test claim", prior=0.5)
    graph.add_proposition(prop)
    
    initial_posterior = compute_posterior(graph, prop.id)
    
    evidence = Evidence(
        text="Strong support",
        evidence_type=EvidenceType.EMPIRICAL,
        polarity=1,
        confidence=0.9,
    )
    graph.add_evidence(evidence, prop.id, EdgeType.SUPPORTS)
    
    final_posterior = compute_posterior(graph, prop.id)
    
    assert final_posterior > initial_posterior
```

---

## Deployment

### Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install .

COPY . .

EXPOSE 8000

CMD ["python", "-m", "argus.server"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  argus:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./data:/app/data
      
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

volumes:
  ollama_data:
```

### Cloud Deployment

**AWS Lambda:**
```python
# handler.py
from argus import RDCOrchestrator, get_llm

def handler(event, context):
    llm = get_llm("openai")
    orchestrator = RDCOrchestrator(llm=llm)
    
    result = orchestrator.debate(
        event["proposition"],
        prior=event.get("prior", 0.5),
    )
    
    return {
        "statusCode": 200,
        "body": {
            "verdict": result.verdict.label,
            "posterior": result.verdict.posterior,
        }
    }
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/argus-ai/argus.git
cd argus

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Style

- **Formatter**: Black (line length 88)
- **Linter**: Ruff
- **Type Checking**: mypy (strict mode)
- **Docstrings**: Google style

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run `pytest` and `mypy`
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## Evaluation Framework

ARGUS includes a comprehensive evaluation framework for benchmarking and testing the AI debate system.

### Datasets (10 domains, 1050+ samples each)

| Dataset | Domain | Description |
|---------|--------|-------------|
| `factual_claims` | General | Knowledge verification |
| `scientific_hypotheses` | Science | Research claims |
| `financial_analysis` | Finance | Market predictions |
| `medical_efficacy` | Medical | Treatment claims |
| `legal_reasoning` | Legal | Case analysis |
| `technical_comparison` | Tech | System comparisons |
| `policy_impact` | Policy | Economic analysis |
| `historical_interpretation` | History | Event analysis |
| `environmental_risk` | Environment | Climate claims |
| `adversarial_edge_cases` | Adversarial | Stress testing |

### Global Benchmark Support

| Benchmark | Task | Description |
|-----------|------|-------------|
| **FEVER** | Fact Verification | Wikipedia-based claim verification |
| **SNLI/MultiNLI** | NLI | Natural language inference |
| **TruthfulQA** | Truthfulness | Truthfulness evaluation |
| **BoolQ** | Yes/No QA | Boolean questions |
| **ARC** | Science QA | Grade-school science |

### Scoring Metrics

#### Novel ARGUS Metrics (Unique to ARGUS)

| Metric | Full Name | Description |
|--------|-----------|-------------|
| **ARCIS** | Argus Reasoning Coherence Index Score | Logical consistency across rounds |
| **EVID-Q** | Evidence Quality Quotient | relevance × confidence × source quality |
| **DIALEC** | Dialectical Depth Evaluation Coefficient | Attack/defense sophistication |
| **REBUT-F** | Rebuttal Effectiveness Factor | Rebuttal impact measurement |
| **CONV-S** | Convergence Stability Score | Posterior convergence quality |
| **PROV-I** | Provenance Integrity Index | Citation chain completeness |
| **CALIB-M** | Calibration Matrix Score | Confidence alignment |
| **EIG-U** | Expected Information Gain Utilization | Uncertainty reduction efficiency |

#### Standard Industry Metrics

| Metric | Category | Description |
|--------|----------|-------------|
| **Accuracy** | Classification | Proportion of correct predictions |
| **F1 / Macro F1** | Classification | Precision-recall balance |
| **Brier Score** | Calibration | Probability prediction accuracy |
| **ECE / MCE** | Calibration | Expected/Maximum calibration error |
| **Log Loss** | Information | Cross-entropy loss |
| **Dialectical Balance** | Argumentation | Support/attack balance |

### Quick Start

```python
from argus.evaluation import BenchmarkRunner, load_dataset
from argus.evaluation.datasets import load_global_benchmark

# Load FEVER benchmark
fever_df = load_global_benchmark("fever", max_samples=1000)

# Compute standard and novel scores
from argus.evaluation.scoring import compute_all_scores, compute_all_standard_metrics
novel_scores = compute_all_scores(debate_result)
standard_scores = compute_all_standard_metrics(predictions, ground_truths)
```

### CLI Usage

```bash
# Dry run (no LLM calls)
python -m argus.evaluation.runner.benchmark_runner --dry-run

# Full benchmark run
python -m argus.evaluation.runner.benchmark_runner \
    --datasets factual_claims scientific_hypotheses \
    --benchmarks debate_quality \
    --max-samples 10 \
    --num-rounds 1
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Inspired by debate-native reasoning approaches in AI safety research
- Built on excellent open-source libraries:
  - [Pydantic](https://pydantic.dev/) - Data validation
  - [NetworkX](https://networkx.org/) - Graph algorithms
  - [FAISS](https://github.com/facebookresearch/faiss) - Vector search
  - [Sentence-Transformers](https://sbert.net/) - Embeddings
  - [BeautifulSoup](https://beautiful-soup-4.readthedocs.io/) - HTML parsing
- LLM integrations powered by:
  - OpenAI, Anthropic, Google, Cohere, Mistral, Groq APIs
  - Ollama for local deployment

---

<!-- ## Support

- 📖 [Documentation](https://argus-ai.readthedocs.io/)
- 🐛 [Issue Tracker](https://github.com/argus-ai/argus/issues)
- 💬 [Discussions](https://github.com/argus-ai/argus/discussions)
- 📧 [Email](mailto:support@argus-ai.dev) -->

---

**[PyPI](https://pypi.org/project/argus-debate-ai/)** | **[GitHub](https://github.com/argus-ai/argus)** | 
