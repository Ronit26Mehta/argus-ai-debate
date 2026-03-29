<div align="center">

# ARGUS
### Agentic Research & Governance Unified System

**Stop trusting AI that can't show its work.**

ARGUS evaluates any claim through structured multi-agent debate — AI agents gather evidence, challenge each other, and reach calibrated verdicts with a full cryptographic audit trail. Every conclusion is traceable. No hallucination hiding.

*MiroFish tells you what the crowd thinks. ARGUS tells you what's actually true — and why.*

---

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/argus-debate-ai.svg)](https://pypi.org/project/argus-debate-ai/5.5.0/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue.svg)](https://mypy.readthedocs.io/)
[![Tools: 50+](https://img.shields.io/badge/tools-50+-green.svg)](https://github.com/Ronit26Mehta/argus-ai-debate#tool-integrations-50)
[![LLM Providers: 27+](https://img.shields.io/badge/LLM%20providers-27+-purple.svg)](https://github.com/Ronit26Mehta/argus-ai-debate#llm-providers-27)
[![GitHub Stars](https://img.shields.io/github/stars/Ronit26Mehta/argus-ai-debate?style=social)](https://github.com/Ronit26Mehta/argus-ai-debate/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/Ronit26Mehta/argus-ai-debate)](https://github.com/Ronit26Mehta/argus-ai-debate/issues)

</div>

---

## What does ARGUS do?

You give ARGUS a claim. It runs a structured debate between specialist AI agents — one gathers supporting evidence, another finds counter-evidence, a refuter challenges both, and a Bayesian jury renders a calibrated verdict. Every step is hash-chain verified and exportable.

```python
from argus import RDCOrchestrator, get_llm

llm = get_llm("openai", model="gpt-4o")
result = RDCOrchestrator(llm=llm, max_rounds=5).debate(
    "Intermittent fasting improves cognitive performance",
    prior=0.5,
)

print(result.verdict.label)      # → "SUPPORTED"
print(result.verdict.posterior)  # → 0.731
print(result.verdict.reasoning)  # → Full Bayesian trace
```

**That's it. One call. You get a verdict, a posterior probability, and a complete evidence audit trail.**

---

## See it in action

### 1. ARGUS CDAG Live Debate

Watch the Conceptual Debate Graph build in real time — nodes and edges grow as agents gather evidence, generate rebuttals, and the posterior probability converges toward a verdict.

https://github.com/user-attachments/assets/e6155c68-988a-44c6-a56e-69f3dc4d5c44

---

### 2. ARISTOTLE — One Question, Full Autonomous Debate

Ask a natural-language question. ARISTOTLE's five-layer meta-orchestrator classifies intent, selects specialist agents, drives multi-round debate, and synthesises a plain-language verdict with a dissent log — all in a WhatsApp-style chat interface.

https://github.com/user-attachments/assets/699cb772-2d38-413b-94f5-6f3abfe4ac44

---

### 3. AGORA — Multi-Agent Debate Sandbox

The full Streamlit sandbox for experimenting with ARGUS debates in real time: live posterior evolution chart, C-DAG network, confidence histograms, Bayesian formula visualiser, and raw JSON export.

https://github.com/user-attachments/assets/2e98605c-8092-4aba-94ac-5b9d54d2f484

---

## Get started in 60 seconds

```bash
pip install argus-debate-ai
```

```python
from argus import RDCOrchestrator, get_llm

orchestrator = RDCOrchestrator(llm=get_llm("openai", model="gpt-4o"))
result = orchestrator.debate("Climate change increases wildfire frequency")

print(f"Verdict  : {result.verdict.label}")
print(f"Posterior: {result.verdict.posterior:.3f}")
print(f"Evidence : {result.num_evidence} items collected")
```

Works with any of 27+ LLM providers — OpenAI, Anthropic, Gemini, Groq, Ollama (local), and more.

---

## ARGUS vs alternatives

| Capability | **ARGUS** | MiroFish | AutoGen | LangGraph | LlamaIndex |
|---|---|---|---|---|---|
| Structured adversarial debate | ✅ | ❌ | Partial | ❌ | ❌ |
| Calibrated Bayesian posteriors | ✅ | ❌ | ❌ | ❌ | ❌ |
| Brier Score / ECE calibration | ✅ | ❌ | ❌ | ❌ | ❌ |
| Hash-chain provenance (PROV-O) | ✅ | ❌ | ❌ | ❌ | ❌ |
| CRUX epistemic protocol | ✅ | ❌ | ❌ | ❌ | ❌ |
| Hybrid BM25 + FAISS retrieval | ✅ | ❌ | Partial | Partial | ✅ |
| Temporal evidence decay (CHRONOS) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Population-scale simulation (PHALANX) | ✅ | ✅ | ❌ | ❌ | ❌ |
| Works with local models (Ollama) | ✅ | Partial | ✅ | ✅ | ✅ |
| PyPI installable | ✅ | ❌ | ✅ | ✅ | ✅ |
| Academic-grade audit trails | ✅ | ❌ | ❌ | ❌ | ❌ |

**ARGUS is the only framework built for the question "what is actually true?"** rather than "what will people believe?" or "what task can agents complete?"

---

## What can you build with ARGUS?

| Use case | How ARGUS helps |
|---|---|
| **Clinical evidence evaluation** | Evaluate treatment claims against medical literature with calibrated confidence |
| **Financial claim verification** | Debate earnings forecasts with specialist agents using SEC filings and news |
| **Research paper fact-checking** | Cross-check scientific claims against arXiv, CrossRef, and PubMed |
| **Policy impact analysis** | Structured debate on policy effectiveness with counterfactual consequence graphs (MIRROR) |
| **Legal argument assessment** | Evidence-weighted claim evaluation with full provenance for audit |
| **News fact-checking pipelines** | Document-to-debate pipeline (SEED) for automated claim extraction and verification |
| **LLM output validation** | Use ARGUS as a post-hoc verifier on top of any LLM's outputs |
| **Market intelligence** | Population-scale epistemic simulation (PHALANX) for consensus and polarisation analysis |

---

## Table of Contents

- [Overview](#overview)
- [Key Innovations](#key-innovations)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [LLM Providers (27+)](#llm-providers-27)
- [Tool Integrations (50+)](#tool-integrations-50)
- [OpenAPI REST Integration](#openapi-rest-integration)
- [Context Caching](#context-caching)
- [Context Compression](#context-compression)
- [Debate Visualization](#debate-visualization)
- [External Connectors](#external-connectors)
- [Visualization & Plotting](#visualization--plotting)
- [Argus Terminal (TUI)](#argus-terminal-tui)
- [Argus-Viz (Streamlit Sandbox)](#argus-viz-streamlit-sandbox)
- [CRUX-Viz (CRUX Protocol Sandbox)](#crux-viz-crux-protocol-sandbox)
- [ARISTOTLE Chat Interface](#aristotle-chat-interface)
- [CRUX Protocol](#crux-protocol)
- [Command Line Interface](#command-line-interface)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Core Components](#core-components)
- [Algorithms](#algorithms)
- [ARGUS Evolution Extensions (v5.0)](#argus-evolution-extensions-v50)
  - [CHRONOS — Temporal Evidence Decay](#chronos--temporal-evidence-decay)
  - [PHALANX — Population-Scale Simulation](#phalanx--population-scale-epistemic-simulation)
  - [SEED — Document-to-Debate Pipeline](#seed--document-to-debate-pipeline)
  - [MNEME — Persistent Agent Memory](#mneme--persistent-agent-memory)
  - [FRACTAL — Hierarchical Decomposition](#fractal--hierarchical-proposition-decomposition)
  - [MIRROR — Consequence Inference Graph](#mirror--consequence-inference-graph)
  - [VERICHAIN — Cross-Debate Truth Network](#verichain--cross-debate-truth-network)
  - [PULSE — Operational Intelligence Dashboard](#pulse--operational-intelligence-dashboard)
- [Evaluation Framework](#evaluation-framework)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

ARGUS implements **Research Debate Chain (RDC)** — a novel approach to AI reasoning that structures knowledge evaluation as multi-agent debates. Instead of single-pass inference, ARGUS orchestrates specialist agents that gather evidence, generate rebuttals, and render verdicts through Bayesian aggregation.

### Why ARGUS?

Traditional LLM applications suffer from:
- **Hallucination**: Models generate plausible but incorrect information
- **Overconfidence**: No calibrated uncertainty estimates
- **Opacity**: Black-box reasoning with no audit trail
- **Single-Point Failure**: One model, one perspective

ARGUS addresses these through:
- **Adversarial Debate**: Multiple agents challenge claims with evidence
- **Bayesian Aggregation**: Calibrated confidence through probability theory
- **Full Provenance**: Every claim traced to its source with SHA-256 hash chains
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

### ARGUS Evolution v5.0 — Eight Novel Extensions

| Extension | Innovation |
|-----------|------------|
| **CHRONOS** | Temporal C-DAG with PELT-based belief drift detection and causal attribution |
| **PHALANX** | Population-scale epistemic simulation with 5 quantitative cognitive biases and Jensen-Shannon Polarisation Index |
| **SEED** | Document-to-debate pipeline with novel DebatabilityScore (BiPolarity × Novelty × EvidenceDensity) |
| **MNEME** | Persistent agent memory with Beta-distribution Bayesian competence and rolling Brier Score calibration |
| **FRACTAL** | Hierarchical proposition decomposition with relationship-aware aggregation (AND/OR/Weighted/Geometric) |
| **MIRROR** | Consequence inference graph with counterfactual sensitivity dP(consequence)/dP(verdict) |
| **VERICHAIN** | SHA-256 hash-chained cross-debate truth registry with tamper detection and precedent injection |
| **PULSE** | Always-on operational intelligence with z-score anomaly detection, failure taxonomy, and HTML dashboard |

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
git clone https://github.com/Ronit26Mehta/argus-ai-debate.git
cd argus-ai-debate
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

# Evolution v5.0 extension extras
pip install argus-debate-ai[evolution]         # All 8 Evolution extensions (scipy, plotly, networkx)
pip install argus-debate-ai[verichain-pg]      # VERICHAIN PostgreSQL backend (psycopg2-binary)
pip install argus-debate-ai[mneme-qdrant]      # MNEME Qdrant vector DB backend (qdrant-client)
pip install argus-debate-ai[seed-web]          # SEED URL ingestion (requests, beautifulsoup4)
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

ARGUS v5.5 supports **27+ LLM providers** through a unified interface. All providers implement the same `BaseLLM` interface for seamless interchangeability.

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

#### Ollama (Local — no API key needed)
```python
from argus.core.llm import OllamaLLM

llm = OllamaLLM(model="llama3.1", host="http://localhost:11434")
response = llm.generate("What is the capital of France?")
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

## Embedding Models (16+)

ARGUS v5.5 includes 16 embedding providers for semantic search and RAG applications.

### Available Providers

| Type | Providers |
|------|-----------|
| **Local (Free)** | SentenceTransformers, FastEmbed, Ollama |
| **Cloud APIs** | OpenAI, Cohere, HuggingFace, Voyage, Mistral, Google, Azure, Together, NVIDIA, Jina, Nomic, Bedrock, Fireworks |

### Quick Examples

```python
from argus.embeddings import get_embedding, list_embedding_providers

# Local embedding (free, no API key)
embedder = get_embedding("sentence_transformers", model="all-MiniLM-L6-v2")
vectors = embedder.embed_documents(["Hello world", "Machine learning"])
print(f"Dimension: {len(vectors[0])}")  # 384

# Query embedding for search
query_vec = embedder.embed_query("What is AI?")

# OpenAI embeddings
embedder = get_embedding("openai", model="text-embedding-3-small")
vectors = embedder.embed_documents(["Doc 1", "Doc 2"])
```

---

## Tool Integrations (50+)

ARGUS v5.5 includes **50+ pre-built tools** across 13 categories for comprehensive agent capabilities.

### Available Tools by Category

| Category | Tools | Description |
|----------|-------|-------------|
| **Search** | DuckDuckGo, Wikipedia, ArXiv, Tavily, Brave, Exa | Web and academic search |
| **Web** | Requests, WebScraper, JinaReader, YouTube | Web content access |
| **Productivity** | FileSystem, PythonREPL, Shell, GitHub, JSON | Core productivity |
| **Database** | SQL, Pandas | Data access and manipulation |
| **Finance** | YahooFinance, Weather | Financial and weather data |
| **AI Agents** | AgentMail, AgentOps, GoodMem, Freeplay | AI agent infrastructure |
| **Cloud** | BigQuery, PubSub, CloudTrace, VertexAI Search/RAG | Google Cloud services |
| **Vector DB** | Chroma, Pinecone, Qdrant, MongoDB | Vector databases |
| **Productivity (Extended)** | Asana, Jira, Confluence, Linear, Notion | Project management |
| **Communication** | Mailgun, Stripe, PayPal | Email and payments |
| **DevOps** | GitLab, Postman, Daytona, N8n | Development operations |
| **Media/AI** | ElevenLabs, Cartesia, HuggingFace | Media and AI platforms |
| **Observability** | Arize, Phoenix, Monocle, MLflow, W&B Weave | ML observability |

### Installation

```bash
# Core tools (search, web, productivity, database, finance)
pip install argus-debate-ai[tools]

# Extended tools (all 50+ integrations)
pip install argus-debate-ai[tools-extended]

# Or install all features
pip install argus-debate-ai[all]
```

### Quick Examples

```python
from argus.tools.integrations import (
    DuckDuckGoTool, WikipediaTool, ArxivTool,
    PythonReplTool, AsanaTool, NotionTool,
    BigQueryTool, VertexAISearchTool,
    PineconeTool, QdrantTool,
    MLflowTool, WandBWeaveTool,
)

# Free web search
search = DuckDuckGoTool()
result = search(query="latest AI research 2024", max_results=5)
for r in result.data["results"]:
    print(f"- {r['title']}: {r['url']}")

# ArXiv paper search
arxiv = ArxivTool()
result = arxiv(query="transformer attention", max_results=5)
for paper in result.data["results"]:
    print(f"📄 {paper['title']}")

# Execute Python code
repl = PythonReplTool()
result = repl(code="print(sum([1,2,3,4,5]))")
print(result.data["output"])  # 15

# BigQuery data analysis
bq = BigQueryTool()
result = bq(action="query", query="SELECT * FROM dataset.table LIMIT 10")

# MLflow experiment tracking
mlflow = MLflowTool()
result = mlflow(action="log_metric", run_id="run-123", key="accuracy", value=0.95)
```

### Tool Registry

```python
from argus.tools.integrations import (
    list_all_tools,
    list_tool_categories,
    get_tools_by_category,
    get_tool_count,
)

# List categories (13 categories)
print(list_tool_categories())
# ['search', 'web', 'productivity', 'database', 'finance', 'ai_agents',
#  'cloud', 'vectordb', 'productivity_extended', 'communication',
#  'devops', 'media_ai', 'observability']

print(f"Total tools: {get_tool_count()}")  # 50+
```

---

## OpenAPI REST Integration

ARGUS includes a powerful OpenAPI module for automatically generating tools from REST API specifications.

### Features

- **OpenAPI v2 (Swagger) and v3 support**
- **Automatic client generation** from specs
- **Tool code generation** for agent integrations
- **Full authentication support** (API Key, Bearer, Basic, OAuth2)
- **Type-safe parameter handling**

### Quick Start

```python
from argus.core.openapi import (
    load_openapi_spec,
    OpenAPIParser,
    OpenAPIClient,
    OpenAPIToolGenerator,
)

spec = load_openapi_spec("https://api.example.com/openapi.json")
parser = OpenAPIParser()
api_spec = parser.parse(spec)

print(f"API: {api_spec.title} v{api_spec.version}")
print(f"Endpoints: {len(api_spec.operations)}")
```

### Dynamic Client Generation

```python
from argus.core.openapi import create_client

client = create_client(
    spec_path="https://petstore.swagger.io/v2/swagger.json",
    api_key="your-api-key",
)

# Methods are generated automatically from the spec
pets = client.get_pets(limit=10)
new_pet = client.create_pet(name="Fluffy", status="available")
```

### CLI Usage

```bash
# List available endpoints
argus openapi ./api_spec.yaml --list-endpoints

# Validate a spec
argus openapi https://api.example.com/openapi.json --validate

# Generate tool code
argus openapi ./api_spec.yaml --output my_tool.py --class-name MyAPITool
```

---

## Context Caching

ARGUS includes a comprehensive caching system for optimizing context management, reducing API costs, and improving performance.

### Features

- **Multiple backends**: Memory (LRU), File (persistent), Redis (distributed)
- **Specialized caches**: Conversation, Embedding, LLM Response
- **TTL support**: Automatic expiration
- **Namespaces**: Isolated cache spaces
- **Statistics**: Hit rates, access patterns

```bash
pip install argus-debate-ai[context]
```

### Quick Start

```python
from argus.core.context_caching import (
    ContextCache, MemoryBackend, FileBackend,
    ConversationCache, EmbeddingCache, LLMResponseCache,
)

# Simple in-memory cache
cache = ContextCache(backend=MemoryBackend())
cache.set("key", {"data": "value"}, ttl=3600)
result = cache.get("key")

# Conversation cache for multi-turn context
conv_cache = ConversationCache(max_messages=100, max_tokens=8000)
conv_cache.add_message("user", "Hello, how are you?")
conv_cache.add_message("assistant", "I'm doing well, thank you!")
messages = conv_cache.get_messages()

# Embedding cache to reduce API calls
embed_cache = EmbeddingCache(
    backend=FileBackend(cache_dir=".embeddings_cache"),
    model_name="text-embedding-3-small",
)
```

### Decorator Pattern

```python
from argus.core.context_caching import ContextCache

cache = ContextCache(backend=MemoryBackend())

@cache.cached(ttl=3600)
def expensive_computation(input_data: str) -> dict:
    return {"result": process(input_data)}
```

---

## Context Compression

ARGUS includes advanced compression techniques to reduce token usage while preserving meaning.

### Features

- **Multiple compression methods**: Whitespace, Punctuation, Stopword, Sentence, Code, Semantic
- **Compression levels**: Minimal, Moderate, Aggressive, Extreme
- **Token counting**: Accurate token estimation with tiktoken
- **Message compression**: Optimize conversation history
- **Auto-detection**: Automatically select best method for content type

```python
from argus.core.context_compression import compress_text, CompressionLevel

result = compress_text(
    "This is a   very    long text   with   lots of   whitespace...",
    level=CompressionLevel.MODERATE,
)
print(result.compressed_text)
print(f"Savings: {result.savings_percentage:.1f}%")
```

---

## Debate Visualization

ARGUS includes a comprehensive visualization module for debate analysis and presentation.

### Features

- **Argument flow graphs**: NetworkX-based directed graphs
- **Timeline visualization**: Temporal argument progression
- **Agent performance charts**: Multi-metric agent analysis
- **Confidence evolution**: Rolling average tracking
- **Round summaries**: Per-round statistics
- **Interaction heatmaps**: Agent collaboration patterns
- **Interactive dashboards**: Combined multi-panel views
- **Export formats**: HTML, PNG, JSON reports

```bash
pip install argus-debate-ai[plotting]
```

### Quick Start

```python
from argus.debate.visualization import (
    DebateSession, create_debate_dashboard, export_debate_html, plot_argument_flow,
)

with open("debate_results.json") as f:
    session = DebateSession.from_dict(json.load(f))

fig = create_debate_dashboard(session)
export_debate_html(fig, "debate_dashboard.html")
```

### Available Charts

```python
from argus.debate.visualization import (
    plot_argument_flow,        # Hierarchical, radial, or force layout
    plot_debate_timeline,      # Temporal argument progression
    plot_agent_performance,    # Arguments, confidence, acceptance rate
    plot_confidence_evolution, # Rolling average with window_size
    plot_round_summary,        # Per-round stats breakdown
    plot_interaction_heatmap,  # Agent-to-agent interaction matrix
)
```

### CLI Usage

```bash
argus visualize debate_results.json --chart dashboard --output viz
argus visualize debate_results.json --chart flow --layout radial
argus visualize debate_results.json --format all --output debate_viz
```

---

## External Connectors

ARGUS provides connectors for fetching data from external sources.

### Web Connector (with robots.txt compliance)

```python
from argus.knowledge.connectors import WebConnector, WebConnectorConfig

config = WebConnectorConfig(
    respect_robots_txt=True,
    user_agent="ARGUS-Bot/1.0",
    timeout=30,
)

connector = WebConnector(config=config)
result = connector.fetch("https://example.com/article")

if result.success:
    doc = result.documents[0]
    print(f"Title: {doc.title}")
    print(f"Content: {doc.content[:500]}...")
```

### arXiv Connector

```python
from argus.knowledge.connectors import ArxivConnector, ArxivConnectorConfig

connector = ArxivConnector(config=ArxivConnectorConfig(
    sort_by="submittedDate",
    sort_order="descending",
))

result = connector.fetch(
    "machine learning transformers",
    max_results=10,
    categories=["cs.AI", "cs.LG"],
)

for doc in result.documents:
    print(f"Title: {doc.title}")
    print(f"arXiv ID: {doc.metadata['arxiv_id']}")
    print(f"PDF: {doc.metadata['pdf_url']}")
```

**Query Syntax:**
- Author: `au:Einstein`
- Title: `ti:quantum computing`
- Abstract: `abs:neural network`
- Category: `cat:cs.AI`
- Combined: `au:LeCun AND cat:cs.LG`

### CrossRef Connector

```python
from argus.knowledge.connectors import CrossRefConnector, CrossRefConnectorConfig

connector = CrossRefConnector(config=CrossRefConnectorConfig(
    mailto="your@email.com",  # For polite pool (faster rate limits)
))

result = connector.fetch_by_doi("10.1038/nature12373")
if result.success:
    doc = result.documents[0]
    print(f"Title: {doc.title}")
    print(f"Cited by: {doc.metadata['cited_by_count']}")
```

---

## Visualization & Plotting

ARGUS provides publication-quality visualization for debate results.

```bash
pip install argus-debate-ai[plotting]   # Static: matplotlib, seaborn
pip install argus-debate-ai[interactive] # Interactive: adds Plotly
```

### Available Plot Types

| Plot Type | Method | Description |
|-----------|--------|-------------|
| **Posterior Evolution** | `plot_posterior_evolution()` | Probability changes across rounds |
| **Evidence Distribution** | `plot_evidence_distribution()` | Support vs attack evidence |
| **CDAG Network** | `plot_cdag_network()` | Colour-coded argument graph |
| **Summary Radar** | `plot_summary_radar()` | Multi-metric comparison |
| **Interactive Posterior** | `plot_interactive_posterior()` | Zoomable, hoverable chart |
| **Combined Dashboard** | `plot_dashboard()` | Multi-plot HTML dashboard |

### Export Formats

| Format | Use Case |
|--------|----------|
| `png` | Web, presentations (300 DPI default) |
| `pdf` | Academic papers, print (vector) |
| `svg` | Web scalable graphics |
| `html` | Interactive Plotly only |

---

## Argus Terminal (TUI)

Argus includes a Bloomberg-style Terminal User Interface for interactive debates and research.

### Features
- **Retro Aesthetics**: Choose between 1980s Amber (financial) and 1970s Green (CRT) themes
- **Real-time Debate**: Watch agents debate, cite evidence, and reach verdicts live
- **System Monitoring**: Track token usage, costs, and agent states
- **Interactive Tools**: Browser-like tool execution within the terminal

```bash
argus-terminal
```

### Controls
- **1–8**: Switch screens (Dashboard, Debate, Providers, Tools, etc.)
- **Tab/Enter**: Navigate and select
- **q**: Quit

---

## Argus-Viz (Streamlit Sandbox)

**Argus-Viz** is an interactive Streamlit web application for experimenting with and visualising AI debates in real time.

### Features

| Feature | Description |
|---------|-------------|
| **Live Debate Arena** | Run debates with real-time streaming — posterior probability and debate flow graph update each round |
| **10 Interactive Charts** | Posterior evolution, evidence waterfall, CDAG network, specialist radar, confidence histogram, debate timeline, polarity donut, round heatmap, and full lifecycle DAG |
| **Debate Flow Explainer** | Sankey pipeline diagram, step-by-step explanations, Bayesian algorithm visualisation with LaTeX formulas |
| **Configurable Sidebar** | Pick LLM provider/model, set API key, adjust rounds, prior, jury threshold, toggle refuter |
| **Raw Data Export** | Download full debate results as JSON |

```bash
pip install argus-debate-ai[viz]

argus-viz
# or: streamlit run argus_viz/app.py
```

### Tabs

| Tab | What It Shows |
|-----|---------------|
| **⚔️ Debate Arena** | Live posterior chart + debate flow DAG; round logs; verdict card; evidence cards |
| **📊 Analysis Dashboard** | All 10 Plotly charts in a grid layout |
| **🗺️ Debate Flow** | ARGUS pipeline Sankey diagram, step explanations, Bayesian formula |
| **📋 Raw Data** | JSON result viewer, graph summary, download button |

---

## CRUX-Viz (CRUX Protocol Sandbox)

**CRUX-Viz** is a dedicated Streamlit sandbox for the CRUX protocol — visualising all **7 CRUX primitives** in real time.

### Features

| Feature | Description |
|---------|-------------|
| **⚡ CRUX Arena** | Live streaming — posterior, Claim Bundles, auctions, and BRP events in real time |
| **📦 Claim Bundle cards** | Every piece of evidence as a CRUX Claim Bundle with polarity badge, posterior, credibility |
| **🔀 BRP cards** | Belief Reconciliation Protocol sessions with contradiction Δ and reconciled posterior |
| **🏆 Auction cards** | Challenger Auction results with winner, bid count, and DFS score |
| **9 Interactive Charts** | Posterior evolution, CB timeline (gantt), KPI radar, BRP summary, credibility snapshot, and more |
| **📖 Protocol Explainer** | Interactive Sankey of the full CRUX pipeline + docs for all 7 primitives with LaTeX formulas |

```bash
pip install "argus-debate-ai[crux-viz]"

crux-viz
# or: streamlit run crux_viz/app.py
```

### CRUX-Specific Sidebar Options

```
Contradiction Threshold (θ)  — Default 0.20
    Minimum posterior gap that triggers BRP reconciliation

Enable EDR                   — Default On
    Create Epistemic Dead Reckoning checkpoints

Auction Timeout (s)          — Default 30
    Maximum time for Challenger Auction bidding window
```

---

## ARISTOTLE Chat Interface

**ARISTOTLE** (Autonomous Reasoning Intelligence for Structured Topic-Orchestrated Logical Engagement) transforms a single natural-language question into a fully autonomous, visualised, auditable multi-agent debate — from a WhatsApp-style chat interface.

### Launch

```bash
pip install "argus-debate-ai[aristotle]"

aristotle-chat
# or: streamlit run argus/aristotle/interface.py
```

### Architecture Layers

| Layer | Module | Role |
|-------|--------|------|
| **L1** | `framing.py` | Intent Parsing & Framing Engine — classifies debate type, extracts sub-claims |
| **L2** | `topology.py` | Dynamic Topology Builder — selects specialist agents, jury architecture, refuter intensity |
| **L3** | `monitor.py` | Autonomous Execution Engine — drives ARGUS rounds, enforces budgets |
| **L4** | `interface.py` | Single-Pane Split Streamlit UI with live DAG, belief trajectory, evidence heatmap |
| **L5** | `synthesis.py` | Plain-Language Output Synthesis — verdict narrative, dissent log, "What Could Change This" |

### UI Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  Status Bar (full width)                                         │
├────────────────────┬─────────────────────────────────────────────┤
│  LEFT  (38%)       │  RIGHT (62%)                                │
│  ARISTOTLE chat    │  Zone A: Full Lifecycle DAG (55%)           │
│                    │  Zone B: Belief + Heatmap (35%)             │
│  [input box]       │  Zone C: Expander (more panels)            │
└────────────────────┴─────────────────────────────────────────────┘
```

### Programmatic Usage

```python
from argus.aristotle import ARISTOTLE
from argus.core.llm import get_llm

agent = ARISTOTLE(llm=get_llm("openai", model="gpt-4o"))
result = agent.run("Is social media causing the mental health crisis?")

print(result.verdict_narrative)
print(result.dissent_log)
```

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
from argus.crux import CRUXOrchestrator, CRUXConfig

llm = get_llm("openai", model="gpt-4o")
base = RDCOrchestrator(llm=llm, max_rounds=5)

config = CRUXConfig(
    contradiction_threshold=0.20,
    enable_edr=True,
    enable_auction=True,
)
crux = CRUXOrchestrator(base=base, config=config)

result = crux.debate(
    "Treatment X reduces symptoms by more than 20%",
    prior=0.5,
)

print(f"Verdict: {result.verdict.label}")
print(f"Reconciled Posterior: {result.reconciled_cb.posterior:.3f}")
print(f"Credibility Scores: {result.credibility_snapshot}")
```

### Claim Bundle

```python
from argus.crux import ClaimBundle, BetaDistribution

bundle = ClaimBundle(
    claim_id="claim-001",
    text="The intervention reduces mortality by 15%",
    source_agent="specialist-clinical-001",
    confidence_distribution=BetaDistribution(alpha=8.0, beta=2.0),
    lineage=["evidence-001", "evidence-002"],
)

print(f"Posterior: {bundle.posterior:.3f}")     # Mean of Beta: α/(α+β)
print(f"Uncertainty: {bundle.uncertainty:.3f}") # Variance of Beta
print(f"95% CI: {bundle.credible_interval(0.95)}")
```

### Dialectical Fitness Score (DFS)

```
DFS(agent, claim) = w₁·domain_match + w₂·adversarial_potential + w₃·credibility + w₄·recency
```

### Belief Reconciliation Protocol (BRP)

```python
from argus.crux import BeliefReconciliationProtocol

brp = BeliefReconciliationProtocol(contradiction_threshold=0.20)

contradictions = brp.detect_contradictions([bundle1, bundle2, bundle3])
for contradiction in contradictions:
    result = brp.reconcile(contradiction)
    print(f"Merged Posterior: {result.merged_bundle.posterior:.3f}")
    print(f"Method: {result.method}")
    print(f"Proof: {result.proof_certificate}")
```

### Credibility Ledger

```python
from argus.crux import CredibilityLedger, CredibilityUpdate

ledger = CredibilityLedger()
ledger.record_update(
    agent_id="specialist-001",
    update=CredibilityUpdate(
        claim_id="claim-001",
        predicted_probability=0.75,
        actual_outcome=True,
    )
)

cred = ledger.get_credibility("specialist-001")
print(f"Credibility: {cred.score:.3f}")
print(f"Brier Score: {cred.brier_score:.3f}")

# Verify ledger integrity
assert ledger.verify_chain(), "Ledger tampered!"
```

**Hash Chain:**
```
entry_hash = SHA256(prev_hash || agent_id || update_data || timestamp)
```

### CRUX Configuration

```python
from argus.crux import CRUXConfig

config = CRUXConfig(
    contradiction_threshold=0.20,
    reconciliation_method="bayesian",
    dfs_domain_weight=0.3,
    dfs_adversarial_weight=0.3,
    dfs_credibility_weight=0.25,
    dfs_recency_weight=0.15,
    enable_edr=True,
    enable_auction=True,
    auction_timeout=30,
    initial_credibility=0.5,
    credibility_update_rate=0.1,
)
```

---

## Command Line Interface

ARGUS provides a full-featured CLI for common operations.

```bash
# Run a debate
argus debate "The hypothesis is supported by evidence" --prior 0.5 --rounds 3

# Quick evaluation
argus evaluate "Climate change increases wildfire frequency"

# Debate with specific provider
argus debate "Query" --provider anthropic --model claude-3-5-sonnet-20241022

# Verbose output with provenance
argus debate "Claim to evaluate" --verbose --provenance

# Ingest documents
argus ingest ./documents --output ./index
argus ingest ./papers --extensions pdf,md,txt

# Search the index
argus search "treatment efficacy" --index ./index --top-k 10

# List all 50+ tools
argus tools

# Generate debate dashboard
argus visualize debate_results.json --chart dashboard --output viz

# Show current configuration
argus config

# Validate API keys
argus config validate

# Version information
argus --version
```

---

## Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
export COHERE_API_KEY="..."
export MISTRAL_API_KEY="..."
export GROQ_API_KEY="gsk_..."

export ARGUS_DEFAULT_PROVIDER="openai"
export ARGUS_DEFAULT_MODEL="gpt-4o"
export ARGUS_TEMPERATURE="0.7"
export ARGUS_MAX_TOKENS="4096"
export ARGUS_OLLAMA_HOST="http://localhost:11434"
export ARGUS_LOG_LEVEL="INFO"
```

### Configuration File

Create `~/.argus/config.yaml`:

```yaml
default_provider: openai
default_model: gpt-4o
temperature: 0.7
max_tokens: 4096

llm:
  openai_api_key: ${OPENAI_API_KEY}
  anthropic_api_key: ${ANTHROPIC_API_KEY}
  ollama_host: http://localhost:11434

debate:
  max_rounds: 5
  min_evidence: 3
  convergence_threshold: 0.01

retrieval:
  embedding_model: all-MiniLM-L6-v2
  lambda_param: 0.7
  use_reranker: true

chunking:
  chunk_size: 512
  chunk_overlap: 50
  strategy: recursive
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
│  └────────┼────────────────┼────────────────┼────────────────┼─────────┘    │
│           ▼                ▼                ▼                ▼               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    C-DAG (Conceptual Debate Graph)                   │    │
│  │    ┌────────────┐      ┌────────────┐      ┌────────────┐           │    │
│  │    │Propositions│◀────▶│  Evidence  │◀────▶│  Rebuttals │           │    │
│  │    └────────────┘      └────────────┘      └────────────┘           │    │
│  │              Signed Influence Propagation (Log-Odds Bayesian)        │    │
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
│  │   Ingestion    │  │  Web · arXiv   │  │  PROV-O Ledger │                 │
│  │   Chunking     │  │   CrossRef     │  │   Hash Chain   │                 │
│  │   Embeddings   │  │   (Custom)     │  │  Attestations  │                 │
│  │  Hybrid Index  │  │                │  │    Queries     │                 │
│  └────────────────┘  └────────────────┘  └────────────────┘                 │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         LLM Provider Layer                           │    │
│  │  OpenAI · Anthropic · Gemini · Ollama · Cohere · Mistral · Groq     │    │
│  │  DeepSeek · xAI · Perplexity · Together · Fireworks · NVIDIA · ...  │    │
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

### Evidence Types

```python
from argus.cdag.nodes import EvidenceType

EvidenceType.EMPIRICAL       # Experimental/observational data
EvidenceType.THEORETICAL     # Theoretical arguments
EvidenceType.STATISTICAL     # Statistical analysis
EvidenceType.CASE_STUDY      # Case study evidence
EvidenceType.EXPERT_OPINION  # Expert testimony
EvidenceType.LITERATURE      # Literature review
EvidenceType.LOGICAL         # Logical argument
EvidenceType.METHODOLOGICAL  # Methodological critique
EvidenceType.ECONOMIC        # Economic analysis
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

### Expected Information Gain

```
EIG(a) = H(p) - 𝔼ᵧ[H(p|y)]
```

Where `H(p)` is current belief entropy and `𝔼ᵧ[H(p|y)]` is expected post-observation entropy.

### Calibration Methods

**Temperature Scaling:**
```
T* = argmin_T Σᵢ CrossEntropy(yᵢ, σ(zᵢ/T))
```

**Metrics:** Brier Score · ECE · MCE

```python
from argus.decision import Calibrator

calibrator = Calibrator()
calibrator.fit(logits, labels)

calibrated_probs = calibrator.calibrate(new_logits)
brier_score = calibrator.brier_score(labels, probs)
ece = calibrator.expected_calibration_error(labels, probs)
```

---

## ARGUS Evolution Extensions (v5.0)

Version 5.0 introduces eight production-ready extensions. All extensions are verified to import and run successfully, and every visualisation supports dual dark and light themes.

---

### CHRONOS — Temporal Evidence Decay

> Temporal C-DAG with exponential half-life decay and PELT-based belief drift detection.

**Key Classes:** `ChronosOrchestrator`, `TemporalCDAG`, `EvidenceHalfLifeRegistry`, `BeliefDriftDetector`

```python
from argus.chronos import ChronosOrchestrator, EvidenceHalfLifeRegistry
from argus.chronos.visualization import plot_temporal_posterior, plot_drift_timeline

registry = EvidenceHalfLifeRegistry()
registry.register("news", half_life_days=7)
registry.register("research_paper", half_life_days=365)
registry.register("social_media", half_life_days=1)

orchestrator = ChronosOrchestrator(base=rdc, half_life_registry=registry)
result = orchestrator.debate(
    "Interest rates will rise in Q3",
    reference_date="2025-01-01",
)

print(f"Posterior: {result.temporal_posterior.current_value:.3f}")
print(f"Drift events: {len(result.drift_report.inflection_points)}")

fig = plot_temporal_posterior(result.temporal_posterior, theme="dark")
fig.show()
```

**Algorithms:**
- **Exponential half-life decay**: `w(t) = w₀ × 2^(−Δt/t½)`
- **PELT change-point detection**: Pruned Exact Linear Time algorithm for inflection points
- **Causal Attribution**: Each drift event traced to specific evidence nodes

---

### PHALANX — Population-Scale Epistemic Simulation

> Thousands of cognitively-biased personas debate in parallel; polarisation measured with Jensen-Shannon divergence.

**Key Classes:** `PHALANXOrchestrator`, `EpistemicPersona`, `CognitiveBiasEngine`, `EmergentConsensusDetector`

```python
from argus.phalanx import PHALANXOrchestrator, PHALANXConfig

orchestrator = PHALANXOrchestrator(base=rdc, config=PHALANXConfig(
    population_size=500,
    parallel_workers=8,
))
result = orchestrator.debate("Universal Basic Income reduces poverty")

print(f"Polarisation Index (JSD): {result.consensus.polarisation_index.value:.3f}")
print(f"Bimodal: {result.consensus.is_bimodal}")
print(f"Dissent clusters: {len(result.consensus.dissent_clusters)}")
```

**Cognitive Biases Modelled:**
| Bias | Effect |
|------|--------|
| Confirmation | Amplifies evidence aligned with prior |
| Anchoring | Anchors posterior near initial estimate |
| Availability | Over-weights memorable examples |
| Authority | Scales confidence by claimed expertise |
| Recency | Over-weights recent evidence |

---

### SEED — Document-to-Debate Pipeline

> End-to-end ingestion of raw documents into debate-ready bundles with ranked, scored claims.

**Key Classes:** `SEEDOrchestrator`, `ClaimMiner`, `DebatabilityScorer`, `EvidencePrePopulator`

```python
from argus.seed import SEEDOrchestrator, SEEDConfig

orchestrator = SEEDOrchestrator(config=SEEDConfig(
    min_debatability_score=0.4,
    max_claims=20,
    enable_wikidata=True,
))

bundle = orchestrator.process("https://example.com/policy-report.pdf")

print(f"Claims extracted: {bundle.num_claims}")
for claim in bundle.top_claims(5):
    print(f"  [{claim.debatability_score:.2f}] {claim.text[:80]}")
```

**Debatability Score Formula:**
```
DebatabilityScore = 0.4 × BiPolarityRatio
                  + 0.35 × NoveltyQuotient
                  + 0.25 × EvidenceDensity
```

---

### MNEME — Persistent Agent Memory

> Agents remember past debates, grow expertise over time, and self-monitor calibration quality.

**Key Classes:** `MNEMEPlugin`, `KnowledgeReservoir`, `ExpertiseProfile`, `CalibrationHistory`

```python
from argus.mneme import MNEMEPlugin, MNEMEConfig

plugin = MNEMEPlugin(config=MNEMEConfig(
    backend="sqlite",
    db_path="./argus_memory.db",
    decay_rate=0.01,
))
plugin.attach(orchestrator)

# Expertise profile updates automatically after debates
for domain, competence in plugin.expertise_profile.top_domains(3):
    print(f"  {domain}: P(competent)={competence.mean:.3f}")

# Self-monitoring
drift = plugin.calibration_history.check_drift()
if drift.is_drifting:
    print(f"Calibration drift! Brier score: {drift.current_brier:.3f}")
```

**Supported Backends:** `memory` · `sqlite` · `postgres` · `qdrant`

---

### FRACTAL — Hierarchical Proposition Decomposition

> Complex propositions are decomposed into atomic sub-propositions, debated in parallel, and aggregated with relationship-aware Bayesian logic.

```python
from argus.fractal import FRACTALOrchestrator, FRACTALConfig

orchestrator = FRACTALOrchestrator(base=rdc, config=FRACTALConfig(
    max_depth=3,
    max_children=5,
    parallel_workers=4,
))

result = orchestrator.debate(
    "AI will surpass human intelligence AND cause economic disruption by 2035"
)

print(f"Tree nodes: {result.proposition_tree.num_nodes}")
print(f"Root posterior: {result.root_posterior:.3f}")
```

**Aggregation Strategies:**
| Relationship | Rule | Formula |
|-------------|------|---------|
| `NECESSARY` | AND (product) | P(parent) = ∏ P(childᵢ) |
| `SUFFICIENT` | OR (noisy-or) | P(parent) = 1 − ∏ (1 − P(childᵢ)) |
| `CONTRIBUTING` | Weighted Bayesian | P(parent) = Σ wᵢ × P(childᵢ) |
| `INDEPENDENT` | Geometric mean | P(parent) = (∏ P(childᵢ))^(1/n) |

---

### MIRROR — Consequence Inference Graph

> After verdict, two inference agents project downstream consequences and compute counterfactual sensitivity.

```python
from argus.mirror import MIRROROrchestrator

orchestrator = MIRROROrchestrator(base=rdc)
result = orchestrator.debate("Ban single-use plastics globally")

report = result.counterfactual_report
print(f"Max probability swing: {report.max_consequence_swing:.3f}")
print(f"Most sensitive category: {report.most_sensitive_category}")
print(report.narrative())
```

**Counterfactual Sensitivity:**
```
dP(consequence)/dP(root) = P(C | root=TRUE) − P(C | root=FALSE)
```

Nodes with |sensitivity| > 0.3 are flagged as **pivotal**.

---

### VERICHAIN — Cross-Debate Truth Network

> Persistent registry of signed verdict records forming a hash-chained truth network. Past verdicts are retrieved as epistemic precedents for new debates.

```python
from argus.verichain import VERICHAINRegistry, VERICHAINRetriever, EpistemicPrecedentInjector
from argus.verichain.integrity import ChainVerifier

registry = VERICHAINRegistry(backend="sqlite", db_path="./truth.db")

node = registry.register_verdict(
    proposition="Drug X reduces HbA1c by >1% in T2D",
    verdict="supported",
    posterior=0.78,
    domain="clinical",
    debate_id="debate_001",
)

# Retrieve precedents for a new debate
retriever = VERICHAINRetriever(nodes=registry.all_nodes)
precedents = retriever.retrieve("antidiabetic drug effectiveness", top_k=3)

injector = EpistemicPrecedentInjector()
plan = injector.plan_injection(precedents, proposition="Metformin is first-line therapy")
print(f"Prior adjustment: {plan.prior_adjustment:+.3f}")

# Verify chain integrity
verifier = ChainVerifier()
chain = verifier.verify_chain(registry.all_nodes)
print(f"Chain valid: {chain.is_valid} ({chain.chain_length} nodes)")
```

---

### PULSE — Operational Intelligence Dashboard

> Always-on monitoring with latency histograms, token metering, z-score anomaly detection, failure taxonomy, and auto-generated HTML dashboard.

```python
from argus.pulse import PULSEOrchestrator, PULSEConfig

pulse = PULSEOrchestrator(base=rdc, config=PULSEConfig(
    export_format="html",
    output_dir="./pulse_reports",
    anomaly_z_threshold=2.5,
))

for prop in propositions:
    result = pulse.debate(prop)  # Metrics collected automatically

report = pulse.dashboard.generate_report()
path = pulse.export_report()
print(f"Report exported to: {path}")
print(f"Anomalies detected: {len(report.anomalies)}")
```

**Failure Taxonomy:**
| Category | Trigger |
|----------|---------|
| `LLM_TIMEOUT` | LLM call exceeds deadline |
| `LLM_RATE_LIMIT` | HTTP 429 / rate limit errors |
| `EVIDENCE_EMPTY` | No chunks retrieved |
| `PROPAGATION_DIVERGENCE` | C-DAG propagation produces NaN/Inf |
| `VERDICT_ABSTAIN` | Jury abstains from verdict |

---

## Evaluation Framework

ARGUS includes a comprehensive evaluation framework for benchmarking and testing.

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
| `adversarial_edge_cases` | Adversarial | Stress testing |

### Global Benchmark Support

| Benchmark | Task |
|-----------|------|
| **FEVER** | Fact Verification |
| **SNLI/MultiNLI** | Natural Language Inference |
| **TruthfulQA** | Truthfulness Evaluation |
| **BoolQ** | Boolean QA |
| **ARC** | Science QA |

### Novel ARGUS Metrics

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
    def stream(self, prompt: str | List[Message], **kwargs) -> Iterator[str]: ...

    def embed(self, texts: str | List[str], **kwargs) -> List[List[float]]: ...
    def count_tokens(self, text: str) -> int: ...
```

---

## Examples

### Clinical Evidence Evaluation

```python
from argus import RDCOrchestrator, get_llm
from argus.retrieval import HybridRetriever
from argus.knowledge import DocumentLoader, Chunker

loader = DocumentLoader()
documents = [loader.load(f) for f in clinical_papers]

chunker = Chunker(chunk_size=512)
all_chunks = []
for doc in documents:
    all_chunks.extend(chunker.chunk(doc))

retriever = HybridRetriever(use_reranker=True)
retriever.index_chunks(all_chunks)

orchestrator = RDCOrchestrator(
    llm=get_llm("openai", model="gpt-4o"),
    max_rounds=5,
)

result = orchestrator.debate(
    "Metformin reduces HbA1c by >1% in Type 2 diabetes",
    prior=0.6,
    retriever=retriever,
    domain="clinical",
)

print(f"Verdict: {result.verdict.label}")
print(f"Posterior: {result.verdict.posterior:.3f}")
for e in result.evidence[:5]:
    print(f"  [{e.polarity:+d}] {e.text[:80]}...")
```

### Custom Agent Pipeline with Full Provenance

```python
from argus import get_llm, CDAG, Proposition
from argus.agents import Moderator, Specialist, Refuter, Jury
from argus.provenance import ProvenanceLedger, EventType

ledger = ProvenanceLedger()
ledger.record(EventType.SESSION_START)

# Different models for different roles
moderator = Moderator(get_llm("openai", model="gpt-4o"))
specialist = Specialist(get_llm("anthropic", model="claude-3-5-sonnet-20241022"), domain="policy")
refuter = Refuter(get_llm("groq", model="llama-3.1-70b-versatile"))
jury = Jury(get_llm("gemini", model="gemini-1.5-pro"))

graph = CDAG()
prop = Proposition(text="Carbon pricing is effective for reducing emissions", prior=0.5)
graph.add_proposition(prop)
ledger.record(EventType.PROPOSITION_ADDED, entity_id=prop.id)

for round_num in range(3):
    evidence = specialist.gather_evidence(graph, prop.id)
    for e in evidence:
        ledger.record(EventType.EVIDENCE_ADDED, entity_id=e.id)

    rebuttals = refuter.generate_rebuttals(graph, prop.id)
    for r in rebuttals:
        ledger.record(EventType.REBUTTAL_ADDED, entity_id=r.id)

    if moderator.should_stop(graph, prop.id):
        break

verdict = jury.evaluate(graph, prop.id)
ledger.record(EventType.VERDICT_RENDERED, entity_id=prop.id)
ledger.record(EventType.SESSION_END)

print(f"Verdict: {verdict.label}")
print(f"Posterior: {verdict.posterior:.3f}")
print(f"Ledger entries: {len(ledger)}")

is_valid, errors = ledger.verify_integrity()
print(f"Integrity: {'Valid' if is_valid else 'Invalid'}")
```

---

## Testing

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

# Skip slow/network tests
pytest -m "not slow"
```

### Test Categories

| Category | Path | Description |
|----------|------|-------------|
| Unit | `tests/unit/` | Isolated component tests |
| Integration | `tests/integration/` | Multi-component tests |
| E2E | `tests/e2e/` | Full workflow tests |

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

### AWS Lambda

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

### Good first issues

Looking for a place to start? Check out our [good first issues](https://github.com/Ronit26Mehta/argus-ai-debate/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) — labelled tasks that are well-scoped and beginner-friendly.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Ronit26Mehta/argus-ai-debate.git
cd argus-ai-debate

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

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Inspired by debate-native reasoning approaches in AI safety research
- Built on excellent open-source libraries:
  - [Pydantic](https://pydantic.dev/) — Data validation
  - [NetworkX](https://networkx.org/) — Graph algorithms
  - [FAISS](https://github.com/facebookresearch/faiss) — Vector search
  - [Sentence-Transformers](https://sbert.net/) — Embeddings
  - [BeautifulSoup](https://beautiful-soup-4.readthedocs.io/) — HTML parsing
- LLM integrations powered by OpenAI, Anthropic, Google, Cohere, Mistral, Groq, and Ollama

---

<div align="center">

**[PyPI](https://pypi.org/project/argus-debate-ai/)** · **[GitHub](https://github.com/Ronit26Mehta/argus-ai-debate)** · **[Issues](https://github.com/Ronit26Mehta/argus-ai-debate/issues)** · **[Discussions](https://github.com/Ronit26Mehta/argus-ai-debate/discussions)**

If ARGUS is useful to you, consider giving it a ⭐ — it helps others find the project.

</div>
