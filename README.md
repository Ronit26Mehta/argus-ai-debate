# ARGUS

**Agentic Research & Governance Unified System**

*A debate-native, multi-agent AI framework for evidence-based reasoning with structured argumentation, decision-theoretic planning, and full provenance tracking.*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/argus-debate-ai.svg)](https://pypi.org/project/argus-debate-ai/)

---

## Overview

ARGUS implements **Research Debate Chain (RDC)** - a novel approach to AI reasoning that structures knowledge evaluation as multi-agent debates. Instead of single-pass inference, ARGUS orchestrates specialist agents that gather evidence, generate rebuttals, and render verdicts through Bayesian aggregation.

### Key Innovations

- **Conceptual Debate Graph (C-DAG)**: A directed graph structure where propositions, evidence, and rebuttals are nodes with signed edges representing support/attack relationships
- **Evidence-Directed Debate Orchestration (EDDO)**: Algorithm for managing multi-round debates with stopping criteria
- **Value of Information Planning**: Decision-theoretic experiment selection using Expected Information Gain
- **Full Provenance**: PROV-O compatible ledger with hash-chain integrity for audit trails

---

## Features

### Multi-Agent Debate System
- **Moderator**: Creates debate agendas, manages rounds, evaluates stopping criteria
- **Specialist Agents**: Domain-specific evidence gathering with hybrid retrieval
- **Refuter**: Generates counter-evidence and methodological critiques
- **Jury**: Aggregates evidence via Bayesian updating, renders verdicts

### Conceptual Debate Graph (C-DAG)
- **Node Types**: Propositions, Evidence, Rebuttals, Findings, Assumptions
- **Edge Types**: Supports, Attacks, Refines, Rebuts with signed weights
- **Propagation**: Log-odds Bayesian belief updating across the graph
- **Visualization**: Export to NetworkX for analysis

### Hybrid Retrieval System
- **BM25 Sparse**: Traditional keyword-based retrieval
- **FAISS Dense**: Semantic vector search with sentence-transformers
- **Fusion Methods**: Weighted combination or Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Reranking**: Neural reranking for precision

### Decision-Theoretic Planning
- **Expected Information Gain (EIG)**: Monte Carlo estimation for experiment value
- **VoI Planner**: Knapsack-based optimal action selection under budget
- **Calibration**: Brier score, ECE, temperature scaling for confidence tuning

### Provenance & Governance
- **PROV-O Compatible**: W3C standard provenance model
- **Hash-Chain Integrity**: SHA-256 linked events for tamper detection
- **Attestations**: Cryptographic proofs for content integrity
- **Query API**: Filter events by entity, agent, time range

### LLM Provider Support

| Provider | Models | Features |
|----------|--------|----------|
| **OpenAI** | GPT-4o, GPT-4, GPT-3.5 | Generate, Stream, Embed |
| **Anthropic** | Claude 3.5, Claude 3 | Generate, Stream |
| **Google** | Gemini 1.5 Pro/Flash | Generate, Stream, Embed |
| **Ollama** | Llama, Mistral, Phi | Local deployment |

---

## Installation

### From PyPI

```bash
pip install argus-debate-ai
```

### From Source

```bash
git clone https://github.com/argus-ai/argus.git
cd argus
pip install -e ".[dev]"
```

### Optional Dependencies

```bash
# For all features including dev tools
pip install argus-debate-ai[all]

# For Ollama local LLM support
pip install argus-debate-ai[ollama]
```

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

# Compute Bayesian posterior
posterior = compute_posterior(graph, prop.id)
print(f"Posterior probability: {posterior:.3f}")
```

### Document Ingestion & Retrieval

```python
from argus import DocumentLoader, Chunker, EmbeddingGenerator
from argus.retrieval import HybridRetriever

# Load documents
loader = DocumentLoader()
doc = loader.load("research_paper.pdf")

# Chunk with overlap
chunker = Chunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk(doc)

# Create hybrid retriever
retriever = HybridRetriever(
    embedding_model="all-MiniLM-L6-v2",
    lambda_param=0.7,  # Weight toward dense retrieval
    use_reranker=True,
)
retriever.index_chunks(chunks)

# Search
results = retriever.retrieve("treatment efficacy results", top_k=10)
for r in results:
    print(f"[{r.rank}] Score: {r.score:.3f} - {r.chunk.text[:100]}...")
```

### Multi-Agent Debate

```python
from argus import get_llm
from argus.agents import Moderator, Specialist, Refuter, Jury
from argus import CDAG, Proposition

llm = get_llm("anthropic", model="claude-3-5-sonnet-20241022")

# Initialize agents
moderator = Moderator(llm)
specialist = Specialist(llm, domain="clinical")
refuter = Refuter(llm)
jury = Jury(llm)

# Create debate
graph = CDAG()
prop = Proposition(text="The intervention is cost-effective", prior=0.5)
graph.add_proposition(prop)

# Moderator creates agenda
agenda = moderator.create_agenda(graph, prop.id)

# Specialists gather evidence
evidence = specialist.gather_evidence(graph, prop.id)

# Refuter challenges
rebuttals = refuter.generate_rebuttals(graph, prop.id)

# Jury renders verdict
verdict = jury.evaluate(graph, prop.id)
print(f"Verdict: {verdict.label} (posterior={verdict.posterior:.3f})")
```

---

## Command Line Interface

ARGUS provides a CLI for common operations:

```bash
# Run a debate
argus debate "The hypothesis is supported by evidence" --prior 0.5 --rounds 3

# Quick evaluation (single LLM call)
argus evaluate "Climate change increases wildfire frequency"

# Ingest documents into index
argus ingest ./documents --output ./index

# Show configuration
argus config

# Specify provider
argus debate "Query" --provider anthropic --model claude-3-5-sonnet-20241022
```

---

## Configuration

### Environment Variables

```bash
# LLM API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."

# Default settings
export ARGUS_DEFAULT_PROVIDER="openai"
export ARGUS_DEFAULT_MODEL="gpt-4o"
export ARGUS_TEMPERATURE="0.7"
export ARGUS_MAX_TOKENS="2048"

# Ollama (local)
export ARGUS_OLLAMA_HOST="http://localhost:11434"
```

### Programmatic Configuration

```python
from argus import ArgusConfig, get_config

config = ArgusConfig(
    default_provider="anthropic",
    default_model="claude-3-5-sonnet-20241022",
    temperature=0.5,
    max_tokens=4096,
)

# Or get global config
config = get_config()
```

---

## Architecture

```
+-----------------------------------------------------------------------------+
|                              ARGUS Architecture                              |
+-----------------------------------------------------------------------------+
|                                                                              |
|  +---------------+    +---------------+    +---------------+                |
|  |   Moderator   |--->|  Specialists  |--->|    Refuter    |                |
|  |   (Planner)   |    |  (Evidence)   |    | (Challenges)  |                |
|  +-------+-------+    +-------+-------+    +-------+-------+                |
|          |                    |                    |                         |
|          v                    v                    v                         |
|  +---------------------------------------------------------------------+    |
|  |                    C-DAG (Debate Graph)                              |    |
|  |  +--------+     +----------+     +----------+                        |    |
|  |  | Props  |---->| Evidence |---->| Rebuttals|                        |    |
|  |  +--------+     +----------+     +----------+                        |    |
|  |         ^              |               |                              |    |
|  |         +--------------+---------------+                              |    |
|  |                 Signed Influence Propagation                         |    |
|  +---------------------------------------------------------------------+    |
|                                    |                                         |
|                                    v                                         |
|  +---------------------------------------------------------------------+    |
|  |                         Jury (Verdict)                               |    |
|  |           Bayesian Aggregation -> Posterior -> Label                 |    |
|  +---------------------------------------------------------------------+    |
|                                                                              |
|  +-----------------+  +-----------------+  +-----------------+              |
|  | Knowledge Layer |  | Decision Layer  |  |   Provenance    |              |
|  | - Ingestion     |  | - Bayesian      |  | - PROV-O Ledger |              |
|  | - Chunking      |  | - EIG/VoI       |  | - Hash Chain    |              |
|  | - Hybrid Index  |  | - Calibration   |  | - Attestations  |              |
|  +-----------------+  +-----------------+  +-----------------+              |
|                                                                              |
+-----------------------------------------------------------------------------+
```

### Module Overview

| Module | Description |
|--------|-------------|
| `argus.core` | Configuration, data models, LLM abstractions |
| `argus.cdag` | Conceptual Debate Graph implementation |
| `argus.decision` | Bayesian updating, EIG, VoI planning, calibration |
| `argus.knowledge` | Document ingestion, chunking, embeddings, indexing |
| `argus.retrieval` | Hybrid retrieval, reranking |
| `argus.agents` | Moderator, Specialist, Refuter, Jury agents |
| `argus.provenance` | PROV-O ledger, integrity, attestations |
| `argus.orchestrator` | RDC orchestration engine |

---

## Algorithms

### Signed Influence Propagation

The C-DAG uses log-odds space for numerically stable belief propagation:

```
posterior = sigmoid(log-odds(prior) + sum(signed_weight_i * log(LR_i)))
```

Where:
- `sigmoid` is the logistic function
- `LR_i` is the likelihood ratio for evidence i
- `signed_weight = polarity * confidence * relevance * quality`

### Expected Information Gain

For experiment planning, ARGUS computes EIG via Monte Carlo:

```
EIG(a) = H(p) - E_y[H(p|y)]
```

Where `H` is entropy and the expectation is over possible outcomes.

### Calibration

Temperature scaling optimizes:

```
T* = argmin_T sum(-y_i * log(sigmoid(z_i/T)) - (1-y_i) * log(1-sigmoid(z_i/T)))
```

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=argus --cov-report=html

# Run specific test module
pytest tests/unit/test_cdag.py -v

# Run integration tests
pytest tests/integration/ -v
```

---

## Examples

### Clinical Evidence Evaluation

```python
from argus import RDCOrchestrator, get_llm
from argus.retrieval import HybridRetriever

# Load clinical literature
retriever = HybridRetriever()
retriever.index_chunks(clinical_chunks)

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
```

### Research Claim Verification

```python
from argus import CDAG, Proposition, Evidence
from argus.cdag.propagation import compute_all_posteriors

graph = CDAG(name="research_verification")

# Main claim
claim = Proposition(
    text="Neural scaling laws predict emergent capabilities",
    prior=0.5,
)
graph.add_proposition(claim)

# Add evidence from multiple papers
# ... (add supporting/attacking evidence)

# Compute all posteriors
posteriors = compute_all_posteriors(graph)

for prop_id, posterior in posteriors.items():
    prop = graph.get_proposition(prop_id)
    print(f"{prop.text[:50]}... : {posterior:.3f}")
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Inspired by debate-native reasoning approaches in AI safety research
- Built on excellent open-source libraries: Pydantic, NetworkX, FAISS, Sentence-Transformers
- LLM integrations powered by OpenAI, Anthropic, and Google APIs

---

**[PyPI](https://pypi.org/project/argus-debate-ai/)** | **[GitHub](https://github.com/argus-ai/argus)**
