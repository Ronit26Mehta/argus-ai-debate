# 🔴 fsociety — Multi-Agent VAPT Intelligence Terminal

<p align="center">
<img src="https://img.shields.io/badge/version-1.0.0-00ff41?style=for-the-badge&labelColor=0a0a0a" />
<img src="https://img.shields.io/badge/python-3.11+-00ff41?style=for-the-badge&logo=python&labelColor=0a0a0a" />
<img src="https://img.shields.io/badge/license-MIT-00ff41?style=for-the-badge&labelColor=0a0a0a" />
<img src="https://img.shields.io/badge/powered_by-ARGUS-00ff41?style=for-the-badge&labelColor=0a0a0a" />
</p>

> *"Hello, friend. Let me tell you everything that's wrong with your system."*

**fsociety** is a terminal-native, multi-agent **Vulnerability Assessment & Penetration Testing (VAPT)** intelligence platform. It deploys **13 specialized AI agents** — each modeled after a character from *Mr. Robot* — to adversarially debate, challenge, and converge on a target system's security posture using the [ARGUS](https://github.com/Ronit26Mehta/argus-ai-debate) debate framework.

---

## ⚡ How It Works

```
Target Codebase → Ingestion Engine → 13-Agent Adversarial Debate → Verdict → Report
                   (AST chunking,     (Multi-round RDC with        (P0-P3    (Markdown +
                    dependency scan,    Bayesian convergence)        rated)     JSON dossier)
                    git history)
```

1. **Ingest** — AST-aware code chunking, dependency scanning, git history analysis
2. **Debate** — Up to 6 rounds of multi-agent adversarial reasoning with Bayesian convergence
3. **Verdict** — Whiterose (the jury) renders P0-P3 severity verdicts with posteriors
4. **Report** — Executive summary, findings register, remediation roadmap

---

## 🕵️ The 13 Agents

| Tier | Agent | Domain |
|------|-------|--------|
| 🔴 Core | **ELLIOT** | Recon & Attack Surface Mapping |
| 🔴 Core | **MR.ROBOT** | Exploit Chain Builder & Severity Escalation |
| 🔴 Core | **DARLENE** | Auth/Logic Flaws, Social Engineering Surface |
| 🔴 Core | **WHITEROSE** | Jury — CVE Correlation, Bayesian Verdict, Compliance |
| 🔴 Core | **IRVING** | Master Orchestrator, DAG Lifecycle Manager |
| 🟡 Specialist | **ROMERO** | Malware Patterns, Legacy Vulnerability Research |
| 🟡 Specialist | **MOBLEY** | Network/Cloud Misconfiguration, Lateral Movement |
| 🟡 Specialist | **TRENTON** | Persistence Mechanisms, APT Pattern Matching |
| 🟡 Specialist | **TYRELL** | Insider Threat, Privilege Escalation |
| 🟡 Specialist | **ANGELA** | Phishing Surface, Credential Exposure |
| 🟡 Specialist | **DOM** | Blue Team / Defensive Posture Analysis |
| 🟢 Output | **LEON** | Remediation Planning, Patch Prioritization |
| 🟢 Output | **CISCO** | OSINT Enrichment, Breach Intelligence |

---

## 🚀 Installation

### Prerequisites

- **Python 3.11+**
- **ARGUS** (`argus-debate-ai>=5.5.0`) — install from the parent repo
- **Local LLM server** running at `localhost:8080` (e.g., [Ollama](https://ollama.ai), [LM Studio](https://lmstudio.ai), [llama.cpp server](https://github.com/ggerganov/llama.cpp), or [vLLM](https://vllm.ai))

### Install from source (development)

```bash
# 1. Clone the repo
git clone https://github.com/Ronit26Mehta/argus-ai-debate.git
cd argus-ai-debate

# 2. Install ARGUS first (if not already installed)
pip install -e .

# 3. Install fsociety
cd fsociety
pip install -e .
```

### Install from PyPI (when published)

```bash
pip install fsociety-vapt
```

---

## 🖥️ Start Your Local LLM

fsociety uses an **OpenAI-compatible local LLM** at `http://localhost:8080`. Start one of these before scanning:

**Ollama:**
```bash
ollama serve                          # starts on :11434 by default
# or with explicit port:
OLLAMA_HOST=0.0.0.0:8080 ollama serve
ollama run qwen2.5:7b
```

**LM Studio:**
- Open LM Studio → load a model → Start Server on port 8080

**llama.cpp server:**
```bash
./llama-server -m model.gguf --port 8080
```

**vLLM:**
```bash
vllm serve model-name --port 8080
```

---

## 📖 Usage

### CLI Scan — Analyze a Codebase

```bash
# Basic scan (surface depth, 6 rounds, default model)
fsociety scan --path /path/to/your/codebase

# Deep scan with custom model
fsociety scan --path ./my-project --depth deep --rounds 8 --model qwen2.5:7b

# Custom LLM server URL
fsociety scan --path ./my-project --base-url http://localhost:11434

# Scan with custom output directory
fsociety scan --path ./my-project --output ./my-reports
```

### List All Agents

```bash
fsociety agents
```

### Launch Terminal UI (TUI)

```bash
fsociety tui
```

### View Past Sessions

```bash
fsociety sessions
```

### Run as Python Module

```bash
python -m fsociety scan --path ./my-project
```

### Programmatic Usage

```python
from fsociety import VAPTOrchestrator, FsocietyConfig
from fsociety.config import LLMConfig

# Configure (defaults to localhost:8080)
config = FsocietyConfig(
    llm=LLMConfig(
        base_url="http://localhost:8080",
        model_name="qwen2.5:7b",
    )
)

# Run full pipeline
orchestrator = VAPTOrchestrator(config=config)
result = orchestrator.scan(path="/path/to/codebase")

# Access results
print(f"Total findings: {len(result['findings'])}")
print(f"Final posterior: {result['posteriors'][-1]:.1%}")
print(f"Report: {result['report_path']}")
```

---

## 📊 Output Structure

Each scan creates a structured output directory:

```
fsociety_reports/
└── <target_name>/
    └── fs-<session-id>/
        ├── report/
        │   ├── executive_summary.md     # Risk rating, severity breakdown
        │   ├── findings_register.md     # All findings with agent attribution
        │   └── remediation_roadmap.md   # Prioritized fix guidance
        ├── graphs/                      # Exploit chain visualizations
        ├── heatmaps/                    # Severity heatmaps
        ├── tables/                      # Compliance matrices
        └── raw/
            ├── debate_result.json       # Full debate data
            └── session.json             # Session metadata
```

---

## 🏗️ Architecture

```
fsociety (this package)
    │
    ├── config.py           ─── OpenAI-compatible LLM (localhost:8080)
    ├── models.py           ─── VKG node types, severity enums
    ├── vkg.py              ─── Vulnerability Knowledge Graph (extends ARGUS CDAG)
    ├── orchestrator.py     ─── Ingest → Debate → Report pipeline
    │
    ├── agents/             ─── 13 specialist agents (extends ARGUS BaseAgent)
    ├── ingestion/          ─── AST chunker, dep scanner, git analyzer
    ├── outputs/            ─── Directory manager, report builder
    │
    ├── cli.py              ─── Click CLI entry point
    └── tui.py              ─── Textual terminal UI
         │
         └── Built on: argus-debate-ai (ARGUS v5.5.0)
              ├── RDCOrchestrator (debate engine)
              ├── CDAG (Conceptual Debate Graph)
              ├── OpenAILLM (LLM provider)
              ├── BayesianUpdater (posterior computation)
              └── ProvenanceLedger (audit trail)
```

---

## ⚙️ Configuration

fsociety can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FSOCIETY_LLM__BASE_URL` | `http://localhost:8080` | Local LLM server URL |
| `FSOCIETY_LLM__MODEL_NAME` | `local-model` | Model name sent to server |
| `FSOCIETY_LLM__MAX_TOKENS` | `4096` | Max tokens per response |
| `FSOCIETY_LLM__TEMPERATURE` | `0.3` | Sampling temperature |
| `FSOCIETY_SCAN__MAX_DEBATE_ROUNDS` | `6` | Max debate rounds |
| `FSOCIETY_SCAN__POSTERIOR_THRESHOLD` | `0.85` | Early stopping threshold |
| `FSOCIETY_OUTPUT_DIR` | `./fsociety_reports` | Report output directory |

---

## 🧪 Testing

```bash
cd fsociety
python -m pytest tests/ -v --tb=short
```

---

## 📋 Severity Levels

| Level | Label | Action |
|-------|-------|--------|
| 🔴 P0 | CRITICAL | Immediate action — active exploitation likely |
| 🟠 P1 | HIGH | Fix within 48 hours |
| 🟡 P2 | MEDIUM | Schedule for next sprint |
| 🟢 P3 | LOW | Backlog |
| ℹ️ INFO | Informational | No action required |

---

## 🔮 Roadmap

- [ ] PDF report generation (WeasyPrint)
- [ ] MNEME integration (cross-session memory)
- [ ] VERICHAIN integration (finding registry)
- [ ] Active testing mode (`--active-test` flag)
- [ ] MITRE ATT&CK mapping visualizations
- [ ] Real-time OSINT API integrations
- [ ] Streamlit dashboard

---

## 📜 License

MIT License — see [LICENSE](../LICENSE) for details.

---

## 🙏 Credits

Built on top of **[ARGUS](https://github.com/Ronit26Mehta/argus-ai-debate)** — the multi-agent AI debate framework.

Agent personas inspired by [Mr. Robot](https://en.wikipedia.org/wiki/Mr._Robot) (USA Network).

---

<p align="center">
<b>fsociety</b> — <i>"Control is an illusion."</i>
</p>
