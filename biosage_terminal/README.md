# BioSage Terminal

**Version 2.0.0** | **Production Release**

[![PyPI version](https://img.shields.io/pypi/v/biosage-terminal.svg)](https://pypi.org/project/biosage-terminal/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production](https://img.shields.io/badge/Status-Production-green.svg)]()

AI-Powered Medical Diagnostic Assistant - Terminal User Interface

## Overview

BioSage Terminal is a sophisticated TUI (Terminal User Interface) application for medical diagnostics powered by AI. It leverages the ARGUS debate-based reasoning framework to provide evidence-based diagnostic suggestions through multiple specialist AI agents.

## Features

- **Full ARGUS Debate Framework Integration**:
  - **Moderator Agent**: Orchestrates debate agenda and controls flow
  - **Specialist Agents**: Infectious Disease, Cardiology, Neurology, Oncology, Autoimmune, and Toxicology experts
  - **Refuter Agent**: Challenges weak evidence and generates rebuttals
  - **Jury Agent**: Renders final verdicts via Bayesian aggregation
  - **CDAG**: Conceptual Debate Graph tracks all propositions, evidence, and rebuttals
  
- **Evidence-Based Reasoning**: Uses Bayesian posterior computation for calibrated confidence scores
- **Patient Management**: Complete patient onboarding, vitals tracking, and case management
- **Persistent Storage**: Local JSON-based storage for all patient data and cases
- **Beautiful TUI**: Rich terminal interface with intuitive navigation using Textual framework

## Installation

```bash
pip install biosage-terminal

# Install with preferred LLM provider
pip install biosage-terminal[gemini]   # Google Gemini (recommended)
pip install biosage-terminal[openai]   # OpenAI
pip install biosage-terminal[anthropic] # Anthropic Claude
pip install biosage-terminal[all-llms]  # All providers
```

## Quick Start

After installation, simply run:

```bash
biosage
```

Or check your API configuration first:

```bash
biosage --check-api
```

## Configuration

API keys and model names are configured via environment variables. BioSage auto-detects available providers in this order:
**Gemini > OpenAI > Anthropic > Groq > Mistral > Cohere > Ollama**

### Environment Variables

```bash
# API Keys
export GEMINI_API_KEY=your_key_here        # Google Gemini (recommended)
export OPENAI_API_KEY=your_key_here        # OpenAI
export ANTHROPIC_API_KEY=your_key_here     # Anthropic Claude
export GROQ_API_KEY=your_key_here          # Groq
export MISTRAL_API_KEY=your_key_here       # Mistral
export COHERE_API_KEY=your_key_here        # Cohere

# Optional: Custom model names
export GEMINI_MODEL=gemini-1.5-pro
export OPENAI_MODEL=gpt-4o
export ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
export GROQ_MODEL=llama-3.1-70b-versatile

# For local Ollama (no API key needed)
export OLLAMA_MODEL=llama3.1

# Data storage directory (default: ~/.biosage)
export BIOSAGE_DATA_DIR=/path/to/data
```

You can also create a `.env` file in your working directory.

## Navigation

| Key | Action |
|-----|--------|
| `d` | Dashboard |
| `o` | Patient Onboarding (New Case) |
| `r` | Run Diagnosis |
| `s` | Specialist Grid |
| `e` | Evidence Explorer |
| `Tab` | Switch tabs / Next field |
| `Enter` | Select / Confirm |
| `Esc` | Back / Cancel |
| `q` | Quit |

## Data Storage

All data is stored locally in `~/.biosage/`:

- `patients/` - Patient records (JSON)
- `cases/` - Diagnostic cases with ARGUS debate results
- `evidence/` - Evidence and citations
- `reports/` - Generated reports
- `audit/` - Audit trail of all actions

## ARGUS Debate Architecture

BioSage Terminal uses the full ARGUS multi-agent debate framework:

1. **Moderator** creates debate agenda for the diagnosis
2. **Specialist Agents** (6 domains) gather domain-specific evidence
3. **Refuter Agent** challenges evidence and generates rebuttals
4. **CDAG** (Conceptual Debate Graph) tracks:
   - Propositions (diagnosis hypotheses)
   - Evidence nodes (supporting/attacking)
   - Rebuttal nodes (challenges to evidence)
5. **Bayesian Propagation** computes posterior probabilities
6. **Jury Agent** renders final verdicts with:
   - Verdict label (supported/rejected/undecided)
   - Posterior confidence
   - Clinical reasoning

## Development

```bash
git clone https://github.com/biosage/biosage-terminal
cd biosage-terminal
pip install -e .[dev]
```

## Changelog

### v2.0.0 (Production Release)
- **Production-ready release** with stable APIs
- Full ARGUS debate framework integration
- Multi-provider LLM support (Gemini, OpenAI, Anthropic, Groq, Mistral, Cohere, Ollama)
- Complete patient management and case tracking
- Evidence-based Bayesian diagnostic reasoning
- Rich terminal UI with intuitive navigation

## License

MIT License
