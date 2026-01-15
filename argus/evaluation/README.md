# ARGUS Evaluation Framework

Comprehensive benchmarking, scoring, and testing infrastructure for the ARGUS AI Debate system.

## Quick Start

```python
from argus.evaluation import BenchmarkRunner, load_dataset
from argus.core.llm import GeminiLLM

# Initialize LLM
llm = GeminiLLM(api_key="YOUR_API_KEY")

# Run benchmarks
runner = BenchmarkRunner()
results = runner.run(llm)

# View results
for r in results:
    print(f"{r.benchmark_name}: accuracy={r.accuracy:.3f}")
```

---

## Datasets

### Custom ARGUS Datasets (10 domains, 1050+ samples each)

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

### Global Benchmarks via Hugging Face

Load industry-standard benchmarks directly from Hugging Face Hub:

| Benchmark | HF Dataset ID | Task | Samples |
|-----------|---------------|------|---------|
| **FEVER** | `fever` | Fact Verification | 185K |
| **SNLI** | `snli` | NLI | 570K |
| **MultiNLI** | `multi_nli` | Multi-Genre NLI | 433K |
| **TruthfulQA** | `truthful_qa` | Truthfulness | 817 |
| **BoolQ** | `google/boolq` | Yes/No QA | 16K |
| **ARC** | `allenai/ai2_arc` | Science QA | 8K |
| **HellaSwag** | `Rowan/hellaswag` | Commonsense | 70K |
| **WinoGrande** | `allenai/winogrande` | Winograd | 44K |

```python
# Install Hugging Face datasets
pip install datasets

# Load a benchmark
from argus.evaluation.datasets import load_global_benchmark
fever_df = load_global_benchmark("fever", max_samples=1000)
print(fever_df.head())
```

---

## Scoring Metrics

### Novel ARGUS Metrics (8 unique metrics)

| Metric | Full Name | Description |
|--------|-----------|-------------|
| **ARCIS** | Argus Reasoning Coherence Index Score | Logical consistency |
| **EVID-Q** | Evidence Quality Quotient | Evidence quality composite |
| **DIALEC** | Dialectical Depth Evaluation Coefficient | Debate sophistication |
| **REBUT-F** | Rebuttal Effectiveness Factor | Rebuttal impact |
| **CONV-S** | Convergence Stability Score | Posterior convergence |
| **PROV-I** | Provenance Integrity Index | Citation completeness |
| **CALIB-M** | Calibration Matrix Score | Confidence alignment |
| **EIG-U** | Expected Information Gain Utilization | Information efficiency |

### Standard Industry Metrics (11 global standards)

| Metric | Category | Description |
|--------|----------|-------------|
| **Accuracy** | Classification | Proportion correct |
| **Precision/Recall/F1** | Classification | Standard metrics |
| **Macro F1** | Multi-class | Average across classes |
| **Brier Score** | Calibration | Probability error (lower=better) |
| **ECE/MCE** | Calibration | Calibration error |
| **Log Loss** | Information | Cross-entropy |
| **Dialectical Balance** | Argumentation | Support/attack balance |

---

## Usage Examples

### Load Hugging Face Benchmark

```python
from argus.evaluation.datasets import (
    load_global_benchmark,
    list_global_benchmarks,
    get_benchmark_info,
)

# List available benchmarks
print(list_global_benchmarks())
# ['fever', 'snli', 'multinli', 'truthfulqa', 'boolq', 'arc', 'hellaswag', 'winogrande']

# Get benchmark details
info = get_benchmark_info("fever")
print(f"Dataset: {info.hf_dataset_id}, Task: {info.task_type}")

# Load FEVER
fever_df = load_global_benchmark("fever", max_samples=1000)

# Load SNLI
snli_df = load_global_benchmark("snli", max_samples=500)

# Load TruthfulQA
tqa_df = load_global_benchmark("truthfulqa")
```

### Compute All Metrics

```python
from argus.evaluation.scoring import (
    compute_all_scores,
    compute_all_standard_metrics,
    ScoreCard,
)

# Novel ARGUS scores
novel = compute_all_scores(debate_result)

# Standard metrics
standard = compute_all_standard_metrics(
    predictions, ground_truths,
    confidences=posteriors, outcomes=correct
)

# Score card
card = ScoreCard.from_result(debate_result)
print(card)
```

---

## CLI Usage

```bash
# Dry run (no LLM calls)
python -m argus.evaluation.runner.benchmark_runner --dry-run

# Full run
python -m argus.evaluation.runner.benchmark_runner \
    --datasets factual_claims \
    --benchmarks debate_quality \
    --max-samples 10 \
    --num-rounds 1
```

---

## Requirements

```bash
# Core requirements
pip install pandas numpy

# For Hugging Face benchmarks
pip install datasets

# For plotting
pip install matplotlib seaborn plotly
```
