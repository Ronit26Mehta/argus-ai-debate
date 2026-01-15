"""
ARGUS Novel Scoring Metrics.

Unique metrics designed specifically for evaluating multi-agent AI debate systems.
These metrics are original to ARGUS and not found elsewhere.
"""

from argus.evaluation.scoring.metrics import (
    # Individual metrics
    compute_arcis,
    compute_evid_q,
    compute_dialec,
    compute_rebut_f,
    compute_conv_s,
    compute_prov_i,
    compute_calib_m,
    compute_eig_u,
    # Aggregated scoring
    compute_all_scores,
    ScoreCard,
    MetricDefinition,
    METRIC_REGISTRY,
)

from argus.evaluation.scoring.aggregate import (
    CompositeScore,
    ScoreComparison,
    generate_score_report,
)

from argus.evaluation.scoring.standard_metrics import (
    # Standard classification metrics
    compute_accuracy,
    compute_precision_recall_f1,
    compute_macro_f1,
    # Calibration metrics
    compute_brier_score,
    compute_ece,
    compute_mce,
    # Information-theoretic
    compute_cross_entropy,
    compute_log_loss,
    # Argumentation metrics
    compute_argument_coverage,
    compute_dialectical_balance,
    compute_verdict_confidence_correlation,
    # Aggregate
    compute_all_standard_metrics,
    StandardMetricsResult,
    STANDARD_METRIC_DESCRIPTIONS,
)

__all__ = [
    # Novel ARGUS metrics
    "compute_arcis",
    "compute_evid_q",
    "compute_dialec",
    "compute_rebut_f",
    "compute_conv_s",
    "compute_prov_i",
    "compute_calib_m",
    "compute_eig_u",
    "compute_all_scores",
    "ScoreCard",
    "MetricDefinition",
    "METRIC_REGISTRY",
    # Aggregation
    "CompositeScore",
    "ScoreComparison",
    "generate_score_report",
    # Standard metrics
    "compute_accuracy",
    "compute_precision_recall_f1",
    "compute_macro_f1",
    "compute_brier_score",
    "compute_ece",
    "compute_mce",
    "compute_cross_entropy",
    "compute_log_loss",
    "compute_argument_coverage",
    "compute_dialectical_balance",
    "compute_verdict_confidence_correlation",
    "compute_all_standard_metrics",
    "StandardMetricsResult",
    "STANDARD_METRIC_DESCRIPTIONS",
]
