"""
PHALANX — Population-Scale Epistemic Simulation for ARGUS.

Wraps the existing RDCOrchestrator with a population layer. Spawns N
EpistemicPersona instances — each a lightweight virtual analyst with a
distinct cognitive prior, domain expertise weighting, and cognitive biases.

Key innovations:
    - CognitiveBiasEngine as quantitative DFS weight modifiers
    - Jensen-Shannon Polarisation Index over population quartiles
    - EmergentConsensusDetector for bimodality, variance, skew analysis

Example:
    >>> from argus.phalanx import PHALANXOrchestrator, PHALANXConfig
    >>> config = PHALANXConfig(population_size=200)
    >>> orchestrator = PHALANXOrchestrator(base=rdc, config=config)
    >>> result = orchestrator.debate('The intervention is cost-effective')
    >>> print(result.polarisation_index)
"""

from argus.phalanx.persona import (
    EpistemicPersona,
    PriorDistribution,
    ExpertiseProfile,
)
from argus.phalanx.bias_engine import (
    CognitiveBiasEngine,
    CognitiveBias,
    BiasWeightFn,
)
from argus.phalanx.population import (
    PersonaPopulation,
    PopulationSampler,
)
from argus.phalanx.runner import (
    ParallelPersonaRunner,
    MicroDebateResult,
)
from argus.phalanx.consensus import (
    EmergentConsensusDetector,
    PolarisationIndex,
    DissentCluster,
)
from argus.phalanx.posterior import (
    PopulationPosterior,
    BeliefDistribution,
)
from argus.phalanx.orchestrator import (
    PHALANXOrchestrator,
    PHALANXConfig,
    PHALANXResult,
)
from argus.phalanx.visualization import (
    plot_population_posterior,
    plot_bias_heatmap,
)

__all__ = [
    "EpistemicPersona", "PriorDistribution", "ExpertiseProfile",
    "CognitiveBiasEngine", "CognitiveBias", "BiasWeightFn",
    "PersonaPopulation", "PopulationSampler",
    "ParallelPersonaRunner", "MicroDebateResult",
    "EmergentConsensusDetector", "PolarisationIndex", "DissentCluster",
    "PopulationPosterior", "BeliefDistribution",
    "PHALANXOrchestrator", "PHALANXConfig", "PHALANXResult",
    "plot_population_posterior", "plot_bias_heatmap",
]
