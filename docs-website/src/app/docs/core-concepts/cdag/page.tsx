import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'
import { Callout } from '@/components/Callout'

export const metadata: Metadata = {
    title: 'Conceptual Debate Graph (C-DAG) | ARGUS Documentation',
    description: 'Learn about the C-DAG - the core data structure for representing debates in ARGUS',
}

export default function CDAGPage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Conceptual Debate Graph (C-DAG)
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        A directed graph structure where propositions, evidence, and rebuttals are nodes with signed edges representing support/attack relationships.
                    </p>
                </div>

                {/* Overview */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Overview</h2>
                    <p className="text-muted-foreground">
                        The C-DAG is the core data structure in ARGUS for representing structured arguments. It enables:
                    </p>
                    <ul className="list-disc list-inside space-y-2 text-muted-foreground ml-4">
                        <li>Structured argument representation</li>
                        <li>Influence propagation via Bayesian updating</li>
                        <li>Conflict detection and resolution</li>
                        <li>Visual debugging and analysis</li>
                    </ul>
                </section>

                {/* Node Types */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Node Types</h2>

                    <div className="overflow-x-auto">
                        <table className="w-full border-collapse rounded-lg overflow-hidden">
                            <thead className="bg-muted">
                                <tr>
                                    <th className="px-4 py-3 text-left font-semibold">Type</th>
                                    <th className="px-4 py-3 text-left font-semibold">Description</th>
                                    <th className="px-4 py-3 text-left font-semibold">Key Attributes</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr className="border-t">
                                    <td className="px-4 py-3 font-mono text-sm">Proposition</td>
                                    <td className="px-4 py-3">Main claims under evaluation</td>
                                    <td className="px-4 py-3 text-sm">text, prior, domain, status</td>
                                </tr>
                                <tr className="border-t bg-muted/30">
                                    <td className="px-4 py-3 font-mono text-sm">Evidence</td>
                                    <td className="px-4 py-3">Supporting/attacking information</td>
                                    <td className="px-4 py-3 text-sm">polarity, confidence, source, type</td>
                                </tr>
                                <tr className="border-t">
                                    <td className="px-4 py-3 font-mono text-sm">Rebuttal</td>
                                    <td className="px-4 py-3">Challenges to evidence</td>
                                    <td className="px-4 py-3 text-sm">target_id, strength, rebuttal_type</td>
                                </tr>
                                <tr className="border-t bg-muted/30">
                                    <td className="px-4 py-3 font-mono text-sm">Finding</td>
                                    <td className="px-4 py-3">Intermediate conclusions</td>
                                    <td className="px-4 py-3 text-sm">derived_from, confidence</td>
                                </tr>
                                <tr className="border-t">
                                    <td className="px-4 py-3 font-mono text-sm">Assumption</td>
                                    <td className="px-4 py-3">Underlying premises</td>
                                    <td className="px-4 py-3 text-sm">explicit, challenged</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </section>

                {/* Edge Types */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Edge Types</h2>

                    <div className="grid gap-4 md:grid-cols-2">
                        <div className="p-4 rounded-lg border bg-green-50 dark:bg-green-950/20 border-green-200 dark:border-green-900/50">
                            <h4 className="font-semibold mb-2 text-green-900 dark:text-green-300">SUPPORTS (+1)</h4>
                            <p className="text-sm text-muted-foreground">
                                Evidence supporting a proposition
                            </p>
                        </div>
                        <div className="p-4 rounded-lg border bg-red-50 dark:bg-red-950/20 border-red-200 dark:border-red-900/50">
                            <h4 className="font-semibold mb-2 text-red-900 dark:text-red-300">ATTACKS (-1)</h4>
                            <p className="text-sm text-muted-foreground">
                                Evidence challenging a proposition
                            </p>
                        </div>
                        <div className="p-4 rounded-lg border bg-amber-50 dark:bg-amber-950/20 border-amber-200 dark:border-amber-900/50">
                            <h4 className="font-semibold mb-2 text-amber-900 dark:text-amber-300">REBUTS (-1)</h4>
                            <p className="text-sm text-muted-foreground">
                                Rebuttal targeting evidence
                            </p>
                        </div>
                        <div className="p-4 rounded-lg border bg-blue-50 dark:bg-blue-950/20 border-blue-200 dark:border-blue-900/50">
                            <h4 className="font-semibold mb-2 text-blue-900 dark:text-blue-300">REFINES (0)</h4>
                            <p className="text-sm text-muted-foreground">
                                Clarification or specification
                            </p>
                        </div>
                    </div>
                </section>

                {/* Basic Usage */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Basic Usage</h2>
                    <p className="text-muted-foreground">
                        Create and manipulate a C-DAG:
                    </p>

                    <CodeBlock
                        code={`from argus import CDAG, Proposition, Evidence, EdgeType
from argus.cdag.nodes import EvidenceType
from argus.cdag.propagation import compute_posterior

# Create graph
graph = CDAG(name="my_debate")

# Add proposition
prop = Proposition(
    text="AI will transform healthcare",
    prior=0.5,
    domain="technology",
)
graph.add_proposition(prop)

# Add supporting evidence
evidence = Evidence(
    text="FDA approved 521 AI medical devices in 2023",
    evidence_type=EvidenceType.EMPIRICAL,
    polarity=1,  # +1 supports, -1 attacks
    confidence=0.9,
    relevance=0.85,
)
graph.add_evidence(evidence, prop.id, EdgeType.SUPPORTS)

# Compute posterior probability
posterior = compute_posterior(graph, prop.id)
print(f"Posterior: {posterior:.3f}")`}
                        language="python"
                        filename="basic_cdag.py"
                    />
                </section>

                {/* Evidence Types */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Evidence Types</h2>
                    <p className="text-muted-foreground">
                        Different types of evidence for different domains:
                    </p>

                    <CodeBlock
                        code={`from argus.cdag.nodes import EvidenceType

# Available types
EvidenceType.EMPIRICAL      # Data, studies, measurements
EvidenceType.THEORETICAL    # Logical/theoretical argument
EvidenceType.STATISTICAL    # Statistical analysis
EvidenceType.CASE_STUDY     # Case studies, examples
EvidenceType.EXPERT_OPINION # Expert testimony
EvidenceType.LITERATURE     # Literature review
EvidenceType.LOGICAL        # Logical argument
EvidenceType.METHODOLOGICAL # Methodological critique
EvidenceType.ECONOMIC       # Economic analysis`}
                        language="python"
                    />
                </section>

                {/* Belief Propagation */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Bayesian Belief Propagation</h2>
                    <p className="text-muted-foreground">
                        The C-DAG uses log-odds space for numerically stable Bayesian belief updating:
                    </p>

                    <div className="p-6 rounded-xl border bg-muted/30">
                        <p className="font-mono text-center text-lg">
                            posterior = σ(log-odds(prior) + Σᵢ wᵢ · log(LRᵢ))
                        </p>
                    </div>

                    <p className="text-sm text-muted-foreground">
                        Where:
                    </p>
                    <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground ml-4">
                        <li><code>σ</code> is the logistic (sigmoid) function</li>
                        <li><code>LRᵢ</code> is the likelihood ratio for evidence i</li>
                        <li><code>wᵢ = polarityᵢ × confidenceᵢ × relevanceᵢ × qualityᵢ</code></li>
                    </ul>

                    <CodeBlock
                        code={`from argus.cdag.propagation import (
    compute_posterior,
    propagate_beliefs,
    log_odds_update,
)

# Compute posterior using log-odds Bayesian updating
posterior = compute_posterior(graph, proposition_id)

# Propagate beliefs through entire graph
posteriors = propagate_beliefs(graph, max_iterations=10)

# Manual log-odds update
new_odds = log_odds_update(
    prior_odds=1.0,
    likelihood_ratio=2.5,
    confidence=0.8,
)`}
                        language="python"
                    />
                </section>

                {/* Graph Operations */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Graph Operations</h2>

                    <div className="space-y-6">
                        <div>
                            <h3 className="text-xl font-semibold mb-3">Querying the Graph</h3>
                            <CodeBlock
                                code={`# Get all evidence for a proposition
evidence_list = graph.get_evidence(prop.id)

# Get rebuttals targeting evidence
rebuttals = graph.get_rebuttals(evidence.id)

# Get all nodes of a specific type
all_propositions = graph.get_nodes_by_type("proposition")

# Get neighbors
neighbors = graph.get_neighbors(node_id)`}
                                language="python"
                            />
                        </div>

                        <div>
                            <h3 className="text-xl font-semibold mb-3">Serialization</h3>
                            <CodeBlock
                                code={`# Export to JSON
graph.save("debate.json")

# Load from JSON
graph = CDAG.load("debate.json")

# Export to dict
graph_dict = graph.to_dict()

# Convert to NetworkX for visualization
import networkx as nx
nx_graph = graph.to_networkx()`}
                                language="python"
                            />
                        </div>

                        <div>
                            <h3 className="text-xl font-semibold mb-3">Visualization</h3>
                            <CodeBlock
                                code={`# Visualize with matplotlib
graph.visualize("cdag_plot.png")

# Interactive visualization with plotly
from argus.outputs import InteractivePlotter

plotter = InteractivePlotter()
plotter.plot_interactive_network(graph)`}
                                language="python"
                            />
                        </div>
                    </div>
                </section>

                {/* Complete Example */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Complete Example</h2>
                    <p className="text-muted-foreground">
                        Building a complex debate graph:
                    </p>

                    <CodeBlock
                        code={`from argus import CDAG, Proposition, Evidence, Rebuttal, EdgeType
from argus.cdag.nodes import EvidenceType
from argus.cdag.propagation import compute_posterior

# Create graph
graph = CDAG(name="drug_efficacy_debate")

# Add proposition
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

# Get summary statistics
print(f"Total nodes: {len(graph.nodes)}")
print(f"Total edges: {len(graph.edges)}")
print(f"Evidence count: {len(graph.get_evidence(prop.id))}")

# Export for analysis
graph.save("drug_efficacy_debate.json")`}
                        language="python"
                        filename="complete_cdag_example.py"
                    />
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a
                            href="/docs/core-concepts/agents"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">Agents →</h3>
                            <p className="text-sm text-muted-foreground">
                                Learn about the multi-agent system
                            </p>
                        </a>
                        <a
                            href="/docs/modules/decision"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">Decision Module →</h3>
                            <p className="text-sm text-muted-foreground">
                                Bayesian updating and VoI planning
                            </p>
                        </a>
                        <a
                            href="/docs/modules/provenance"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">Provenance →</h3>
                            <p className="text-sm text-muted-foreground">
                                Track and verify debate integrity
                            </p>
                        </a>
                    </div>
                </section>
            </div>
        </div>
    )
}
