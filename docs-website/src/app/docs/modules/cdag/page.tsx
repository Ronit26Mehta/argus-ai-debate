import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'
import { Callout } from '@/components/Callout'

export const metadata: Metadata = {
    title: 'C-DAG Module | ARGUS Documentation',
    description: 'Complete guide to the C-DAG module - Conceptual Debate Argumentation Graph',
}

export default function CDAGModulePage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        C-DAG Module
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Conceptual Debate Argumentation Graph - the core data structure for representing structured arguments.
                    </p>
                </div>

                {/* Components */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Module Components</h2>
                    <ul className="list-disc list-inside space-y-2 text-muted-foreground ml-4">
                        <li><code>graph.py</code> - Main CDAG class and graph operations</li>
                        <li><code>nodes.py</code> - Node types (Proposition, Evidence, Rebuttal, etc.)</li>
                        <li><code>edges.py</code> - Edge types (SUPPORTS, ATTACKS, REBUTS, REFINES)</li>
                        <li><code>propagation.py</code> - Bayesian belief propagation algorithms</li>
                    </ul>
                </section>

                {/* Building Graphs */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Building a C-DAG</h2>
                    <CodeBlock
                        code={`from argus import CDAG, Proposition, Evidence, EdgeType
from argus.cdag.nodes import EvidenceType

# Create graph
graph = CDAG(name="my_debate")

# Add proposition
prop = Proposition(
    text="AI will transform healthcare",
    prior=0.5,
    domain="technology"
)
graph.add_proposition(prop)

# Add supporting evidence
evidence = Evidence(
    text="FDA approved 521 AI medical devices in 2023",
    evidence_type=EvidenceType.EMPIRICAL,
    polarity=1,
    confidence=0.9,
    relevance=0.85
)
graph.add_evidence(evidence, prop.id, EdgeType.SUPPORTS)

# Add challenging evidence
counter = Evidence(
    text="Only 15% showed clinical efficacy",
    evidence_type=EvidenceType.STATISTICAL,
    polarity=-1,
    confidence=0.8,
    relevance=0.9
)
graph.add_evidence(counter, prop.id, EdgeType.ATTACKS)`}
                        language="python"
                    />
                </section>

                {/* Belief Propagation */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Belief Propagation</h2>
                    <CodeBlock
                        code={`from argus.cdag.propagation import (
    compute_posterior,
    propagate_beliefs,
    signed_influence_propagation
)

# Compute posterior for single proposition
posterior = compute_posterior(graph, prop.id)
print(f"Posterior: {posterior:.3f}")

# Propagate through entire graph
posteriors = propagate_beliefs(
    graph,
    max_iterations=10,
    convergence_threshold=0.001
)

# Signed influence propagation
influences = signed_influence_propagation(graph, prop.id)
for node_id, influence in influences.items():
    print(f"{node_id}: {influence:+.3f}")`}
                        language="python"
                    />
                </section>

                {/* Graph Operations */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Graph Operations</h2>
                    <CodeBlock
                        code={`# Query operations
evidence_list = graph.get_evidence(prop.id)
rebuttals = graph.get_rebuttals(evidence.id)
neighbors = graph.get_neighbors(node_id)

# Graph statistics
print(f"Nodes: {len(graph.nodes)}")
print(f"Edges: {len(graph.edges)}")
print(f"Depth: {graph.get_depth()}")

# Serialization
graph.save("debate.json")
loaded = CDAG.load("debate.json")

# Export to NetworkX
import networkx as nx
nx_graph = graph.to_networkx()

# Visualization
graph.visualize("cdag_plot.png")`}
                        language="python"
                    />
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a href="/docs/core-concepts/cdag" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Core Concepts →</h3>
                            <p className="text-sm text-muted-foreground">Learn C-DAG fundamentals</p>
                        </a>
                        <a href="/docs/modules/decision" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Decision →</h3>
                            <p className="text-sm text-muted-foreground">Bayesian algorithms</p>
                        </a>
                        <a href="/docs/modules/outputs" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Outputs →</h3>
                            <p className="text-sm text-muted-foreground">Visualize C-DAGs</p>
                        </a>
                    </div>
                </section>
            </div>
        </div>
    )
}
