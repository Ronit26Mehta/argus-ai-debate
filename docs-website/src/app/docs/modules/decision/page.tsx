import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'
import { Callout } from '@/components/Callout'

export const metadata: Metadata = {
    title: 'Decision Module | ARGUS Documentation',
    description: 'Bayesian updating, value of information, and decision-theoretic planning',
}

export default function DecisionModulePage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Decision Module
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Bayesian updating, value of information (VoI), and decision-theoretic planning.
                    </p>
                </div>

                {/* Bayesian Updating */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Bayesian Updating</h2>
                    <CodeBlock
                        code={`from argus.decision import bayesian_update, log_odds_update

# Simple Bayesian update
prior = 0.5
likelihood_ratio = 2.5
posterior = bayesian_update(prior, likelihood_ratio)
print(f"Posterior: {posterior:.3f}")

# Log-odds space (numerically stable)
posterior = log_odds_update(
    prior_odds=1.0,
    likelihood_ratio=2.5,
    confidence=0.8
)
print(f"Posterior: {posterior:.3f}")`}
                        language="python"
                    />
                </section>

                {/* Value of Information */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Value of Information (VoI)</h2>
                    <CodeBlock
                        code={`from argus.decision import compute_voi, expected_information_gain

# Compute VoI for potential evidence
voi = compute_voi(
    prior=0.5,
    cost=10.0,
    benefit_if_true=100.0,
    benefit_if_false=50.0,
    evidence_reliability=0.8
)
print(f"Value of Information: ${'{'}{voi:.2f}{'}'}")

# Expected information gain
eig = expected_information_gain(
    prior=0.5,
    likelihood_positive=0.9,
    likelihood_negative=0.1
)
print(f"Expected Info Gain: ${'{'}{eig:.3f}{'}'} bits")`}
                        language="python"
                    />
                </section>

                {/* Decision Planning */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Decision-Theoretic Planning</h2>
                    <CodeBlock
                        code={`from argus.decision import DecisionPlanner

planner = DecisionPlanner(
    budget=100.0,
    max_queries=10
)

# Plan information gathering
plan = planner.plan(
    graph=cdag,
    proposition_id=prop_id,
    objective="maximize_posterior_confidence"
)

for action in plan.actions:
    print(f"Action: {action.type}")
    print(f"Expected VoI: {action.expected_voi:.2f}")
    print(f"Cost: {action.cost:.2f}")`}
                        language="python"
                    />
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a href="/docs/modules/cdag" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">C-DAG →</h3>
                            <p className="text-sm text-muted-foreground">Graph structure</p>
                        </a>
                        <a href="/docs/modules/orchestrator" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Orchestrator →</h3>
                            <p className="text-sm text-muted-foreground">Debate orchestration</p>
                        </a>
                        <a href="/tutorials" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Tutorials →</h3>
                            <p className="text-sm text-muted-foreground">Practical examples</p>
                        </a>
                    </div>
                </section>
            </div>
        </div>
    )
}
