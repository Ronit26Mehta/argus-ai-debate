import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'

export const metadata: Metadata = {
    title: 'Evaluation Module | ARGUS Documentation',
    description: 'Metrics and evaluation for debate quality and performance',
}

export default function EvaluationModulePage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Evaluation Module
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Metrics and evaluation for debate quality, evidence strength, and system performance.
                    </p>
                </div>

                {/* Debate Metrics */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Debate Metrics</h2>
                    <CodeBlock
                        code={`from argus.evaluation import DebateEvaluator

evaluator = DebateEvaluator()

# Evaluate debate quality
metrics = evaluator.evaluate(
    graph=cdag,
    proposition_id=prop_id
)

print(f"Evidence Quality: {metrics.evidence_quality:.2f}")
print(f"Argument Diversity: {metrics.diversity:.2f}")
print(f"Convergence: {metrics.convergence:.2f}")
print(f"Debate Depth: {metrics.depth}")`}
                        language="python"
                    />
                </section>

                {/* Evidence Evaluation */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Evidence Evaluation</h2>
                    <CodeBlock
                        code={`from argus.evaluation import evaluate_evidence

# Evaluate single evidence
score = evaluate_evidence(
    evidence=evidence,
    criteria=["relevance", "credibility", "recency"]
)

print(f"Overall Score: {score.overall:.2f}")
print(f"Relevance: {score.relevance:.2f}")
print(f"Credibility: {score.credibility:.2f}")`}
                        language="python"
                    />
                </section>

                {/* Performance Metrics */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Performance Metrics</h2>
                    <CodeBlock
                        code={`from argus.evaluation import PerformanceTracker

tracker = PerformanceTracker()

# Track debate performance
with tracker.track("debate"):
    result = orchestrator.debate(proposition)

print(f"Duration: ${'{'}{tracker.duration:.2f}{'}'} s")
print(f"Tokens: ${'{'}{tracker.total_tokens}{'}'}") 
print(f"Cost: ${'{'}{tracker.total_cost:.4f}{'}'}")`}
                        language="python"
                    />
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a href="/docs/modules/metrics" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Metrics →</h3>
                            <p className="text-sm text-muted-foreground">Detailed metrics</p>
                        </a>
                        <a href="/docs/modules/outputs" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Outputs →</h3>
                            <p className="text-sm text-muted-foreground">Visualizations</p>
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
