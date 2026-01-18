import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'

export const metadata: Metadata = {
    title: 'Metrics Module | ARGUS Documentation',
    description: 'Performance metrics and monitoring',
}

export default function MetricsModulePage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Metrics Module
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Performance metrics, monitoring, and observability for debates.
                    </p>
                </div>

                {/* Performance Tracking */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Performance Tracking</h2>
                    <CodeBlock
                        code={`from argus.metrics import MetricsCollector

collector = MetricsCollector()

# Track debate
with collector.track("debate"):
    result = orchestrator.debate("proposition")

# Get metrics
metrics = collector.get_metrics("debate")
print(f"Duration: ${'{'}{metrics.duration:.2f}{'}'} s")
print(f"Tokens: ${'{'}{metrics.total_tokens}{'}'}")
print(f"Cost: ${'{'}{metrics.total_cost:.4f}{'}'}")
print(f"Latency: ${'{'}{metrics.avg_latency:.2f}{'}'} ms")`}
                        language="python"
                    />
                </section>

                {/* Export Metrics */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Export Metrics</h2>
                    <CodeBlock
                        code={`# Export to JSON
collector.export("metrics.json", format="json")

# Export to Prometheus
collector.export("metrics.prom", format="prometheus")

# Export to CSV
collector.export("metrics.csv", format="csv")`}
                        language="python"
                    />
                </section>

                {/* Real-time Monitoring */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Real-time Monitoring</h2>
                    <CodeBlock
                        code={`from argus.metrics import MetricsServer

# Start metrics server
server = MetricsServer(port=9090)
server.start()

# Metrics available at http://localhost:9090/metrics`}
                        language="python"
                    />
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a href="/docs/modules/evaluation" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Evaluation →</h3>
                            <p className="text-sm text-muted-foreground">Debate quality metrics</p>
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
