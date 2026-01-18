import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'

export const metadata: Metadata = {
    title: 'Outputs Module | ARGUS Documentation',
    description: 'Visualization and reporting for debates, graphs, and metrics',
}

export default function OutputsModulePage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Outputs Module
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Visualization and reporting for debates, C-DAGs, and performance metrics.
                    </p>
                </div>

                {/* Static Plots */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Static Plots</h2>
                    <CodeBlock
                        code={`from argus.outputs import StaticPlotter

plotter = StaticPlotter(theme="dark", style="seaborn")

# Plot C-DAG
plotter.plot_cdag(graph, save_path="cdag.png")

# Plot posterior evolution
plotter.plot_posterior_evolution(
    history=debate_history,
    save_path="posterior.png"
)

# Plot evidence distribution
plotter.plot_evidence_distribution(
    graph,
    save_path="evidence.png"
)`}
                        language="python"
                    />
                </section>

                {/* Interactive Plots */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Interactive Plots</h2>
                    <CodeBlock
                        code={`from argus.outputs import InteractivePlotter

plotter = InteractivePlotter()

# Interactive network
plotter.plot_interactive_network(
    graph,
    save_path="network.html"
)

# Interactive timeline
plotter.plot_debate_timeline(
    debate_result,
    save_path="timeline.html"
)`}
                        language="python"
                    />
                </section>

                {/* Reports */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Reports</h2>
                    <CodeBlock
                        code={`from argus.outputs import ReportGenerator

generator = ReportGenerator()

# Generate HTML report
report = generator.generate_report(
    debate_result,
    format="html",
    save_path="report.html"
)

# Generate PDF
report = generator.generate_report(
    debate_result,
    format="pdf",
    save_path="report.pdf"
)`}
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
                        <a href="/docs/modules/metrics" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Metrics →</h3>
                            <p className="text-sm text-muted-foreground">Performance metrics</p>
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
