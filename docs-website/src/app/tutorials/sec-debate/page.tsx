import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'
import { Callout } from '@/components/Callout'

export const metadata: Metadata = {
    title: 'SEC Filing Analysis Tutorial | ARGUS',
    description: 'Analyze SEC filings with multi-specialist debates and generate visualizations',
}

export default function SECDebateTutorial() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        SEC Filing Analysis
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Analyze SEC filings with multi-specialist debates and generate comprehensive visualizations.
                    </p>
                    <div className="flex gap-4 mt-4 text-sm text-muted-foreground">
                        <span>⏱️ 40 minutes</span>
                        <span>📊 Advanced</span>
                        <span>💼 Finance</span>
                    </div>
                </div>

                {/* Load SEC Data */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Step 1: Load SEC Filings</h2>
                    <CodeBlock
                        code={`from argus.knowledge import DocumentLoader, Chunker
from argus.retrieval import HybridRetriever

# Load SEC 10-K filing
loader = DocumentLoader()
filing = loader.load("company_10k.pdf")

# Chunk document
chunker = Chunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk(filing)

# Create retriever
retriever = HybridRetriever(
    embedding_model="all-MiniLM-L6-v2",
    use_reranker=True
)
retriever.index_chunks(chunks)

print(f"Indexed {len(chunks)} chunks from SEC filing")`}
                        language="python"
                    />
                </section>

                {/* Multi-Specialist Debate */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Step 2: Multi-Specialist Analysis</h2>
                    <CodeBlock
                        code={`from argus import get_llm
from argus.agents import Specialist

# Create domain specialists
financial_analyst = Specialist(
    get_llm("openai", model="gpt-4o"),
    domain="financial_analysis"
)

risk_analyst = Specialist(
    get_llm("anthropic", model="claude-3-5-sonnet-20241022"),
    domain="risk_assessment"
)

compliance_expert = Specialist(
    get_llm("gemini", model="gemini-1.5-pro"),
    domain="regulatory_compliance"
)

# Analyze different aspects
claims = [
    "The company's financial position is strong",
    "Risk management practices are adequate",
    "Regulatory compliance is satisfactory"
]`}
                        language="python"
                    />
                </section>

                {/* Run Analysis */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Step 3: Run Comprehensive Analysis</h2>
                    <CodeBlock
                        code={`from argus import RDCOrchestrator

results = {}

for claim in claims:
    # Determine specialist
    if "financial" in claim.lower():
        specialist = financial_analyst
    elif "risk" in claim.lower():
        specialist = risk_analyst
    else:
        specialist = compliance_expert
    
    # Run debate
    orchestrator = RDCOrchestrator(
        llm=specialist.llm,
        retriever=retriever,
        max_rounds=3,
        enable_provenance=True
    )
    
    result = orchestrator.debate(claim, prior=0.5, domain="finance")
    results[claim] = result
    
    print(f"\\n{claim}")
    print(f"Verdict: {result.verdict.label}")
    print(f"Posterior: {result.verdict.posterior:.3f}")`}
                        language="python"
                    />
                </section>

                {/* Visualize */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Step 4: Generate Visualizations</h2>
                    <CodeBlock
                        code={`from argus.outputs import StaticPlotter, InteractivePlotter, ReportGenerator

# Static plots
plotter = StaticPlotter(theme="professional")

for claim, result in results.items():
    safe_name = claim.replace(" ", "_")[:30]
    plotter.plot_cdag(result.graph, save_path=f"{safe_name}_cdag.png")
    plotter.plot_evidence_distribution(
        result.graph,
        save_path=f"{safe_name}_evidence.png"
    )

# Interactive dashboard
interactive = InteractivePlotter()
interactive.plot_interactive_network(
    results[claims[0]].graph,
    save_path="sec_analysis_network.html"
)

# Generate report
generator = ReportGenerator()
report = generator.generate_report(
    results,
    format="html",
    save_path="sec_analysis_report.html"
)

print("Visualizations and report generated!")`}
                        language="python"
                    />
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-2">
                        <a href="/docs/modules/outputs" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Outputs Module →</h3>
                            <p className="text-sm text-muted-foreground">More visualizations</p>
                        </a>
                        <a href="/docs/tools" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Tools →</h3>
                            <p className="text-sm text-muted-foreground">Finance tools</p>
                        </a>
                    </div>
                </section>
            </div>
        </div>
    )
}
