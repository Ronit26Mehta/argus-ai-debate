import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'
import { Callout } from '@/components/Callout'

export const metadata: Metadata = {
    title: 'Clinical Evidence Evaluation Tutorial | ARGUS',
    description: 'Evaluate medical treatment claims using multi-agent debates with clinical literature',
}

export default function ClinicalEvidenceTutorial() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Clinical Evidence Evaluation
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Learn how to evaluate medical treatment claims using multi-agent debates with clinical literature.
                    </p>
                    <div className="flex gap-4 mt-4 text-sm text-muted-foreground">
                        <span>⏱️ 30 minutes</span>
                        <span>📊 Intermediate</span>
                        <span>🏥 Medical</span>
                    </div>
                </div>

                {/* Prerequisites */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Prerequisites</h2>
                    <Callout variant="info">
                        <ul className="list-disc list-inside space-y-1">
                            <li>ARGUS installed with medical tools</li>
                            <li>API key for LLM provider (or Ollama)</li>
                            <li>Basic understanding of RDC and C-DAG</li>
                        </ul>
                    </Callout>
                </section>

                {/* Step 1 */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Step 1: Setup</h2>
                    <CodeBlock
                        code={`from argus import RDCOrchestrator, get_llm
from argus.knowledge import DocumentLoader, Chunker
from argus.retrieval import HybridRetriever
from argus.knowledge.connectors import ArxivConnector

# Initialize LLM
llm = get_llm("openai", model="gpt-4o")

# Fetch clinical papers
connector = ArxivConnector()
papers = connector.fetch(
    query="randomized controlled trial treatment efficacy",
    max_results=10
)

print(f"Fetched {len(papers)} clinical papers")`}
                        language="python"
                        filename="clinical_eval.py"
                    />
                </section>

                {/* Step 2 */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Step 2: Build Knowledge Base</h2>
                    <CodeBlock
                        code={`# Chunk documents
chunker = Chunker(chunk_size=512, chunk_overlap=50)
chunks = []
for paper in papers:
    chunks.extend(chunker.chunk(paper))

# Create retriever
retriever = HybridRetriever(
    embedding_model="all-MiniLM-L6-v2",
    use_reranker=True
)
retriever.index_chunks(chunks)

print(f"Indexed {len(chunks)} chunks")`}
                        language="python"
                    />
                </section>

                {/* Step 3 */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Step 3: Run Debate</h2>
                    <CodeBlock
                        code={`# Create orchestrator
orchestrator = RDCOrchestrator(
    llm=llm,
    retriever=retriever,
    max_rounds=5,
    enable_provenance=True
)

# Evaluate treatment claim
result = orchestrator.debate(
    "The new treatment reduces symptoms by more than 20%",
    prior=0.5,
    domain="clinical"
)

print(f"Verdict: {result.verdict.label}")
print(f"Posterior: {result.verdict.posterior:.3f}")
print(f"Confidence: {result.verdict.confidence:.3f}")`}
                        language="python"
                    />
                </section>

                {/* Step 4 */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Step 4: Analyze Results</h2>
                    <CodeBlock
                        code={`# View evidence
print(f"\\nEvidence ({result.num_evidence} items):")
for evidence in result.evidence[:5]:
    print(f"[{evidence.polarity:+d}] {evidence.text[:100]}...")
    print(f"    Confidence: {evidence.confidence:.2f}")
    print(f"    Source: {evidence.source}")

# Verify provenance
if result.provenance:
    is_valid, errors = result.provenance.verify_integrity()
    print(f"\\nProvenance valid: {is_valid}")`}
                        language="python"
                    />
                </section>

                {/* Step 5 */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Step 5: Visualize</h2>
                    <CodeBlock
                        code={`from argus.outputs import StaticPlotter, InteractivePlotter

# Static plots
plotter = StaticPlotter(theme="dark")
plotter.plot_cdag(result.graph, save_path="clinical_cdag.png")
plotter.plot_evidence_distribution(result.graph, save_path="evidence.png")

# Interactive
interactive = InteractivePlotter()
interactive.plot_interactive_network(result.graph, save_path="network.html")`}
                        language="python"
                    />
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-2">
                        <a href="/tutorials/research-verification" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Research Verification →</h3>
                            <p className="text-sm text-muted-foreground">Verify scientific claims</p>
                        </a>
                        <a href="/docs/modules/orchestrator" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Orchestrator Docs →</h3>
                            <p className="text-sm text-muted-foreground">Learn more about RDC</p>
                        </a>
                    </div>
                </section>
            </div>
        </div>
    )
}
