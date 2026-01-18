import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'
import { Callout } from '@/components/Callout'

export const metadata: Metadata = {
    title: 'Research Claim Verification Tutorial | ARGUS',
    description: 'Verify scientific claims by fetching papers from arXiv and running structured debates',
}

export default function ResearchVerificationTutorial() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Research Claim Verification
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Verify scientific claims by fetching papers from arXiv and running structured debates.
                    </p>
                    <div className="flex gap-4 mt-4 text-sm text-muted-foreground">
                        <span>⏱️ 25 minutes</span>
                        <span>📊 Intermediate</span>
                        <span>🔬 Research</span>
                    </div>
                </div>

                {/* Overview */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">What You'll Build</h2>
                    <p className="text-muted-foreground">
                        A system that automatically verifies scientific claims by:
                    </p>
                    <ul className="list-disc list-inside space-y-2 text-muted-foreground ml-4">
                        <li>Fetching relevant papers from arXiv</li>
                        <li>Extracting evidence for and against the claim</li>
                        <li>Running multi-agent debates</li>
                        <li>Generating a calibrated verdict with provenance</li>
                    </ul>
                </section>

                {/* Complete Code */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Complete Implementation</h2>
                    <CodeBlock
                        code={`from argus import RDCOrchestrator, get_llm
from argus.knowledge.connectors import ArxivConnector
from argus.knowledge import Chunker
from argus.retrieval import HybridRetriever

# 1. Fetch papers
connector = ArxivConnector()
papers = connector.fetch(
    query="transformer attention mechanism",
    max_results=15
)

# 2. Process documents
chunker = Chunker(chunk_size=512, chunk_overlap=50)
chunks = []
for paper in papers:
    chunks.extend(chunker.chunk(paper))

# 3. Create retriever
retriever = HybridRetriever(
    embedding_model="all-MiniLM-L6-v2",
    use_reranker=True
)
retriever.index_chunks(chunks)

# 4. Run debate
llm = get_llm("openai", model="gpt-4o")
orchestrator = RDCOrchestrator(
    llm=llm,
    retriever=retriever,
    max_rounds=5,
    enable_provenance=True
)

result = orchestrator.debate(
    "Transformers are more efficient than RNNs for long sequences",
    prior=0.5,
    domain="research"
)

# 5. Results
print(f"Verdict: {result.verdict.label}")
print(f"Posterior: {result.verdict.posterior:.3f}")
print(f"Evidence: {result.num_evidence} items")

# View top evidence
for evidence in result.evidence[:3]:
    print(f"\\n[{evidence.polarity:+d}] {evidence.text}")
    print(f"Source: {evidence.metadata.get('title', 'Unknown')}")`}
                        language="python"
                        filename="research_verification.py"
                    />
                </section>

                {/* Advanced */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Advanced: Multi-Model Setup</h2>
                    <CodeBlock
                        code={`from argus.agents import Moderator, Specialist, Refuter, Jury

# Use different models for different roles
orchestrator = RDCOrchestrator(
    moderator=Moderator(get_llm("openai", model="gpt-4o")),
    specialist=Specialist(get_llm("anthropic", model="claude-3-5-sonnet-20241022")),
    refuter=Refuter(get_llm("groq", model="llama-3.1-70b-versatile")),
    jury=Jury(get_llm("gemini", model="gemini-1.5-pro")),
    retriever=retriever,
    max_rounds=3
)

result = orchestrator.debate("claim")`}
                        language="python"
                    />
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-2">
                        <a href="/tutorials/custom-agent-pipeline" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Custom Agents →</h3>
                            <p className="text-sm text-muted-foreground">Build custom pipelines</p>
                        </a>
                        <a href="/docs/connectors" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Connectors →</h3>
                            <p className="text-sm text-muted-foreground">More data sources</p>
                        </a>
                    </div>
                </section>
            </div>
        </div>
    )
}
