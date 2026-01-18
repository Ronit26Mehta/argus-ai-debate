import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'

export const metadata: Metadata = {
    title: 'Orchestrator Module | ARGUS Documentation',
    description: 'RDC orchestration and multi-round debate management',
}

export default function OrchestratorModulePage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Orchestrator Module
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Research Debate Chain (RDC) orchestration and multi-round debate management.
                    </p>
                </div>

                {/* RDC Orchestrator */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">RDC Orchestrator</h2>
                    <CodeBlock
                        code={`from argus import RDCOrchestrator, get_llm

llm = get_llm("openai", model="gpt-4o")

orchestrator = RDCOrchestrator(
    llm=llm,
    max_rounds=5,
    min_evidence=3,
    convergence_threshold=0.01,
    budget=10000,
    enable_provenance=True
)

# Run debate
result = orchestrator.debate(
    "The treatment is effective",
    prior=0.5,
    domain="clinical"
)

print(f"Verdict: {result.verdict.label}")
print(f"Posterior: {result.verdict.posterior:.3f}")
print(f"Rounds: {result.num_rounds}")
print(f"Evidence: {result.num_evidence}")`}
                        language="python"
                    />
                </section>

                {/* With Retrieval */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">With Retrieval</h2>
                    <CodeBlock
                        code={`from argus.retrieval import HybridRetriever

# Create retriever
retriever = HybridRetriever(
    embedding_model="all-MiniLM-L6-v2",
    use_reranker=True
)
retriever.index_chunks(chunks)

# Orchestrator with retrieval
orchestrator = RDCOrchestrator(
    llm=llm,
    retriever=retriever,
    max_rounds=5
)

result = orchestrator.debate("proposition")`}
                        language="python"
                    />
                </section>

                {/* Custom Agents */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Custom Agent Configuration</h2>
                    <CodeBlock
                        code={`from argus.agents import Moderator, Specialist, Refuter, Jury

# Different LLMs for different roles
orchestrator = RDCOrchestrator(
    moderator=Moderator(get_llm("openai", model="gpt-4o")),
    specialist=Specialist(get_llm("anthropic", model="claude-3-5-sonnet-20241022")),
    refuter=Refuter(get_llm("groq", model="llama-3.1-70b-versatile")),
    jury=Jury(get_llm("gemini", model="gemini-1.5-pro")),
    max_rounds=3
)

result = orchestrator.debate("proposition")`}
                        language="python"
                    />
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a href="/docs/core-concepts/rdc" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">RDC Concept →</h3>
                            <p className="text-sm text-muted-foreground">Learn RDC</p>
                        </a>
                        <a href="/docs/modules/agents" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Agents →</h3>
                            <p className="text-sm text-muted-foreground">Multi-agent system</p>
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
