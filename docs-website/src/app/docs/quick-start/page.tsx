import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'
import { Callout } from '@/components/Callout'

export const metadata: Metadata = {
    title: 'Quick Start | ARGUS Documentation',
    description: 'Get started with ARGUS in 5 minutes - your first multi-agent debate',
}

export default function QuickStartPage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Quick Start
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Get up and running with ARGUS in 5 minutes. Run your first multi-agent debate.
                    </p>
                </div>

                {/* Prerequisites */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Prerequisites</h2>
                    <Callout variant="info">
                        Make sure you have ARGUS installed. If not, see the{' '}
                        <a href="/docs/installation" className="text-primary hover:underline">
                            installation guide
                        </a>.
                    </Callout>
                </section>

                {/* Basic Debate */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Your First Debate</h2>
                    <p className="text-muted-foreground">
                        Let's run a simple debate to evaluate a proposition:
                    </p>

                    <CodeBlock
                        code={`from argus import RDCOrchestrator, get_llm

# Initialize with any supported LLM
llm = get_llm("openai", model="gpt-4o")

# Create orchestrator
orchestrator = RDCOrchestrator(llm=llm, max_rounds=3)

# Run debate
result = orchestrator.debate(
    "The new treatment reduces symptoms by more than 20%",
    prior=0.5,  # Start with 50/50 uncertainty
)

# View results
print(f"Verdict: {result.verdict.label}")
print(f"Posterior: {result.verdict.posterior:.3f}")
print(f"Evidence: {result.num_evidence} items")
print(f"Reasoning: {result.verdict.reasoning}")`}
                        language="python"
                        filename="first_debate.py"
                    />

                    <Callout variant="tip" title="API Key Required">
                        Set your OpenAI API key: <code>export OPENAI_API_KEY="sk-..."</code>
                        <br />
                        Or use a local model like Ollama (no API key needed).
                    </Callout>
                </section>

                {/* Using Local LLMs */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Using Local LLMs (Free)</h2>
                    <p className="text-muted-foreground">
                        No API key? Use Ollama for completely free local inference:
                    </p>

                    <CodeBlock
                        code={`from argus import RDCOrchestrator, get_llm

# Use local Ollama (free, no API key)
llm = get_llm("ollama", model="llama3.2")

orchestrator = RDCOrchestrator(llm=llm, max_rounds=3)
result = orchestrator.debate("AI will transform healthcare")`}
                        language="python"
                    />

                    <Callout variant="note">
                        Install Ollama from{' '}
                        <a href="https://ollama.ai" target="_blank" rel="noopener" className="text-primary hover:underline">
                            ollama.ai
                        </a>{' '}
                        and run <code>ollama pull llama3.2</code>
                    </Callout>
                </section>

                {/* Document Ingestion */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Document Ingestion & Retrieval</h2>
                    <p className="text-muted-foreground">
                        Add documents to your knowledge base for evidence-based debates:
                    </p>

                    <CodeBlock
                        code={`from argus import DocumentLoader, Chunker, HybridRetriever
from argus import RDCOrchestrator, get_llm

# Load documents
loader = DocumentLoader()
doc = loader.load("research_paper.pdf")

# Chunk with overlap
chunker = Chunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk(doc)

# Create hybrid retriever
retriever = HybridRetriever(
    embedding_model="all-MiniLM-L6-v2",  # Free local embeddings
    use_reranker=True,
)
retriever.index_chunks(chunks)

# Run debate with retrieval
llm = get_llm("openai", model="gpt-4o")
orchestrator = RDCOrchestrator(llm=llm, retriever=retriever)

result = orchestrator.debate(
    "The study's methodology is sound",
    domain="research",
)`}
                        language="python"
                        filename="with_retrieval.py"
                    />
                </section>

                {/* Multi-Agent Debate */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Multi-Agent Debate</h2>
                    <p className="text-muted-foreground">
                        Use different agents for specialized roles:
                    </p>

                    <CodeBlock
                        code={`from argus import get_llm, CDAG, Proposition
from argus.agents import Moderator, Specialist, Refuter, Jury

# Initialize agents with different LLMs
moderator_llm = get_llm("openai", model="gpt-4o")
specialist_llm = get_llm("anthropic", model="claude-3-5-sonnet-20241022")
refuter_llm = get_llm("groq", model="llama-3.1-70b-versatile")
jury_llm = get_llm("gemini", model="gemini-1.5-pro")

# Create agents
moderator = Moderator(moderator_llm)
specialist = Specialist(specialist_llm, domain="policy")
refuter = Refuter(refuter_llm)
jury = Jury(jury_llm)

# Create debate graph
graph = CDAG()
prop = Proposition(
    text="Carbon pricing is effective for reducing emissions",
    prior=0.5,
)
graph.add_proposition(prop)

# Run debate rounds
for round_num in range(3):
    # Gather evidence
    evidence = specialist.gather_evidence(graph, prop.id)
    
    # Generate rebuttals
    rebuttals = refuter.generate_rebuttals(graph, prop.id)
    
    # Check stopping criteria
    if moderator.should_stop(graph, prop.id):
        break

# Render verdict
verdict = jury.evaluate(graph, prop.id)
print(f"Verdict: {verdict.label}")
print(f"Posterior: {verdict.posterior:.3f}")`}
                        language="python"
                        filename="multi_agent.py"
                    />

                    <Callout variant="success" title="Mix and Match">
                        Use different LLMs for different agents based on their strengths:
                        <ul className="list-disc list-inside mt-2 space-y-1">
                            <li>GPT-4 for moderation and planning</li>
                            <li>Claude for deep analysis</li>
                            <li>Groq for fast rebuttals</li>
                            <li>Gemini for final verdicts</li>
                        </ul>
                    </Callout>
                </section>

                {/* Building C-DAG Manually */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Building a C-DAG Manually</h2>
                    <p className="text-muted-foreground">
                        For fine-grained control, build the debate graph yourself:
                    </p>

                    <CodeBlock
                        code={`from argus import CDAG, Proposition, Evidence, EdgeType
from argus.cdag.nodes import EvidenceType
from argus.cdag.propagation import compute_posterior

# Create graph
graph = CDAG(name="drug_efficacy")

# Add proposition
prop = Proposition(
    text="Drug X is effective for treating condition Y",
    prior=0.5,
    domain="clinical",
)
graph.add_proposition(prop)

# Add supporting evidence
trial = Evidence(
    text="Phase 3 RCT showed 35% symptom reduction (n=500, p<0.001)",
    evidence_type=EvidenceType.EMPIRICAL,
    polarity=1,  # Supports
    confidence=0.9,
    relevance=0.95,
)
graph.add_evidence(trial, prop.id, EdgeType.SUPPORTS)

# Add challenging evidence
side_effect = Evidence(
    text="15% of patients experienced adverse events",
    evidence_type=EvidenceType.EMPIRICAL,
    polarity=-1,  # Attacks
    confidence=0.8,
    relevance=0.7,
)
graph.add_evidence(side_effect, prop.id, EdgeType.ATTACKS)

# Compute Bayesian posterior
posterior = compute_posterior(graph, prop.id)
print(f"Posterior probability: {posterior:.3f}")`}
                        language="python"
                        filename="manual_cdag.py"
                    />
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a
                            href="/docs/core-concepts/rdc"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">Learn RDC →</h3>
                            <p className="text-sm text-muted-foreground">
                                Understand the Research Debate Chain framework
                            </p>
                        </a>
                        <a
                            href="/docs/llm-providers"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">LLM Providers →</h3>
                            <p className="text-sm text-muted-foreground">
                                Explore 27+ supported LLM providers
                            </p>
                        </a>
                        <a
                            href="/tutorials"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">Tutorials →</h3>
                            <p className="text-sm text-muted-foreground">
                                Step-by-step guides for real-world use cases
                            </p>
                        </a>
                    </div>
                </section>

                {/* Common Patterns */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Common Patterns</h2>

                    <div className="space-y-6">
                        <div>
                            <h3 className="text-xl font-semibold mb-3">Streaming Responses</h3>
                            <CodeBlock
                                code={`llm = get_llm("openai", model="gpt-4o")

for chunk in llm.stream("Tell me about quantum computing"):
    print(chunk, end="", flush=True)`}
                                language="python"
                            />
                        </div>

                        <div>
                            <h3 className="text-xl font-semibold mb-3">Using Tools</h3>
                            <CodeBlock
                                code={`from argus.tools.integrations import DuckDuckGoTool, WikipediaTool

# Web search
search = DuckDuckGoTool()
results = search(query="latest AI research 2024", max_results=5)

# Wikipedia lookup
wiki = WikipediaTool()
summary = wiki(query="Machine Learning", action="summary")`}
                                language="python"
                            />
                        </div>

                        <div>
                            <h3 className="text-xl font-semibold mb-3">Embeddings</h3>
                            <CodeBlock
                                code={`from argus.embeddings import get_embedding

# Local (free)
embedder = get_embedding("sentence_transformers", model="all-MiniLM-L6-v2")
vectors = embedder.embed_documents(["Hello", "World"])

# Cloud
embedder = get_embedding("openai", model="text-embedding-3-small")
query_vec = embedder.embed_query("What is AI?")`}
                                language="python"
                            />
                        </div>
                    </div>
                </section>
            </div>
        </div>
    )
}
