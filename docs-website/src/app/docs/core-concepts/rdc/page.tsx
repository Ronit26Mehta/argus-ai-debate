import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'
import { Callout } from '@/components/Callout'
import { InteractiveDiagram, FlowDiagram } from '@/components/InteractiveDiagram'

export const metadata: Metadata = {
    title: 'Research Debate Chain (RDC) | ARGUS Documentation',
    description: 'Learn about the Research Debate Chain framework - the core of ARGUS multi-agent reasoning',
}

export default function RDCPage() {
    const debateFlow = {
        nodes: [
            { id: '1', label: 'Proposition', type: 'start' as const },
            { id: '2', label: 'Moderator Creates Agenda', type: 'process' as const },
            { id: '3', label: 'Specialists Gather Evidence', type: 'process' as const },
            { id: '4', label: 'Refuters Challenge Evidence', type: 'process' as const },
            { id: '5', label: 'Convergence Check', type: 'decision' as const },
            { id: '6', label: 'Jury Renders Verdict', type: 'end' as const },
        ],
        edges: [],
    }

    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Research Debate Chain (RDC)
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        A novel approach to AI reasoning that structures knowledge evaluation as multi-agent debates.
                    </p>
                </div>

                {/* Overview */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Overview</h2>
                    <p className="text-muted-foreground">
                        ARGUS implements <strong>Research Debate Chain (RDC)</strong> - instead of single-pass inference,
                        ARGUS orchestrates specialist agents that gather evidence, generate rebuttals, and render verdicts
                        through Bayesian aggregation.
                    </p>

                    <Callout variant="info" title="Why RDC?">
                        Traditional LLM applications suffer from hallucination, overconfidence, opacity, and single-point failure.
                        RDC addresses these through adversarial debate, Bayesian aggregation, full provenance, and multi-model support.
                    </Callout>
                </section>

                {/* Debate Flow */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Debate Flow</h2>
                    <FlowDiagram nodes={debateFlow.nodes} edges={debateFlow.edges} title="RDC Debate Process" />
                </section>

                {/* Key Components */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Key Components</h2>

                    <div className="space-y-6">
                        <div className="p-6 rounded-xl border bg-card">
                            <h3 className="text-xl font-semibold mb-3">1. Moderator (Orchestration)</h3>
                            <p className="text-muted-foreground mb-3">
                                Creates debate agendas, manages rounds, evaluates stopping criteria, and breaks ties.
                            </p>
                            <CodeBlock
                                code={`from argus.agents import Moderator

moderator = Moderator(llm)
agenda = moderator.create_agenda(graph, proposition_id)
should_stop = moderator.should_stop(graph, proposition_id)`}
                                language="python"
                            />
                        </div>

                        <div className="p-6 rounded-xl border bg-card">
                            <h3 className="text-xl font-semibold mb-3">2. Specialist (Evidence Gathering)</h3>
                            <p className="text-muted-foreground mb-3">
                                Domain-specific research, hybrid retrieval, and source quality assessment.
                            </p>
                            <CodeBlock
                                code={`from argus.agents import Specialist

specialist = Specialist(llm, domain="clinical")
evidence = specialist.gather_evidence(graph, proposition_id)`}
                                language="python"
                            />
                        </div>

                        <div className="p-6 rounded-xl border bg-card">
                            <h3 className="text-xl font-semibold mb-3">3. Refuter (Challenge Generation)</h3>
                            <p className="text-muted-foreground mb-3">
                                Counter-evidence, methodological critiques, and logical fallacy detection.
                            </p>
                            <CodeBlock
                                code={`from argus.agents import Refuter

refuter = Refuter(llm)
rebuttals = refuter.generate_rebuttals(graph, proposition_id)`}
                                language="python"
                            />
                        </div>

                        <div className="p-6 rounded-xl border bg-card">
                            <h3 className="text-xl font-semibold mb-3">4. Jury (Verdict Rendering)</h3>
                            <p className="text-muted-foreground mb-3">
                                Bayesian aggregation, confidence calibration, and label assignment.
                            </p>
                            <CodeBlock
                                code={`from argus.agents import Jury

jury = Jury(llm)
verdict = jury.evaluate(graph, proposition_id)
print(f"Verdict: {verdict.label}")
print(f"Posterior: {verdict.posterior:.3f}")`}
                                language="python"
                            />
                        </div>
                    </div>
                </section>

                {/* EDDO Algorithm */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Evidence-Directed Debate Orchestration (EDDO)</h2>
                    <p className="text-muted-foreground">
                        EDDO manages multi-round debates with configurable stopping criteria:
                    </p>

                    <div className="grid gap-4 md:grid-cols-2">
                        <div className="p-4 rounded-lg border bg-muted/30">
                            <h4 className="font-semibold mb-2">Convergence Detection</h4>
                            <p className="text-sm text-muted-foreground">
                                Stop when posterior probability stabilizes (changes \u003c threshold)
                            </p>
                        </div>
                        <div className="p-4 rounded-lg border bg-muted/30">
                            <h4 className="font-semibold mb-2">Maximum Rounds</h4>
                            <p className="text-sm text-muted-foreground">
                                Enforce hard limit on debate rounds
                            </p>
                        </div>
                        <div className="p-4 rounded-lg border bg-muted/30">
                            <h4 className="font-semibold mb-2">Budget-Based Termination</h4>
                            <p className="text-sm text-muted-foreground">
                                Stop when token budget is exhausted
                            </p>
                        </div>
                        <div className="p-4 rounded-lg border bg-muted/30">
                            <h4 className="font-semibold mb-2">Information Gain Threshold</h4>
                            <p className="text-sm text-muted-foreground">
                                Stop when new evidence provides minimal information
                            </p>
                        </div>
                    </div>

                    <CodeBlock
                        code={`from argus import RDCOrchestrator, get_llm

llm = get_llm("openai", model="gpt-4o")

orchestrator = RDCOrchestrator(
    llm=llm,
    max_rounds=5,              # Maximum rounds
    min_evidence=3,            # Minimum evidence per side
    convergence_threshold=0.01, # Stop if Δposterior < 1%
    budget=10000,              # Token budget
)

result = orchestrator.debate(
    "The treatment is effective",
    prior=0.5,
)`}
                        language="python"
                    />
                </section>

                {/* Advantages */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Advantages of RDC</h2>

                    <div className="space-y-4">
                        <Callout variant="success" title="Reduced Hallucination">
                            Multiple agents challenge claims with evidence, reducing the likelihood of unchecked hallucinations.
                        </Callout>

                        <Callout variant="success" title="Calibrated Confidence">
                            Bayesian aggregation provides calibrated probability estimates, not just binary predictions.
                        </Callout>

                        <Callout variant="success" title="Full Audit Trail">
                            Every claim is traced to its source with complete provenance tracking.
                        </Callout>

                        <Callout variant="success" title="Multi-Model Flexibility">
                            Use different LLMs for different roles based on their strengths.
                        </Callout>
                    </div>
                </section>

                {/* Example */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Complete Example</h2>
                    <p className="text-muted-foreground">
                        Here's a full RDC workflow:
                    </p>

                    <CodeBlock
                        code={`from argus import RDCOrchestrator, get_llm
from argus.retrieval import HybridRetriever
from argus.knowledge import DocumentLoader, Chunker

# Load documents
loader = DocumentLoader()
docs = loader.load_directory("./research_papers/")

# Create retriever
chunker = Chunker(chunk_size=512, chunk_overlap=50)
chunks = []
for doc in docs:
    chunks.extend(chunker.chunk(doc))

retriever = HybridRetriever(
    embedding_model="all-MiniLM-L6-v2",
    use_reranker=True,
)
retriever.index_chunks(chunks)

# Initialize orchestrator
llm = get_llm("openai", model="gpt-4o")
orchestrator = RDCOrchestrator(
    llm=llm,
    retriever=retriever,
    max_rounds=5,
    convergence_threshold=0.01,
    enable_provenance=True,
)

# Run debate
result = orchestrator.debate(
    "The proposed methodology is scientifically sound",
    prior=0.5,
    domain="research",
)

# Analyze results
print(f"Verdict: {result.verdict.label}")
print(f"Posterior: {result.verdict.posterior:.3f}")
print(f"Confidence: {result.verdict.confidence:.3f}")
print(f"Rounds: {result.num_rounds}")
print(f"Evidence: {result.num_evidence} items")

# View evidence
for evidence in result.evidence[:5]:
    print(f"\\n[{evidence.polarity:+d}] {evidence.text[:100]}...")
    print(f"   Confidence: {evidence.confidence:.2f}")
    print(f"   Source: {evidence.source}")

# Check provenance
if result.provenance:
    is_valid, errors = result.provenance.verify_integrity()
    print(f"\\nProvenance valid: {is_valid}")`}
                        language="python"
                        filename="complete_rdc_example.py"
                    />
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a
                            href="/docs/core-concepts/cdag"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">C-DAG →</h3>
                            <p className="text-sm text-muted-foreground">
                                Learn about the Conceptual Debate Graph
                            </p>
                        </a>
                        <a
                            href="/docs/core-concepts/agents"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">Agents →</h3>
                            <p className="text-sm text-muted-foreground">
                                Explore the multi-agent system
                            </p>
                        </a>
                        <a
                            href="/tutorials"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">Tutorials →</h3>
                            <p className="text-sm text-muted-foreground">
                                Step-by-step practical guides
                            </p>
                        </a>
                    </div>
                </section>
            </div>
        </div>
    )
}
