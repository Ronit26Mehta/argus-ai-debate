import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'
import { Callout } from '@/components/Callout'

export const metadata: Metadata = {
    title: 'Multi-Agent System | ARGUS Documentation',
    description: 'Learn about the multi-agent architecture in ARGUS - Moderator, Specialist, Refuter, and Jury',
}

export default function AgentsPage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Multi-Agent System
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        ARGUS orchestrates specialized AI agents that work together to evaluate propositions through structured debate.
                    </p>
                </div>

                {/* Agent Roles */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Agent Roles</h2>

                    <div className="overflow-x-auto">
                        <table className="w-full border-collapse rounded-lg overflow-hidden">
                            <thead className="bg-muted">
                                <tr>
                                    <th className="px-4 py-3 text-left font-semibold">Agent</th>
                                    <th className="px-4 py-3 text-left font-semibold">Role</th>
                                    <th className="px-4 py-3 text-left font-semibold">Capabilities</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr className="border-t">
                                    <td className="px-4 py-3 font-semibold">Moderator</td>
                                    <td className="px-4 py-3">Orchestration</td>
                                    <td className="px-4 py-3 text-sm">Creates agendas, manages rounds, evaluates stopping criteria, breaks ties</td>
                                </tr>
                                <tr className="border-t bg-muted/30">
                                    <td className="px-4 py-3 font-semibold">Specialist</td>
                                    <td className="px-4 py-3">Evidence Gathering</td>
                                    <td className="px-4 py-3 text-sm">Domain-specific research, hybrid retrieval, source quality assessment</td>
                                </tr>
                                <tr className="border-t">
                                    <td className="px-4 py-3 font-semibold">Refuter</td>
                                    <td className="px-4 py-3">Challenge Generation</td>
                                    <td className="px-4 py-3 text-sm">Counter-evidence, methodological critiques, logical fallacy detection</td>
                                </tr>
                                <tr className="border-t bg-muted/30">
                                    <td className="px-4 py-3 font-semibold">Jury</td>
                                    <td className="px-4 py-3">Verdict Rendering</td>
                                    <td className="px-4 py-3 text-sm">Bayesian aggregation, confidence calibration, label assignment</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </section>

                {/* Moderator */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Moderator Agent</h2>
                    <p className="text-muted-foreground">
                        The Moderator orchestrates the debate process and ensures fair proceedings:
                    </p>

                    <CodeBlock
                        code={`from argus.agents import Moderator
from argus import get_llm, CDAG, Proposition

# Initialize moderator
llm = get_llm("openai", model="gpt-4o")
moderator = Moderator(llm)

# Create debate graph
graph = CDAG()
prop = Proposition(text="The intervention is cost-effective", prior=0.5)
graph.add_proposition(prop)

# Moderator creates agenda
agenda = moderator.create_agenda(graph, prop.id)
print(f"Agenda: {agenda}")

# Check stopping criteria
should_stop = moderator.should_stop(graph, prop.id)
print(f"Should stop: {should_stop}")`}
                        language="python"
                    />

                    <Callout variant="tip" title="Best Model for Moderator">
                        Use GPT-4 or Claude for moderation - they excel at planning and orchestration tasks.
                    </Callout>
                </section>

                {/* Specialist */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Specialist Agent</h2>
                    <p className="text-muted-foreground">
                        Specialists gather domain-specific evidence to support or challenge propositions:
                    </p>

                    <CodeBlock
                        code={`from argus.agents import Specialist
from argus import get_llm

# Initialize specialist with domain expertise
llm = get_llm("anthropic", model="claude-3-5-sonnet-20241022")
specialist = Specialist(llm, domain="clinical")

# Gather evidence
evidence = specialist.gather_evidence(graph, prop.id)

for e in evidence:
    print(f"[{e.polarity:+d}] {e.text}")
    print(f"   Confidence: {e.confidence:.2f}")
    print(f"   Source: {e.source}")`}
                        language="python"
                    />

                    <div className="grid gap-4 md:grid-cols-2">
                        <div className="p-4 rounded-lg border bg-muted/30">
                            <h4 className="font-semibold mb-2">Domain Specialization</h4>
                            <p className="text-sm text-muted-foreground">
                                Configure specialists for specific domains: clinical, policy, research, finance, etc.
                            </p>
                        </div>
                        <div className="p-4 rounded-lg border bg-muted/30">
                            <h4 className="font-semibold mb-2">Retrieval Integration</h4>
                            <p className="text-sm text-muted-foreground">
                                Specialists can use hybrid retrieval to find relevant evidence from documents.
                            </p>
                        </div>
                    </div>
                </section>

                {/* Refuter */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Refuter Agent</h2>
                    <p className="text-muted-foreground">
                        Refuters challenge evidence and identify weaknesses in arguments:
                    </p>

                    <CodeBlock
                        code={`from argus.agents import Refuter
from argus import get_llm

# Initialize refuter
llm = get_llm("groq", model="llama-3.1-70b-versatile")  # Fast inference
refuter = Refuter(llm)

# Generate rebuttals
rebuttals = refuter.generate_rebuttals(graph, prop.id)

for r in rebuttals:
    print(f"Rebuttal to: {r.target_id}")
    print(f"Type: {r.rebuttal_type}")
    print(f"Strength: {r.strength:.2f}")
    print(f"Text: {r.text}")`}
                        language="python"
                    />

                    <Callout variant="info" title="Rebuttal Types">
                        Refuters can generate different types of rebuttals: methodological critiques, alternative explanations,
                        data quality challenges, and logical fallacy detection.
                    </Callout>
                </section>

                {/* Jury */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Jury Agent</h2>
                    <p className="text-muted-foreground">
                        The Jury evaluates all evidence and renders a final verdict:
                    </p>

                    <CodeBlock
                        code={`from argus.agents import Jury
from argus import get_llm

# Initialize jury
llm = get_llm("gemini", model="gemini-1.5-pro")
jury = Jury(llm)

# Render verdict
verdict = jury.evaluate(graph, prop.id)

print(f"Verdict: {verdict.label}")
print(f"Posterior: {verdict.posterior:.3f}")
print(f"Confidence: {verdict.confidence:.3f}")
print(f"Reasoning: {verdict.reasoning}")`}
                        language="python"
                    />

                    <div className="p-6 rounded-xl border bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-950/20 dark:to-purple-950/20">
                        <h4 className="font-semibold mb-3">Verdict Structure</h4>
                        <ul className="space-y-2 text-sm">
                            <li><strong>Label:</strong> SUPPORTED, REFUTED, or UNCERTAIN</li>
                            <li><strong>Posterior:</strong> Calibrated probability (0-1)</li>
                            <li><strong>Confidence:</strong> Jury's confidence in the verdict</li>
                            <li><strong>Reasoning:</strong> Detailed explanation</li>
                        </ul>
                    </div>
                </section>

                {/* Multi-Model Configuration */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Multi-Model Configuration</h2>
                    <p className="text-muted-foreground">
                        Use different LLMs for different agents based on their strengths:
                    </p>

                    <CodeBlock
                        code={`from argus import get_llm
from argus.agents import Moderator, Specialist, Refuter, Jury

# Different models for different tasks
moderator = Moderator(get_llm("openai", model="gpt-4o"))
specialist = Specialist(get_llm("anthropic", model="claude-3-5-sonnet-20241022"), domain="policy")
refuter = Refuter(get_llm("groq", model="llama-3.1-70b-versatile"))
jury = Jury(get_llm("gemini", model="gemini-1.5-pro"))

# Or use local models
moderator = Moderator(get_llm("ollama", model="llama3.2"))
specialist = Specialist(get_llm("ollama", model="mistral"), domain="research")
refuter = Refuter(get_llm("ollama", model="llama3.2"))
jury = Jury(get_llm("ollama", model="llama3.2"))`}
                        language="python"
                    />

                    <Callout variant="success" title="Recommended Combinations">
                        <ul className="list-disc list-inside space-y-1">
                            <li><strong>GPT-4:</strong> Moderator (planning, orchestration)</li>
                            <li><strong>Claude:</strong> Specialist (deep analysis, research)</li>
                            <li><strong>Groq/Llama:</strong> Refuter (fast, critical thinking)</li>
                            <li><strong>Gemini:</strong> Jury (balanced, comprehensive evaluation)</li>
                        </ul>
                    </Callout>
                </section>

                {/* Complete Example */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Complete Multi-Agent Example</h2>
                    <p className="text-muted-foreground">
                        Putting it all together:
                    </p>

                    <CodeBlock
                        code={`from argus import get_llm, CDAG, Proposition
from argus.agents import Moderator, Specialist, Refuter, Jury
from argus.provenance import ProvenanceLedger, EventType

# Initialize provenance tracking
ledger = ProvenanceLedger()
ledger.record(EventType.SESSION_START)

# Initialize agents with different LLMs
moderator = Moderator(get_llm("openai", model="gpt-4o"))
specialist = Specialist(get_llm("anthropic", model="claude-3-5-sonnet-20241022"), domain="policy")
refuter = Refuter(get_llm("groq", model="llama-3.1-70b-versatile"))
jury = Jury(get_llm("gemini", model="gemini-1.5-pro"))

# Create debate
graph = CDAG()
prop = Proposition(
    text="Carbon pricing is effective for reducing emissions",
    prior=0.5,
)
graph.add_proposition(prop)
ledger.record(EventType.PROPOSITION_ADDED, entity_id=prop.id)

# Run debate rounds
for round_num in range(3):
    print(f"\\n=== Round {round_num + 1} ===")
    
    # Gather evidence
    evidence = specialist.gather_evidence(graph, prop.id)
    for e in evidence:
        ledger.record(EventType.EVIDENCE_ADDED, entity_id=e.id)
        print(f"Evidence: {e.text[:80]}...")
    
    # Generate rebuttals
    rebuttals = refuter.generate_rebuttals(graph, prop.id)
    for r in rebuttals:
        ledger.record(EventType.REBUTTAL_ADDED, entity_id=r.id)
        print(f"Rebuttal: {r.text[:80]}...")
    
    # Check stopping criteria
    if moderator.should_stop(graph, prop.id):
        print("Convergence reached!")
        break

# Render verdict
verdict = jury.evaluate(graph, prop.id)
ledger.record(EventType.VERDICT_RENDERED, entity_id=prop.id)
ledger.record(EventType.SESSION_END)

print(f"\\n=== Final Verdict ===")
print(f"Label: {verdict.label}")
print(f"Posterior: {verdict.posterior:.3f}")
print(f"Confidence: {verdict.confidence:.3f}")
print(f"Reasoning: {verdict.reasoning}")

# Verify integrity
is_valid, errors = ledger.verify_integrity()
print(f"\\nProvenance valid: {is_valid}")
print(f"Total events: {len(ledger)}")`}
                        language="python"
                        filename="multi_agent_debate.py"
                    />
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a
                            href="/docs/modules/agents"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">Agents Module →</h3>
                            <p className="text-sm text-muted-foreground">
                                Deep dive into agent implementation
                            </p>
                        </a>
                        <a
                            href="/docs/llm-providers"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">LLM Providers →</h3>
                            <p className="text-sm text-muted-foreground">
                                Explore 27+ supported providers
                            </p>
                        </a>
                        <a
                            href="/tutorials/custom-agent-pipeline"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">Tutorial →</h3>
                            <p className="text-sm text-muted-foreground">
                                Build a custom agent pipeline
                            </p>
                        </a>
                    </div>
                </section>
            </div>
        </div>
    )
}
