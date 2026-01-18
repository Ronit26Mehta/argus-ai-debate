import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'
import { Callout } from '@/components/Callout'

export const metadata: Metadata = {
    title: 'Custom Agent Pipeline Tutorial | ARGUS',
    description: 'Build a custom multi-agent pipeline with different LLMs and provenance tracking',
}

export default function CustomAgentPipelineTutorial() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Custom Agent Pipeline
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Build a custom multi-agent pipeline with different LLMs and full provenance tracking.
                    </p>
                    <div className="flex gap-4 mt-4 text-sm text-muted-foreground">
                        <span>⏱️ 45 minutes</span>
                        <span>📊 Advanced</span>
                        <span>🤖 Multi-Agent</span>
                    </div>
                </div>

                {/* Custom Agent */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Step 1: Create Custom Agent</h2>
                    <CodeBlock
                        code={`from argus.agents import BaseAgent
from argus import CDAG, Evidence
from argus.cdag.nodes import EvidenceType

class DomainExpertAgent(BaseAgent):
    """Custom domain expert agent."""
    
    def __init__(self, llm, domain: str, expertise_level: str = "expert"):
        super().__init__(llm)
        self.domain = domain
        self.expertise_level = expertise_level
    
    def analyze(self, graph: CDAG, proposition_id: str):
        """Perform domain-specific analysis."""
        prop = graph.get_node(proposition_id)
        
        prompt = f"""As a {self.expertise_level} in {self.domain}, 
analyze this claim: {prop.text}

Provide:
1. Domain-specific evidence
2. Methodological considerations
3. Confidence assessment"""
        
        response = self.llm.generate(prompt)
        return self._parse_analysis(response)
    
    def _parse_analysis(self, response):
        # Parse LLM response into structured evidence
        evidence_list = []
        # Implementation here
        return evidence_list

# Use custom agent
expert = DomainExpertAgent(llm, domain="finance", expertise_level="senior")`}
                        language="python"
                    />
                </section>

                {/* Multi-Agent Pipeline */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Step 2: Build Pipeline</h2>
                    <CodeBlock
                        code={`from argus import get_llm, CDAG, Proposition
from argus.agents import Moderator, Specialist, Refuter, Jury
from argus.provenance import ProvenanceLedger, EventType

# Initialize different LLMs
gpt4 = get_llm("openai", model="gpt-4o")
claude = get_llm("anthropic", model="claude-3-5-sonnet-20241022")
groq = get_llm("groq", model="llama-3.1-70b-versatile")
gemini = get_llm("gemini", model="gemini-1.5-pro")

# Create agents
moderator = Moderator(gpt4)
specialist = Specialist(claude, domain="policy")
refuter = Refuter(groq)
jury = Jury(gemini)
expert = DomainExpertAgent(claude, domain="policy")

# Provenance tracking
ledger = ProvenanceLedger()
ledger.record(EventType.SESSION_START)

# Create debate
graph = CDAG()
prop = Proposition(
    text="Carbon pricing is effective for reducing emissions",
    prior=0.5
)
graph.add_proposition(prop)
ledger.record(EventType.PROPOSITION_ADDED, entity_id=prop.id)`}
                        language="python"
                    />
                </section>

                {/* Run Pipeline */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Step 3: Execute Pipeline</h2>
                    <CodeBlock
                        code={`# Run multi-round debate
for round_num in range(3):
    print(f"\\n=== Round {round_num + 1} ===")
    
    # Custom expert analysis
    expert_evidence = expert.analyze(graph, prop.id)
    for e in expert_evidence:
        ledger.record(EventType.EVIDENCE_ADDED, entity_id=e.id)
    
    # Specialist gathers evidence
    specialist_evidence = specialist.gather_evidence(graph, prop.id)
    for e in specialist_evidence:
        ledger.record(EventType.EVIDENCE_ADDED, entity_id=e.id)
    
    # Refuter challenges
    rebuttals = refuter.generate_rebuttals(graph, prop.id)
    for r in rebuttals:
        ledger.record(EventType.REBUTTAL_ADDED, entity_id=r.id)
    
    # Check stopping
    if moderator.should_stop(graph, prop.id):
        print("Convergence reached!")
        break

# Final verdict
verdict = jury.evaluate(graph, prop.id)
ledger.record(EventType.VERDICT_RENDERED, entity_id=prop.id)
ledger.record(EventType.SESSION_END)

print(f"\\nVerdict: {verdict.label}")
print(f"Posterior: {verdict.posterior:.3f}")`}
                        language="python"
                    />
                </section>

                {/* Verify Provenance */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Step 4: Verify Provenance</h2>
                    <CodeBlock
                        code={`# Verify integrity
is_valid, errors = ledger.verify_integrity()
print(f"Provenance valid: {is_valid}")
print(f"Total events: {len(ledger)}")

# Export ledger
ledger.save("debate_provenance.json")

# View event timeline
for event in ledger.events[:10]:
    print(f"{event.timestamp}: {event.event_type}")
    print(f"  Entity: {event.entity_id}")
    print(f"  Hash: {event.content_hash[:16]}...")`}
                        language="python"
                    />
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-2">
                        <a href="/docs/modules/agents" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Agents Module →</h3>
                            <p className="text-sm text-muted-foreground">Deep dive into agents</p>
                        </a>
                        <a href="/docs/modules/provenance" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Provenance →</h3>
                            <p className="text-sm text-muted-foreground">Learn provenance tracking</p>
                        </a>
                    </div>
                </section>
            </div>
        </div>
    )
}
