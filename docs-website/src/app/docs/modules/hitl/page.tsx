import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'

export const metadata: Metadata = {
    title: 'HITL Module | ARGUS Documentation',
    description: 'Human-in-the-loop integration for debates',
}

export default function HITLModulePage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        HITL Module
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Human-in-the-loop integration for debates - get human feedback and intervention.
                    </p>
                </div>

                {/* Basic HITL */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Human Feedback</h2>
                    <CodeBlock
                        code={`from argus.hitl import HumanFeedback

feedback = HumanFeedback()

# Request human input
response = feedback.request_input(
    prompt="Is this evidence credible?",
    options=["Yes", "No", "Unsure"],
    context={"evidence": evidence.text}
)

print(f"Human response: {response}")`}
                        language="python"
                    />
                </section>

                {/* With Orchestrator */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">With Orchestrator</h2>
                    <CodeBlock
                        code={`from argus import RDCOrchestrator
from argus.hitl import HumanReviewer

reviewer = HumanReviewer()

orchestrator = RDCOrchestrator(
    llm=llm,
    human_reviewer=reviewer,
    review_frequency="per_round"  # or "on_uncertainty"
)

# Human will be prompted during debate
result = orchestrator.debate("proposition")`}
                        language="python"
                    />
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a href="/docs/modules/orchestrator" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Orchestrator →</h3>
                            <p className="text-sm text-muted-foreground">Debate orchestration</p>
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
