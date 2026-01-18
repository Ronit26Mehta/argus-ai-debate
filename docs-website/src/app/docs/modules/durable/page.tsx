import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'

export const metadata: Metadata = {
    title: 'Durable Module | ARGUS Documentation',
    description: 'Durable execution and state persistence',
}

export default function DurableModulePage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Durable Module
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Durable execution and state persistence for long-running debates.
                    </p>
                </div>

                {/* Durable Execution */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Durable Execution</h2>
                    <CodeBlock
                        code={`from argus.durable import DurableOrchestrator

# Create durable orchestrator
orchestrator = DurableOrchestrator(
    llm=llm,
    state_dir="./debate_state"
)

# Run debate (can resume if interrupted)
result = orchestrator.debate(
    "proposition",
    debate_id="debate_001"
)

# Resume interrupted debate
result = orchestrator.resume("debate_001")`}
                        language="python"
                    />
                </section>

                {/* State Management */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">State Management</h2>
                    <CodeBlock
                        code={`from argus.durable import StateManager

manager = StateManager(state_dir="./state")

# Save state
manager.save_state("debate_001", debate_state)

# Load state
state = manager.load_state("debate_001")

# List all states
states = manager.list_states()`}
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
                        <a href="/docs/modules/provenance" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Provenance →</h3>
                            <p className="text-sm text-muted-foreground">Audit trail</p>
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
