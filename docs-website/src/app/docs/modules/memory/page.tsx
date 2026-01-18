import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'

export const metadata: Metadata = {
    title: 'Memory Module | ARGUS Documentation',
    description: 'Conversation memory and context management for agents',
}

export default function MemoryModulePage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Memory Module
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Conversation memory and context management for agents.
                    </p>
                </div>

                {/* Buffer Memory */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Buffer Memory</h2>
                    <CodeBlock
                        code={`from argus.memory import ConversationBufferMemory

# Create memory
memory = ConversationBufferMemory(max_messages=10)

# Add messages
memory.add_message("user", "What is AI?")
memory.add_message("assistant", "AI is...")

# Get messages
messages = memory.get_messages()

# Clear memory
memory.clear()`}
                        language="python"
                    />
                </section>

                {/* Summary Memory */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Summary Memory</h2>
                    <CodeBlock
                        code={`from argus.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(
    llm=llm,
    max_tokens=1000
)

# Automatically summarizes old messages
memory.add_message("user", "Long conversation...")
memory.add_message("assistant", "Response...")

# Get summary + recent messages
context = memory.get_context()`}
                        language="python"
                    />
                </section>

                {/* With Agents */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">With Agents</h2>
                    <CodeBlock
                        code={`from argus.agents import ProponentAgent
from argus.memory import ConversationBufferMemory

memory = ConversationBufferMemory()

agent = ProponentAgent(llm, memory=memory)

# Memory persists across calls
evidence1 = agent.gather_evidence(cdag, prop_id)
evidence2 = agent.gather_evidence(cdag, prop_id)  # Has context`}
                        language="python"
                    />
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a href="/docs/modules/agents" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Agents →</h3>
                            <p className="text-sm text-muted-foreground">Multi-agent system</p>
                        </a>
                        <a href="/docs/modules/orchestrator" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Orchestrator →</h3>
                            <p className="text-sm text-muted-foreground">Debate orchestration</p>
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
