import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'
import { Callout } from '@/components/Callout'

export const metadata: Metadata = {
    title: 'Agents Module | ARGUS Documentation',
    description: 'Complete guide to the agents module - multi-agent debate system implementation',
}

export default function AgentsModulePage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Agents Module
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Specialized AI agents for multi-agent debate systems with role-based capabilities.
                    </p>
                </div>

                {/* Overview */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Overview</h2>
                    <p className="text-muted-foreground">
                        The agents module implements specialized AI agents that work together in structured debates:
                    </p>
                    <ul className="list-disc list-inside space-y-2 text-muted-foreground ml-4">
                        <li><strong>ProponentAgent:</strong> Gathers supporting evidence</li>
                        <li><strong>OpponentAgent:</strong> Challenges claims with counter-evidence</li>
                        <li><strong>JudgeAgent:</strong> Evaluates arguments and renders verdicts</li>
                        <li><strong>ResearchAgent:</strong> Conducts deep research with retrieval</li>
                        <li><strong>SynthesisAgent:</strong> Synthesizes findings into coherent conclusions</li>
                    </ul>
                </section>

                {/* Agent Types */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Agent Types</h2>

                    <div className="space-y-6">
                        <div className="p-6 rounded-xl border bg-card">
                            <h3 className="text-xl font-semibold mb-3">Proponent Agent</h3>
                            <CodeBlock
                                code={`from argus.agents import ProponentAgent
from argus import get_llm

llm = get_llm("openai", model="gpt-4o")
agent = ProponentAgent(llm, domain="clinical")

# Gather supporting evidence
evidence = agent.gather_evidence(
    graph=cdag,
    proposition_id=prop_id,
    max_evidence=5
)

for e in evidence:
    print(f"[+] {e.text}")
    print(f"    Confidence: {e.confidence:.2f}")`}
                                language="python"
                            />
                        </div>

                        <div className="p-6 rounded-xl border bg-card">
                            <h3 className="text-xl font-semibold mb-3">Opponent Agent</h3>
                            <CodeBlock
                                code={`from argus.agents import OpponentAgent

agent = OpponentAgent(llm)

# Generate counter-evidence
counter_evidence = agent.challenge_claim(
    graph=cdag,
    proposition_id=prop_id,
    max_challenges=3
)

for e in counter_evidence:
    print(f"[-] {e.text}")
    print(f"    Strength: {e.confidence:.2f}")`}
                                language="python"
                            />
                        </div>

                        <div className="p-6 rounded-xl border bg-card">
                            <h3 className="text-xl font-semibold mb-3">Judge Agent</h3>
                            <CodeBlock
                                code={`from argus.agents import JudgeAgent

agent = JudgeAgent(llm)

# Evaluate and render verdict
verdict = agent.evaluate(
    graph=cdag,
    proposition_id=prop_id
)

print(f"Verdict: {verdict.label}")
print(f"Posterior: {verdict.posterior:.3f}")
print(f"Reasoning: {verdict.reasoning}")`}
                                language="python"
                            />
                        </div>
                    </div>
                </section>

                {/* Configuration */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Agent Configuration</h2>
                    <CodeBlock
                        code={`from argus.agents import ProponentAgent
from argus import get_llm

# Configure agent
agent = ProponentAgent(
    llm=get_llm("anthropic", model="claude-3-5-sonnet-20241022"),
    domain="policy",           # Domain expertise
    temperature=0.7,           # Creativity level
    max_tokens=4096,          # Response length
    tools=[search_tool],      # Available tools
    memory=memory_store,      # Conversation memory
)

# Use with retriever
agent.set_retriever(hybrid_retriever)`}
                        language="python"
                    />
                </section>

                {/* Custom Agents */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Creating Custom Agents</h2>
                    <CodeBlock
                        code={`from argus.agents import BaseAgent
from argus import CDAG

class MyCustomAgent(BaseAgent):
    """Custom agent implementation."""
    
    def __init__(self, llm, domain: str = None, **kwargs):
        super().__init__(llm, **kwargs)
        self.domain = domain
    
    def process(self, graph: CDAG, proposition_id: str):
        """Custom processing logic."""
        # Your implementation
        prompt = self._build_prompt(graph, proposition_id)
        response = self.llm.generate(prompt)
        return self._parse_response(response)
    
    def _build_prompt(self, graph, prop_id):
        # Build custom prompt
        return f"Analyze: {graph.get_node(prop_id).text}"
    
    def _parse_response(self, response):
        # Parse LLM response
        return response

# Use custom agent
agent = MyCustomAgent(llm, domain="finance")
result = agent.process(cdag, prop_id)`}
                        language="python"
                    />
                </section>

                {/* Agent Memory */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Agent Memory</h2>
                    <CodeBlock
                        code={`from argus.agents import ProponentAgent
from argus.memory import ConversationBufferMemory

# Create memory
memory = ConversationBufferMemory(max_messages=10)

# Agent with memory
agent = ProponentAgent(llm, memory=memory)

# Memory persists across calls
evidence1 = agent.gather_evidence(cdag, prop_id)
evidence2 = agent.gather_evidence(cdag, prop_id)  # Remembers context

# View memory
print(agent.memory.get_messages())`}
                        language="python"
                    />
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a href="/docs/core-concepts/agents" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Core Concepts →</h3>
                            <p className="text-sm text-muted-foreground">Learn about multi-agent systems</p>
                        </a>
                        <a href="/docs/modules/memory" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Memory →</h3>
                            <p className="text-sm text-muted-foreground">Agent memory systems</p>
                        </a>
                        <a href="/tutorials" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Tutorials →</h3>
                            <p className="text-sm text-muted-foreground">Build with agents</p>
                        </a>
                    </div>
                </section>
            </div>
        </div>
    )
}
