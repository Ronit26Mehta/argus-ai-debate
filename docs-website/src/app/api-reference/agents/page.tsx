import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'

export const metadata: Metadata = {
    title: 'Agents API Reference | ARGUS',
    description: 'API reference for agents module - multi-agent system classes',
}

export default function AgentsAPIPage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Agents API Reference
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Multi-agent system classes and interfaces.
                    </p>
                </div>

                {/* ProponentAgent */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold" id="ProponentAgent">ProponentAgent</h2>
                    <div className="p-6 rounded-xl border bg-muted/30">
                        <CodeBlock
                            code={`class ProponentAgent(BaseAgent):
    """Gathers supporting evidence for propositions."""
    
    def __init__(
        self,
        llm: BaseLLM,
        domain: str | None = None,
        **kwargs
    ):
        """
        Initialize proponent agent.
        
        Args:
            llm: Language model instance
            domain: Optional domain expertise
        """
        pass
    
    def gather_evidence(
        self,
        graph: CDAG,
        proposition_id: str,
        max_evidence: int = 5
    ) -> list[Evidence]:
        """
        Gather supporting evidence.
        
        Args:
            graph: C-DAG instance
            proposition_id: Proposition to support
            max_evidence: Maximum evidence items
        
        Returns:
            List of Evidence objects
        """
        pass`}
                            language="python"
                        />
                    </div>
                </section>

                {/* OpponentAgent */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold" id="OpponentAgent">OpponentAgent</h2>
                    <div className="p-6 rounded-xl border bg-muted/30">
                        <CodeBlock
                            code={`class OpponentAgent(BaseAgent):
    """Challenges claims with counter-evidence."""
    
    def challenge_claim(
        self,
        graph: CDAG,
        proposition_id: str,
        max_challenges: int = 3
    ) -> list[Evidence]:
        """
        Generate counter-evidence.
        
        Args:
            graph: C-DAG instance
            proposition_id: Proposition to challenge
            max_challenges: Maximum challenges
        
        Returns:
            List of counter-evidence
        """
        pass`}
                            language="python"
                        />
                    </div>
                </section>

                {/* JudgeAgent */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold" id="JudgeAgent">JudgeAgent</h2>
                    <div className="p-6 rounded-xl border bg-muted/30">
                        <CodeBlock
                            code={`class JudgeAgent(BaseAgent):
    """Evaluates arguments and renders verdicts."""
    
    def evaluate(
        self,
        graph: CDAG,
        proposition_id: str
    ) -> Verdict:
        """
        Evaluate proposition and render verdict.
        
        Args:
            graph: C-DAG instance
            proposition_id: Proposition to evaluate
        
        Returns:
            Verdict with label, posterior, and reasoning
        """
        pass`}
                            language="python"
                        />
                    </div>
                </section>

                {/* Next */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">See Also</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a href="/api-reference/cdag" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">C-DAG →</h3>
                            <p className="text-sm text-muted-foreground">Graph classes</p>
                        </a>
                        <a href="/docs/modules/agents" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Agents Docs →</h3>
                            <p className="text-sm text-muted-foreground">Module documentation</p>
                        </a>
                        <a href="/tutorials/custom-agent-pipeline" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Tutorial →</h3>
                            <p className="text-sm text-muted-foreground">Custom agents</p>
                        </a>
                    </div>
                </section>
            </div>
        </div>
    )
}
