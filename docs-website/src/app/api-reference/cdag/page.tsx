import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'

export const metadata: Metadata = {
    title: 'C-DAG API Reference | ARGUS',
    description: 'API reference for C-DAG module - graph structure and operations',
}

export default function CDAGAPIPage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        C-DAG API Reference
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Conceptual Debate Argumentation Graph classes and operations.
                    </p>
                </div>

                {/* CDAG */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold" id="CDAG">CDAG</h2>
                    <div className="p-6 rounded-xl border bg-muted/30">
                        <CodeBlock
                            code={`class CDAG:
    """Conceptual Debate Argumentation Graph."""
    
    def __init__(self, name: str = "debate"):
        """Initialize C-DAG."""
        pass
    
    def add_proposition(self, prop: Proposition) -> str:
        """Add proposition node."""
        pass
    
    def add_evidence(
        self,
        evidence: Evidence,
        target_id: str,
        edge_type: EdgeType
    ) -> str:
        """Add evidence node with edge."""
        pass
    
    def get_node(self, node_id: str) -> Node:
        """Get node by ID."""
        pass
    
    def get_evidence(self, prop_id: str) -> list[Evidence]:
        """Get all evidence for proposition."""
        pass
    
    def save(self, path: str):
        """Save graph to JSON."""
        pass
    
    @classmethod
    def load(cls, path: str) -> "CDAG":
        """Load graph from JSON."""
        pass`}
                            language="python"
                        />
                    </div>
                </section>

                {/* Proposition */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold" id="Proposition">Proposition</h2>
                    <div className="p-6 rounded-xl border bg-muted/30">
                        <CodeBlock
                            code={`class Proposition:
    """Proposition node."""
    
    def __init__(
        self,
        text: str,
        prior: float = 0.5,
        domain: str | None = None
    ):
        """
        Initialize proposition.
        
        Args:
            text: Proposition text
            prior: Prior probability (0-1)
            domain: Optional domain
        """
        self.id = generate_id()
        self.text = text
        self.prior = prior
        self.domain = domain`}
                            language="python"
                        />
                    </div>
                </section>

                {/* Evidence */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold" id="Evidence">Evidence</h2>
                    <div className="p-6 rounded-xl border bg-muted/30">
                        <CodeBlock
                            code={`class Evidence:
    """Evidence node."""
    
    def __init__(
        self,
        text: str,
        evidence_type: EvidenceType,
        polarity: int,  # +1 or -1
        confidence: float,
        relevance: float,
        source: str | None = None
    ):
        """
        Initialize evidence.
        
        Args:
            text: Evidence text
            evidence_type: Type of evidence
            polarity: +1 (supporting) or -1 (opposing)
            confidence: Confidence (0-1)
            relevance: Relevance (0-1)
            source: Optional source URL
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
                        <a href="/api-reference/core" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Core →</h3>
                            <p className="text-sm text-muted-foreground">Core classes</p>
                        </a>
                        <a href="/docs/modules/cdag" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">C-DAG Docs →</h3>
                            <p className="text-sm text-muted-foreground">Module documentation</p>
                        </a>
                        <a href="/docs/core-concepts/cdag" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Concepts →</h3>
                            <p className="text-sm text-muted-foreground">Learn C-DAG</p>
                        </a>
                    </div>
                </section>
            </div>
        </div>
    )
}
