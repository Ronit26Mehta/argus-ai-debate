import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'

export const metadata: Metadata = {
    title: 'Provenance Module | ARGUS Documentation',
    description: 'Full audit trail and integrity verification for debates',
}

export default function ProvenanceModulePage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Provenance Module
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Full audit trail and cryptographic integrity verification for all debate operations.
                    </p>
                </div>

                {/* Provenance Tracking */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Provenance Tracking</h2>
                    <CodeBlock
                        code={`from argus.provenance import ProvenanceLedger, EventType

# Create ledger
ledger = ProvenanceLedger()

# Record events
ledger.record(EventType.SESSION_START)
ledger.record(EventType.PROPOSITION_ADDED, entity_id=prop.id)
ledger.record(EventType.EVIDENCE_ADDED, entity_id=evidence.id)
ledger.record(EventType.VERDICT_RENDERED, entity_id=prop.id)
ledger.record(EventType.SESSION_END)

# View events
for event in ledger.events:
    print(f"{event.timestamp}: {event.event_type}")
    print(f"  Entity: {event.entity_id}")
    print(f"  Hash: {event.content_hash}")`}
                        language="python"
                    />
                </section>

                {/* Integrity Verification */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Integrity Verification</h2>
                    <CodeBlock
                        code={`# Verify integrity
is_valid, errors = ledger.verify_integrity()

if is_valid:
    print("✓ Provenance chain is valid")
else:
    print("✗ Integrity violations:")
    for error in errors:
        print(f"  - {error}")

# Export ledger
ledger.save("provenance.json")

# Load and verify
loaded = ProvenanceLedger.load("provenance.json")
is_valid, _ = loaded.verify_integrity()`}
                        language="python"
                    />
                </section>

                {/* With Orchestrator */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">With Orchestrator</h2>
                    <CodeBlock
                        code={`from argus import RDCOrchestrator

orchestrator = RDCOrchestrator(
    llm=llm,
    enable_provenance=True  # Enable tracking
)

result = orchestrator.debate("proposition")

# Access provenance
if result.provenance:
    is_valid, errors = result.provenance.verify_integrity()
    print(f"Valid: {is_valid}")
    print(f"Events: {len(result.provenance)}")`}
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
                        <a href="/docs/modules/cdag" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">C-DAG →</h3>
                            <p className="text-sm text-muted-foreground">Graph structure</p>
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
