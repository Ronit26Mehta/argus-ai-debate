import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Network, Brain, Shield, TrendingUp } from "lucide-react"

export default function CoreConceptsPage() {
    return (
        <div className="space-y-8">
            <div>
                <h1 className="text-4xl font-bold mb-4">Core Concepts</h1>
                <p className="text-lg text-muted-foreground">
                    Understanding the fundamental concepts behind ARGUS's approach to evidence-based reasoning.
                </p>
            </div>

            {/* Research Debate Chain */}
            <section className="space-y-4">
                <h2 className="text-2xl font-semibold">Research Debate Chain (RDC)</h2>
                <Card>
                    <CardHeader>
                        <Brain className="h-10 w-10 text-primary mb-2" />
                        <CardTitle>What is RDC?</CardTitle>
                        <CardDescription>
                            A novel approach to AI reasoning that structures knowledge evaluation as multi-agent debates
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <p>
                            Instead of single-pass inference, ARGUS orchestrates specialist agents that gather evidence,
                            generate rebuttals, and render verdicts through Bayesian aggregation.
                        </p>

                        <div className="space-y-2">
                            <h4 className="font-semibold">Key Principles:</h4>
                            <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                                <li>Adversarial debate reduces hallucination</li>
                                <li>Multiple perspectives challenge single-point failures</li>
                                <li>Bayesian aggregation provides calibrated confidence</li>
                                <li>Full provenance enables audit trails</li>
                            </ul>
                        </div>

                        <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                            <code className="text-sm">{`from argus import RDCOrchestrator, get_llm

llm = get_llm("openai", model="gpt-4o")
orchestrator = RDCOrchestrator(llm=llm, max_rounds=5)

result = orchestrator.debate(
    "The treatment is effective",
    prior=0.5,
)

print(f"Posterior: {result.verdict.posterior:.3f}")
print(f"Confidence: {result.verdict.confidence:.3f}")`}</code>
                        </pre>
                    </CardContent>
                </Card>
            </section>

            {/* C-DAG */}
            <section className="space-y-4">
                <h2 className="text-2xl font-semibold">Conceptual Debate Graph (C-DAG)</h2>
                <Card>
                    <CardHeader>
                        <Network className="h-10 w-10 text-primary mb-2" />
                        <CardTitle>Graph-Based Argumentation</CardTitle>
                        <CardDescription>
                            Directed graph where propositions, evidence, and rebuttals are nodes with signed edges
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="space-y-2">
                            <h4 className="font-semibold">Node Types:</h4>
                            <ul className="space-y-1 text-sm">
                                <li><strong>Proposition</strong>: Main claims under evaluation</li>
                                <li><strong>Evidence</strong>: Supporting/attacking information</li>
                                <li><strong>Rebuttal</strong>: Challenges to evidence</li>
                                <li><strong>Finding</strong>: Intermediate conclusions</li>
                                <li><strong>Assumption</strong>: Underlying premises</li>
                            </ul>
                        </div>

                        <div className="space-y-2">
                            <h4 className="font-semibold">Edge Types:</h4>
                            <ul className="space-y-1 text-sm">
                                <li><strong>SUPPORTS (+1)</strong>: Evidence supporting a proposition</li>
                                <li><strong>ATTACKS (-1)</strong>: Evidence challenging a proposition</li>
                                <li><strong>REBUTS (-1)</strong>: Rebuttal targeting evidence</li>
                                <li><strong>REFINES (0)</strong>: Clarification or specification</li>
                            </ul>
                        </div>

                        <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                            <code className="text-sm">{`from argus import CDAG, Proposition, Evidence, EdgeType

graph = CDAG(name="debate")
prop = Proposition(text="Drug X is effective", prior=0.5)
graph.add_proposition(prop)

evidence = Evidence(
    text="Phase 3 RCT showed 35% reduction",
    polarity=1,
    confidence=0.9,
)
graph.add_evidence(evidence, prop.id, EdgeType.SUPPORTS)

posterior = compute_posterior(graph, prop.id)
print(f"Posterior: {posterior:.3f}")`}</code>
                        </pre>
                    </CardContent>
                </Card>
            </section>

            {/* Multi-Agent System */}
            <section className="space-y-4">
                <h2 className="text-2xl font-semibold">Multi-Agent System</h2>
                <div className="grid md:grid-cols-2 gap-4">
                    <Card>
                        <CardHeader>
                            <CardTitle>Moderator</CardTitle>
                            <CardDescription>Orchestration</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <ul className="space-y-1 text-sm text-muted-foreground">
                                <li>• Creates debate agendas</li>
                                <li>• Manages rounds</li>
                                <li>• Evaluates stopping criteria</li>
                                <li>• Breaks ties</li>
                            </ul>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader>
                            <CardTitle>Specialist</CardTitle>
                            <CardDescription>Evidence Gathering</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <ul className="space-y-1 text-sm text-muted-foreground">
                                <li>• Domain-specific research</li>
                                <li>• Hybrid retrieval</li>
                                <li>• Source quality assessment</li>
                                <li>• Evidence synthesis</li>
                            </ul>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader>
                            <CardTitle>Refuter</CardTitle>
                            <CardDescription>Challenge Generation</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <ul className="space-y-1 text-sm text-muted-foreground">
                                <li>• Counter-evidence</li>
                                <li>• Methodological critiques</li>
                                <li>• Logical fallacy detection</li>
                                <li>• Alternative explanations</li>
                            </ul>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader>
                            <CardTitle>Jury</CardTitle>
                            <CardDescription>Verdict Rendering</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <ul className="space-y-1 text-sm text-muted-foreground">
                                <li>• Bayesian aggregation</li>
                                <li>• Confidence calibration</li>
                                <li>• Label assignment</li>
                                <li>• Reasoning synthesis</li>
                            </ul>
                        </CardContent>
                    </Card>
                </div>
            </section>

            {/* Bayesian Reasoning */}
            <section className="space-y-4">
                <h2 className="text-2xl font-semibold">Bayesian Reasoning</h2>
                <Card>
                    <CardHeader>
                        <TrendingUp className="h-10 w-10 text-primary mb-2" />
                        <CardTitle>Probabilistic Belief Updating</CardTitle>
                        <CardDescription>
                            Log-odds space for numerically stable Bayesian belief propagation
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="bg-muted p-4 rounded-lg">
                            <p className="font-mono text-sm">
                                posterior = σ(log-odds(prior) + Σᵢ wᵢ · log(LRᵢ))
                            </p>
                        </div>

                        <div className="space-y-2">
                            <p className="text-sm">Where:</p>
                            <ul className="space-y-1 text-sm text-muted-foreground">
                                <li>• σ is the logistic (sigmoid) function</li>
                                <li>• LRᵢ is the likelihood ratio for evidence i</li>
                                <li>• wᵢ = polarityᵢ × confidenceᵢ × relevanceᵢ × qualityᵢ</li>
                            </ul>
                        </div>

                        <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                            <code className="text-sm">{`from argus.decision import BayesianUpdater

updater = BayesianUpdater()
posterior = updater.update(
    prior=0.5,
    likelihood_ratio=2.5,
    confidence=0.8,
)

print(f"Updated belief: {posterior:.3f}")`}</code>
                        </pre>
                    </CardContent>
                </Card>
            </section>

            {/* Provenance */}
            <section className="space-y-4">
                <h2 className="text-2xl font-semibold">Provenance Tracking</h2>
                <Card>
                    <CardHeader>
                        <Shield className="h-10 w-10 text-primary mb-2" />
                        <CardTitle>PROV-O Compatible Ledger</CardTitle>
                        <CardDescription>
                            W3C standard compliance with cryptographic attestations
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="space-y-2">
                            <h4 className="font-semibold">Features:</h4>
                            <ul className="space-y-1 text-sm text-muted-foreground">
                                <li>• SHA-256 hash chain for tamper detection</li>
                                <li>• PROV-O compatible event model</li>
                                <li>• Cryptographic attestations for content</li>
                                <li>• Complete audit trails</li>
                                <li>• Query API for filtering and analysis</li>
                            </ul>
                        </div>

                        <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                            <code className="text-sm">{`from argus.provenance import ProvenanceLedger, EventType

ledger = ProvenanceLedger()
ledger.record(EventType.SESSION_START)
ledger.record(EventType.PROPOSITION_ADDED, entity_id=prop.id)
ledger.record(EventType.EVIDENCE_ADDED, entity_id=evidence.id)

# Verify integrity
is_valid, errors = ledger.verify_integrity()
print(f"Ledger valid: {is_valid}")`}</code>
                        </pre>
                    </CardContent>
                </Card>
            </section>
        </div>
    )
}
