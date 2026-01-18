import Link from "next/link"
import { ArrowRight, Github, Package, Zap, Shield, Network, Brain, Database, Code2, CheckCircle2, TrendingUp } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function HomePage() {
    return (
        <div className="flex flex-col">
            {/* Hero Section */}
            <section className="relative overflow-hidden py-20 md:py-32">
                <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 dark:from-blue-950/20 dark:via-purple-950/20 dark:to-pink-950/20 animate-gradient-shift" style={{ backgroundSize: '200% 200%' }} />

                <div className="container relative z-10">
                    <div className="mx-auto max-w-4xl text-center space-y-8 animate-fade-in">
                        <Badge className="text-sm px-4 py-1.5">
                            <TrendingUp className="h-3 w-3 mr-1" />
                            v1.4.2 - Production Ready
                        </Badge>

                        <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold tracking-tight">
                            <span className="gradient-text">Multi-Agent AI Debate</span>
                            <br />
                            Framework for Evidence-Based Reasoning
                        </h1>

                        <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto">
                            ARGUS implements Research Debate Chain (RDC) - a novel approach to AI reasoning that structures knowledge evaluation as multi-agent debates with Bayesian aggregation and full provenance tracking.
                        </p>

                        <div className="flex flex-col sm:flex-row gap-4 justify-center">
                            <Link href="/docs/getting-started">
                                <Button size="lg" className="text-base px-8">
                                    Get Started
                                    <ArrowRight className="ml-2 h-5 w-5" />
                                </Button>
                            </Link>

                            <Link href="https://github.com/Ronit26Mehta/argus-ai-debate" target="_blank">
                                <Button size="lg" variant="outline" className="text-base px-8">
                                    <Github className="mr-2 h-5 w-5" />
                                    View on GitHub
                                </Button>
                            </Link>
                        </div>

                        {/* Installation */}
                        <div className="mt-8 glass rounded-lg p-6 max-w-2xl mx-auto">
                            <p className="text-sm text-muted-foreground mb-2">Install via pip:</p>
                            <code className="text-sm md:text-base font-mono bg-muted px-4 py-2 rounded block">
                                pip install argus-debate-ai
                            </code>
                        </div>
                    </div>
                </div>
            </section>

            {/* Stats Section */}
            <section className="py-12 border-y bg-muted/50">
                <div className="container">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
                        <div>
                            <div className="text-3xl md:text-4xl font-bold gradient-text">27+</div>
                            <div className="text-sm text-muted-foreground mt-1">LLM Providers</div>
                        </div>
                        <div>
                            <div className="text-3xl md:text-4xl font-bold gradient-text">16</div>
                            <div className="text-sm text-muted-foreground mt-1">Embedding Models</div>
                        </div>
                        <div>
                            <div className="text-3xl md:text-4xl font-bold gradient-text">19+</div>
                            <div className="text-sm text-muted-foreground mt-1">Tool Integrations</div>
                        </div>
                        <div>
                            <div className="text-3xl md:text-4xl font-bold gradient-text">17</div>
                            <div className="text-sm text-muted-foreground mt-1">Core Modules</div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Key Features */}
            <section className="py-20">
                <div className="container">
                    <div className="text-center space-y-4 mb-12">
                        <h2 className="text-3xl md:text-4xl font-bold">Why Choose ARGUS?</h2>
                        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
                            Traditional LLMs suffer from hallucination and overconfidence. ARGUS addresses these through structured multi-agent debates.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                        <Card className="border-2 hover:border-primary transition-colors">
                            <CardHeader>
                                <Network className="h-10 w-10 text-primary mb-2" />
                                <CardTitle>Conceptual Debate Graph (C-DAG)</CardTitle>
                                <CardDescription>
                                    Directed graph structure where propositions, evidence, and rebuttals are nodes with signed edges representing support/attack relationships.
                                </CardDescription>
                            </CardHeader>
                        </Card>

                        <Card className="border-2 hover:border-primary transition-colors">
                            <CardHeader>
                                <Brain className="h-10 w-10 text-primary mb-2" />
                                <CardTitle>Multi-Agent Orchestration</CardTitle>
                                <CardDescription>
                                    Moderator, Specialist, Refuter, and Jury agents work together to gather evidence, challenge claims, and render verdicts through Bayesian aggregation.
                                </CardDescription>
                            </CardHeader>
                        </Card>

                        <Card className="border-2 hover:border-primary transition-colors">
                            <CardHeader>
                                <Zap className="h-10 w-10 text-primary mb-2" />
                                <CardTitle>Value of Information Planning</CardTitle>
                                <CardDescription>
                                    Decision-theoretic experiment selection using Expected Information Gain (EIG) to prioritize high-value evidence gathering.
                                </CardDescription>
                            </CardHeader>
                        </Card>

                        <Card className="border-2 hover:border-primary transition-colors">
                            <CardHeader>
                                <Shield className="h-10 w-10 text-primary mb-2" />
                                <CardTitle>Full Provenance Tracking</CardTitle>
                                <CardDescription>
                                    PROV-O compatible ledger with hash-chain integrity, cryptographic attestations, and complete audit trails for every claim.
                                </CardDescription>
                            </CardHeader>
                        </Card>

                        <Card className="border-2 hover:border-primary transition-colors">
                            <CardHeader>
                                <Database className="h-10 w-10 text-primary mb-2" />
                                <CardTitle>Hybrid Retrieval System</CardTitle>
                                <CardDescription>
                                    Combines BM25 sparse retrieval with FAISS dense search and cross-encoder reranking for optimal evidence discovery.
                                </CardDescription>
                            </CardHeader>
                        </Card>

                        <Card className="border-2 hover:border-primary transition-colors">
                            <CardHeader>
                                <Code2 className="h-10 w-10 text-primary mb-2" />
                                <CardTitle>27+ LLM Providers</CardTitle>
                                <CardDescription>
                                    Unified interface for OpenAI, Anthropic, Google, Ollama, Cohere, Mistral, Groq, and 20+ more providers with seamless switching.
                                </CardDescription>
                            </CardHeader>
                        </Card>
                    </div>
                </div>
            </section>

            {/* Quick Start Example */}
            <section className="py-20 bg-muted/50">
                <div className="container">
                    <div className="text-center space-y-4 mb-12">
                        <h2 className="text-3xl md:text-4xl font-bold">Quick Start</h2>
                        <p className="text-lg text-muted-foreground">
                            Get started with ARGUS in minutes
                        </p>
                    </div>

                    <Tabs defaultValue="basic" className="max-w-4xl mx-auto">
                        <TabsList className="grid w-full grid-cols-3">
                            <TabsTrigger value="basic">Basic Usage</TabsTrigger>
                            <TabsTrigger value="debate">Multi-Agent Debate</TabsTrigger>
                            <TabsTrigger value="retrieval">Hybrid Retrieval</TabsTrigger>
                        </TabsList>

                        <TabsContent value="basic" className="mt-6">
                            <Card>
                                <CardHeader>
                                    <CardTitle>Run Your First Debate</CardTitle>
                                    <CardDescription>
                                        Evaluate a proposition using the RDC orchestrator
                                    </CardDescription>
                                </CardHeader>
                                <CardContent>
                                    <pre className="bg-background p-4 rounded-lg overflow-x-auto">
                                        <code className="text-sm">{`from argus import RDCOrchestrator, get_llm

# Initialize with any supported LLM
llm = get_llm("openai", model="gpt-4o")

# Run a debate
orchestrator = RDCOrchestrator(llm=llm, max_rounds=5)
result = orchestrator.debate(
    "The new treatment reduces symptoms by more than 20%",
    prior=0.5,
)

print(f"Verdict: {result.verdict.label}")
print(f"Posterior: {result.verdict.posterior:.3f}")
print(f"Evidence: {result.num_evidence} items")`}</code>
                                    </pre>
                                </CardContent>
                            </Card>
                        </TabsContent>

                        <TabsContent value="debate" className="mt-6">
                            <Card>
                                <CardHeader>
                                    <CardTitle>Custom Agent Pipeline</CardTitle>
                                    <CardDescription>
                                        Use different LLM providers for different agents
                                    </CardDescription>
                                </CardHeader>
                                <CardContent>
                                    <pre className="bg-background p-4 rounded-lg overflow-x-auto">
                                        <code className="text-sm">{`from argus import get_llm, CDAG, Proposition
from argus.agents import Moderator, Specialist, Refuter, Jury

# Different models for different tasks
moderator = Moderator(get_llm("openai", model="gpt-4o"))
specialist = Specialist(get_llm("anthropic", model="claude-3-5-sonnet-20241022"))
refuter = Refuter(get_llm("groq", model="llama-3.1-70b-versatile"))
jury = Jury(get_llm("gemini", model="gemini-1.5-pro"))

# Create debate
graph = CDAG()
prop = Proposition(text="Carbon pricing reduces emissions", prior=0.5)
graph.add_proposition(prop)

# Run debate rounds
for round_num in range(3):
    evidence = specialist.gather_evidence(graph, prop.id)
    rebuttals = refuter.generate_rebuttals(graph, prop.id)
    if moderator.should_stop(graph, prop.id):
        break

verdict = jury.evaluate(graph, prop.id)
print(f"Verdict: {verdict.label} (posterior={verdict.posterior:.3f})")`}</code>
                                    </pre>
                                </CardContent>
                            </Card>
                        </TabsContent>

                        <TabsContent value="retrieval" className="mt-6">
                            <Card>
                                <CardHeader>
                                    <CardTitle>Hybrid Retrieval</CardTitle>
                                    <CardDescription>
                                        Combine BM25 and dense search with reranking
                                    </CardDescription>
                                </CardHeader>
                                <CardContent>
                                    <pre className="bg-background p-4 rounded-lg overflow-x-auto">
                                        <code className="text-sm">{`from argus.retrieval import HybridRetriever
from argus.knowledge import DocumentLoader, Chunker

# Load and chunk documents
loader = DocumentLoader()
doc = loader.load("research_paper.pdf")

chunker = Chunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk(doc)

# Create hybrid retriever
retriever = HybridRetriever(
    embedding_model="all-MiniLM-L6-v2",
    lambda_param=0.7,  # Weight toward dense retrieval
    use_reranker=True,
)
retriever.index_chunks(chunks)

# Search
results = retriever.retrieve("treatment efficacy results", top_k=10)
for r in results:
    print(f"[{r.rank}] Score: {r.score:.3f} - {r.chunk.text[:100]}...")`}</code>
                                    </pre>
                                </CardContent>
                            </Card>
                        </TabsContent>
                    </Tabs>
                </div>
            </section>

            {/* Comparison Table */}
            <section className="py-20">
                <div className="container">
                    <div className="text-center space-y-4 mb-12">
                        <h2 className="text-3xl md:text-4xl font-bold">How ARGUS Compares</h2>
                        <p className="text-lg text-muted-foreground">
                            See how ARGUS stacks up against other frameworks
                        </p>
                    </div>

                    <div className="overflow-x-auto">
                        <table className="w-full border-collapse">
                            <thead>
                                <tr className="border-b">
                                    <th className="text-left p-4 font-semibold">Feature</th>
                                    <th className="text-center p-4 font-semibold">ARGUS</th>
                                    <th className="text-center p-4 font-semibold">LangChain</th>
                                    <th className="text-center p-4 font-semibold">LangGraph</th>
                                    <th className="text-center p-4 font-semibold">AutoGen</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr className="border-b">
                                    <td className="p-4">Multi-Agent Debates</td>
                                    <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                    <td className="text-center p-4 text-muted-foreground">Partial</td>
                                    <td className="text-center p-4 text-muted-foreground">Partial</td>
                                    <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                </tr>
                                <tr className="border-b">
                                    <td className="p-4">Bayesian Reasoning</td>
                                    <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                    <td className="text-center p-4 text-muted-foreground">-</td>
                                    <td className="text-center p-4 text-muted-foreground">-</td>
                                    <td className="text-center p-4 text-muted-foreground">-</td>
                                </tr>
                                <tr className="border-b">
                                    <td className="p-4">Provenance Tracking</td>
                                    <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                    <td className="text-center p-4 text-muted-foreground">Basic</td>
                                    <td className="text-center p-4 text-muted-foreground">Basic</td>
                                    <td className="text-center p-4 text-muted-foreground">-</td>
                                </tr>
                                <tr className="border-b">
                                    <td className="p-4">Value of Information</td>
                                    <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                    <td className="text-center p-4 text-muted-foreground">-</td>
                                    <td className="text-center p-4 text-muted-foreground">-</td>
                                    <td className="text-center p-4 text-muted-foreground">-</td>
                                </tr>
                                <tr className="border-b">
                                    <td className="p-4">27+ LLM Providers</td>
                                    <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                    <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                    <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                    <td className="text-center p-4 text-muted-foreground">Limited</td>
                                </tr>
                                <tr className="border-b">
                                    <td className="p-4">Hybrid Retrieval</td>
                                    <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                    <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                    <td className="text-center p-4 text-muted-foreground">-</td>
                                    <td className="text-center p-4 text-muted-foreground">-</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>

                    <div className="text-center mt-8">
                        <Link href="/comparison">
                            <Button variant="outline">
                                View Detailed Comparison
                                <ArrowRight className="ml-2 h-4 w-4" />
                            </Button>
                        </Link>
                    </div>
                </div>
            </section>

            {/* Use Cases */}
            <section className="py-20">
                <div className="container">
                    <div className="text-center space-y-4 mb-12">
                        <h2 className="text-3xl md:text-4xl font-bold">Real-World Use Cases</h2>
                        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
                            ARGUS is being used across industries for evidence-based decision making
                        </p>
                    </div>

                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                        <Card className="border-2 hover:shadow-xl transition-all card-hover">
                            <CardHeader>
                                <div className="h-12 w-12 rounded-lg bg-blue-100 dark:bg-blue-950 flex items-center justify-center mb-4">
                                    <span className="text-2xl">🏥</span>
                                </div>
                                <CardTitle>Clinical Research</CardTitle>
                                <CardDescription>
                                    Evaluate treatment efficacy claims by analyzing clinical trial data, systematic reviews, and meta-analyses with multi-agent debates.
                                </CardDescription>
                            </CardHeader>
                        </Card>

                        <Card className="border-2 hover:shadow-xl transition-all card-hover">
                            <CardHeader>
                                <div className="h-12 w-12 rounded-lg bg-purple-100 dark:bg-purple-950 flex items-center justify-center mb-4">
                                    <span className="text-2xl">🔬</span>
                                </div>
                                <CardTitle>Scientific Fact-Checking</CardTitle>
                                <CardDescription>
                                    Verify research claims by fetching papers from arXiv, CrossRef, and other sources, then running structured debates with provenance.
                                </CardDescription>
                            </CardHeader>
                        </Card>

                        <Card className="border-2 hover:shadow-xl transition-all card-hover">
                            <CardHeader>
                                <div className="h-12 w-12 rounded-lg bg-pink-100 dark:bg-pink-950 flex items-center justify-center mb-4">
                                    <span className="text-2xl">💼</span>
                                </div>
                                <CardTitle>Financial Analysis</CardTitle>
                                <CardDescription>
                                    Analyze SEC filings, earnings reports, and market data with multi-specialist agents for investment decisions.
                                </CardDescription>
                            </CardHeader>
                        </Card>

                        <Card className="border-2 hover:shadow-xl transition-all card-hover">
                            <CardHeader>
                                <div className="h-12 w-12 rounded-lg bg-green-100 dark:bg-green-950 flex items-center justify-center mb-4">
                                    <span className="text-2xl">⚖️</span>
                                </div>
                                <CardTitle>Policy Analysis</CardTitle>
                                <CardDescription>
                                    Evaluate policy proposals through structured argumentation with evidence from multiple sources and stakeholder perspectives.
                                </CardDescription>
                            </CardHeader>
                        </Card>

                        <Card className="border-2 hover:shadow-xl transition-all card-hover">
                            <CardHeader>
                                <div className="h-12 w-12 rounded-lg bg-orange-100 dark:bg-orange-950 flex items-center justify-center mb-4">
                                    <span className="text-2xl">📰</span>
                                </div>
                                <CardTitle>News Verification</CardTitle>
                                <CardDescription>
                                    Combat misinformation by verifying news claims with evidence-based debates and credibility scoring.
                                </CardDescription>
                            </CardHeader>
                        </Card>

                        <Card className="border-2 hover:shadow-xl transition-all card-hover">
                            <CardHeader>
                                <div className="h-12 w-12 rounded-lg bg-indigo-100 dark:bg-indigo-950 flex items-center justify-center mb-4">
                                    <span className="text-2xl">🎓</span>
                                </div>
                                <CardTitle>Academic Research</CardTitle>
                                <CardDescription>
                                    Literature review automation, hypothesis testing, and research gap identification through systematic evidence analysis.
                                </CardDescription>
                            </CardHeader>
                        </Card>
                    </div>
                </div>
            </section>

            {/* CTA Section */}
            <section className="py-20 bg-gradient-to-br from-blue-600 via-purple-600 to-pink-600 text-white">
                <div className="container text-center space-y-8">
                    <h2 className="text-3xl md:text-4xl font-bold">Ready to Get Started?</h2>
                    <p className="text-lg md:text-xl max-w-2xl mx-auto opacity-90">
                        Join developers using ARGUS for evidence-based AI reasoning in production
                    </p>
                    <div className="flex flex-col sm:flex-row gap-4 justify-center">
                        <Link href="/docs/getting-started">
                            <Button size="lg" variant="secondary" className="text-base px-8">
                                Read the Docs
                                <ArrowRight className="ml-2 h-5 w-5" />
                            </Button>
                        </Link>
                        <Link href="https://pypi.org/project/argus-debate-ai/" target="_blank">
                            <Button size="lg" variant="outline" className="text-base px-8 bg-transparent border-white text-white hover:bg-white/10">
                                <Package className="mr-2 h-5 w-5" />
                                Install from PyPI
                            </Button>
                        </Link>
                    </div>
                </div>
            </section>
        </div>
    )
}
