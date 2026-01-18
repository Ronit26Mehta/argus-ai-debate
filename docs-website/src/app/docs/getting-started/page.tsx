import Link from "next/link"
import { ArrowRight, Package, Terminal, BookOpen } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function GettingStartedPage() {
    return (
        <div className="space-y-8">
            <div>
                <h1 className="text-4xl font-bold mb-4">Getting Started with ARGUS</h1>
                <p className="text-lg text-muted-foreground">
                    Learn how to install and use ARGUS for evidence-based AI reasoning in your projects.
                </p>
            </div>

            {/* Installation */}
            <section className="space-y-4">
                <h2 className="text-2xl font-semibold">Installation</h2>

                <Tabs defaultValue="pip" className="w-full">
                    <TabsList>
                        <TabsTrigger value="pip">PyPI (Recommended)</TabsTrigger>
                        <TabsTrigger value="source">From Source</TabsTrigger>
                    </TabsList>

                    <TabsContent value="pip" className="mt-4">
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Package className="h-5 w-5" />
                                    Install from PyPI
                                </CardTitle>
                                <CardDescription>
                                    The easiest way to get started with ARGUS
                                </CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div>
                                    <p className="text-sm text-muted-foreground mb-2">Basic installation:</p>
                                    <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                                        <code>pip install argus-debate-ai</code>
                                    </pre>
                                </div>

                                <div>
                                    <p className="text-sm text-muted-foreground mb-2">With all features:</p>
                                    <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                                        <code>pip install argus-debate-ai[all]</code>
                                    </pre>
                                </div>

                                <div>
                                    <p className="text-sm text-muted-foreground mb-2">Individual extras:</p>
                                    <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                                        <code>{`pip install argus-debate-ai[ollama]   # Ollama local LLM support
pip install argus-debate-ai[cohere]   # Cohere integration
pip install argus-debate-ai[mistral]  # Mistral integration
pip install argus-debate-ai[groq]     # Groq LPU inference
pip install argus-debate-ai[tools]    # Tool integrations
pip install argus-debate-ai[plotting] # Visualization`}</code>
                                    </pre>
                                </div>
                            </CardContent>
                        </Card>
                    </TabsContent>

                    <TabsContent value="source" className="mt-4">
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Terminal className="h-5 w-5" />
                                    Install from Source
                                </CardTitle>
                                <CardDescription>
                                    For development or the latest features
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                                    <code>{`git clone https://github.com/Ronit26Mehta/argus-ai-debate.git
cd argus
pip install -e ".[dev]"`}</code>
                                </pre>
                            </CardContent>
                        </Card>
                    </TabsContent>
                </Tabs>
            </section>

            {/* System Requirements */}
            <section className="space-y-4">
                <h2 className="text-2xl font-semibold">System Requirements</h2>
                <div className="overflow-x-auto">
                    <table className="w-full border-collapse">
                        <thead>
                            <tr className="border-b">
                                <th className="text-left p-3">Requirement</th>
                                <th className="text-left p-3">Minimum</th>
                                <th className="text-left p-3">Recommended</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr className="border-b">
                                <td className="p-3">Python</td>
                                <td className="p-3">3.11+</td>
                                <td className="p-3">3.12+</td>
                            </tr>
                            <tr className="border-b">
                                <td className="p-3">RAM</td>
                                <td className="p-3">4 GB</td>
                                <td className="p-3">16 GB</td>
                            </tr>
                            <tr className="border-b">
                                <td className="p-3">Storage</td>
                                <td className="p-3">1 GB</td>
                                <td className="p-3">10 GB (with embeddings)</td>
                            </tr>
                            <tr className="border-b">
                                <td className="p-3">GPU</td>
                                <td className="p-3">None</td>
                                <td className="p-3">CUDA-compatible (for local embeddings)</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </section>

            {/* Quick Start */}
            <section className="space-y-4">
                <h2 className="text-2xl font-semibold">Quick Start</h2>

                <Card>
                    <CardHeader>
                        <CardTitle>Your First Debate</CardTitle>
                        <CardDescription>
                            Run a simple debate to evaluate a proposition
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                            <code>{`from argus import RDCOrchestrator, get_llm

# Initialize with any supported LLM
llm = get_llm("openai", model="gpt-4o")

# Run a debate on a proposition
orchestrator = RDCOrchestrator(llm=llm, max_rounds=5)
result = orchestrator.debate(
    "The new treatment reduces symptoms by more than 20%",
    prior=0.5,  # Start with 50/50 uncertainty
)

print(f"Verdict: {result.verdict.label}")
print(f"Posterior: {result.verdict.posterior:.3f}")
print(f"Evidence: {result.num_evidence} items")
print(f"Reasoning: {result.verdict.reasoning}")`}</code>
                        </pre>
                    </CardContent>
                </Card>
            </section>

            {/* Environment Setup */}
            <section className="space-y-4">
                <h2 className="text-2xl font-semibold">Environment Setup</h2>

                <Card>
                    <CardHeader>
                        <CardTitle>API Keys Configuration</CardTitle>
                        <CardDescription>
                            Set up your LLM provider API keys
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <p className="text-sm text-muted-foreground">
                            Create a <code>.env</code> file in your project root:
                        </p>
                        <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                            <code>{`# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
COHERE_API_KEY=...
MISTRAL_API_KEY=...
GROQ_API_KEY=gsk_...

# Default settings
ARGUS_DEFAULT_PROVIDER=openai
ARGUS_DEFAULT_MODEL=gpt-4o
ARGUS_TEMPERATURE=0.7
ARGUS_MAX_TOKENS=4096

# Ollama (local)
ARGUS_OLLAMA_HOST=http://localhost:11434

# Logging
ARGUS_LOG_LEVEL=INFO`}</code>
                        </pre>
                    </CardContent>
                </Card>
            </section>

            {/* Next Steps */}
            <section className="space-y-4">
                <h2 className="text-2xl font-semibold">Next Steps</h2>

                <div className="grid md:grid-cols-2 gap-4">
                    <Card className="hover:border-primary transition-colors">
                        <CardHeader>
                            <BookOpen className="h-8 w-8 text-primary mb-2" />
                            <CardTitle>Core Concepts</CardTitle>
                            <CardDescription>
                                Learn about Research Debate Chain, C-DAG, and multi-agent systems
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <Link href="/docs/core-concepts">
                                <Button variant="ghost" className="w-full justify-between">
                                    Read Core Concepts
                                    <ArrowRight className="h-4 w-4" />
                                </Button>
                            </Link>
                        </CardContent>
                    </Card>

                    <Card className="hover:border-primary transition-colors">
                        <CardHeader>
                            <Package className="h-8 w-8 text-primary mb-2" />
                            <CardTitle>LLM Providers</CardTitle>
                            <CardDescription>
                                Explore 27+ LLM providers and learn how to use them
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <Link href="/docs/llm-providers">
                                <Button variant="ghost" className="w-full justify-between">
                                    View Providers
                                    <ArrowRight className="h-4 w-4" />
                                </Button>
                            </Link>
                        </CardContent>
                    </Card>
                </div>
            </section>
        </div>
    )
}
