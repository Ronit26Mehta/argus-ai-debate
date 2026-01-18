import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { CheckCircle2, Zap, Cloud, Server } from "lucide-react"

export default function LLMProvidersPage() {
    return (
        <div className="space-y-8">
            <div>
                <h1 className="text-4xl font-bold mb-4">LLM Providers</h1>
                <p className="text-lg text-muted-foreground">
                    ARGUS supports 27+ LLM providers through a unified interface. Use any provider seamlessly with the same API.
                </p>
            </div>

            {/* Overview */}
            <section className="space-y-4">
                <h2 className="text-2xl font-semibold">Overview</h2>
                <div className="grid md:grid-cols-3 gap-4">
                    <Card>
                        <CardHeader>
                            <Cloud className="h-8 w-8 text-primary mb-2" />
                            <CardTitle>Cloud APIs</CardTitle>
                            <CardDescription>
                                20+ cloud-based LLM providers
                            </CardDescription>
                        </CardHeader>
                    </Card>
                    <Card>
                        <CardHeader>
                            <Server className="h-8 w-8 text-primary mb-2" />
                            <CardTitle>Local Models</CardTitle>
                            <CardDescription>
                                Run models locally with Ollama
                            </CardDescription>
                        </CardHeader>
                    </Card>
                    <Card>
                        <CardHeader>
                            <Zap className="h-8 w-8 text-primary mb-2" />
                            <CardTitle>Unified API</CardTitle>
                            <CardDescription>
                                Same interface for all providers
                            </CardDescription>
                        </CardHeader>
                    </Card>
                </div>
            </section>

            {/* Provider List */}
            <section className="space-y-4">
                <h2 className="text-2xl font-semibold">Supported Providers</h2>

                <div className="overflow-x-auto">
                    <table className="w-full border-collapse">
                        <thead>
                            <tr className="border-b">
                                <th className="text-left p-3">Provider</th>
                                <th className="text-left p-3">Models</th>
                                <th className="text-left p-3">Features</th>
                                <th className="text-left p-3">API Key</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr className="border-b">
                                <td className="p-3 font-semibold">OpenAI</td>
                                <td className="p-3">GPT-4o, GPT-4, o1</td>
                                <td className="p-3">
                                    <div className="flex gap-1">
                                        <Badge variant="secondary">Generate</Badge>
                                        <Badge variant="secondary">Stream</Badge>
                                        <Badge variant="secondary">Embed</Badge>
                                    </div>
                                </td>
                                <td className="p-3"><code className="text-xs">OPENAI_API_KEY</code></td>
                            </tr>
                            <tr className="border-b">
                                <td className="p-3 font-semibold">Anthropic</td>
                                <td className="p-3">Claude 3.5 Sonnet, Opus</td>
                                <td className="p-3">
                                    <div className="flex gap-1">
                                        <Badge variant="secondary">Generate</Badge>
                                        <Badge variant="secondary">Stream</Badge>
                                    </div>
                                </td>
                                <td className="p-3"><code className="text-xs">ANTHROPIC_API_KEY</code></td>
                            </tr>
                            <tr className="border-b">
                                <td className="p-3 font-semibold">Google</td>
                                <td className="p-3">Gemini 1.5 Pro/Flash</td>
                                <td className="p-3">
                                    <div className="flex gap-1">
                                        <Badge variant="secondary">Generate</Badge>
                                        <Badge variant="secondary">Stream</Badge>
                                        <Badge variant="secondary">Embed</Badge>
                                    </div>
                                </td>
                                <td className="p-3"><code className="text-xs">GOOGLE_API_KEY</code></td>
                            </tr>
                            <tr className="border-b">
                                <td className="p-3 font-semibold">Ollama</td>
                                <td className="p-3">Llama 3.2, Mistral, Phi</td>
                                <td className="p-3">
                                    <div className="flex gap-1">
                                        <Badge variant="secondary">Local</Badge>
                                        <Badge variant="secondary">Free</Badge>
                                    </div>
                                </td>
                                <td className="p-3">None (local)</td>
                            </tr>
                            <tr className="border-b">
                                <td className="p-3 font-semibold">Cohere</td>
                                <td className="p-3">Command R, R+</td>
                                <td className="p-3">
                                    <div className="flex gap-1">
                                        <Badge variant="secondary">Generate</Badge>
                                        <Badge variant="secondary">Embed</Badge>
                                    </div>
                                </td>
                                <td className="p-3"><code className="text-xs">COHERE_API_KEY</code></td>
                            </tr>
                            <tr className="border-b">
                                <td className="p-3 font-semibold">Mistral</td>
                                <td className="p-3">Large, Small, Codestral</td>
                                <td className="p-3">
                                    <div className="flex gap-1">
                                        <Badge variant="secondary">Generate</Badge>
                                        <Badge variant="secondary">Embed</Badge>
                                    </div>
                                </td>
                                <td className="p-3"><code className="text-xs">MISTRAL_API_KEY</code></td>
                            </tr>
                            <tr className="border-b">
                                <td className="p-3 font-semibold">Groq</td>
                                <td className="p-3">Llama 3.1 70B (ultra-fast)</td>
                                <td className="p-3">
                                    <div className="flex gap-1">
                                        <Badge variant="secondary">Generate</Badge>
                                        <Badge variant="secondary">Stream</Badge>
                                    </div>
                                </td>
                                <td className="p-3"><code className="text-xs">GROQ_API_KEY</code></td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <p className="text-sm text-muted-foreground">
                    + 20 more providers including DeepSeek, xAI, Perplexity, Together, Fireworks, NVIDIA, Azure OpenAI, AWS Bedrock, Vertex AI, and more.
                </p>
            </section>

            {/* Usage Examples */}
            <section className="space-y-4">
                <h2 className="text-2xl font-semibold">Usage Examples</h2>

                <Tabs defaultValue="openai" className="w-full">
                    <TabsList className="grid w-full grid-cols-4">
                        <TabsTrigger value="openai">OpenAI</TabsTrigger>
                        <TabsTrigger value="anthropic">Anthropic</TabsTrigger>
                        <TabsTrigger value="gemini">Google</TabsTrigger>
                        <TabsTrigger value="ollama">Ollama</TabsTrigger>
                    </TabsList>

                    <TabsContent value="openai" className="mt-4">
                        <Card>
                            <CardHeader>
                                <CardTitle>OpenAI Example</CardTitle>
                                <CardDescription>Using GPT-4o with ARGUS</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                                    <code className="text-sm">{`from argus.core.llm import OpenAILLM

llm = OpenAILLM(model="gpt-4o")
response = llm.generate("Explain quantum computing")
print(response.content)

# Streaming
for chunk in llm.stream("Tell me a story"):
    print(chunk, end="", flush=True)`}</code>
                                </pre>
                            </CardContent>
                        </Card>
                    </TabsContent>

                    <TabsContent value="anthropic" className="mt-4">
                        <Card>
                            <CardHeader>
                                <CardTitle>Anthropic Example</CardTitle>
                                <CardDescription>Using Claude 3.5 Sonnet</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                                    <code className="text-sm">{`from argus.core.llm import AnthropicLLM

llm = AnthropicLLM(model="claude-3-5-sonnet-20241022")
response = llm.generate(
    "Analyze this research methodology",
    system_prompt="You are a research expert."
)
print(response.content)`}</code>
                                </pre>
                            </CardContent>
                        </Card>
                    </TabsContent>

                    <TabsContent value="gemini" className="mt-4">
                        <Card>
                            <CardHeader>
                                <CardTitle>Google Gemini Example</CardTitle>
                                <CardDescription>Using Gemini 1.5 Pro</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                                    <code className="text-sm">{`from argus.core.llm import GeminiLLM

llm = GeminiLLM(model="gemini-1.5-pro")
response = llm.generate("Summarize key findings")

# Also supports embeddings
embeddings = llm.embed(["text to embed"])`}</code>
                                </pre>
                            </CardContent>
                        </Card>
                    </TabsContent>

                    <TabsContent value="ollama" className="mt-4">
                        <Card>
                            <CardHeader>
                                <CardTitle>Ollama Example</CardTitle>
                                <CardDescription>Local LLM deployment (free)</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                                    <code className="text-sm">{`from argus.core.llm import OllamaLLM

# Requires Ollama running locally
llm = OllamaLLM(
    model="llama3.1",
    host="http://localhost:11434"
)
response = llm.generate("What is AI?")`}</code>
                                </pre>
                            </CardContent>
                        </Card>
                    </TabsContent>
                </Tabs>
            </section>

            {/* Unified Interface */}
            <section className="space-y-4">
                <h2 className="text-2xl font-semibold">Unified Interface</h2>

                <Card>
                    <CardHeader>
                        <CardTitle>Provider Registry</CardTitle>
                        <CardDescription>
                            Use any provider with the same API via <code>get_llm()</code>
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                            <code className="text-sm">{`from argus.core.llm import get_llm, list_providers

# List available providers
print(list_providers())
# ['openai', 'anthropic', 'gemini', 'ollama', 'cohere', ...]

# Get LLM by provider name
llm = get_llm("groq", model="llama-3.1-70b-versatile")

# Use different providers for different agents
moderator_llm = get_llm("openai", model="gpt-4o")
specialist_llm = get_llm("anthropic", model="claude-3-5-sonnet-20241022")
refuter_llm = get_llm("groq", model="llama-3.1-70b-versatile")
jury_llm = get_llm("gemini", model="gemini-1.5-pro")`}</code>
                        </pre>
                    </CardContent>
                </Card>
            </section>

            {/* Configuration */}
            <section className="space-y-4">
                <h2 className="text-2xl font-semibold">Configuration</h2>

                <Card>
                    <CardHeader>
                        <CardTitle>Environment Variables</CardTitle>
                        <CardDescription>
                            Set up API keys in your <code>.env</code> file
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                            <code className="text-sm">{`# LLM API Keys
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
ARGUS_OLLAMA_HOST=http://localhost:11434`}</code>
                        </pre>
                    </CardContent>
                </Card>
            </section>
        </div>
    )
}
