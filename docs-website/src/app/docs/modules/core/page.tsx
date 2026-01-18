import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'
import { Callout } from '@/components/Callout'

export const metadata: Metadata = {
    title: 'Core Module | ARGUS Documentation',
    description: 'Complete guide to the core module - configuration, LLM integrations, and shared utilities',
}

export default function CoreModulePage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Core Module
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Configuration, LLM integrations (27+ providers), and shared utilities.
                    </p>
                </div>

                {/* LLM Providers */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">LLM Providers (27+)</h2>
                    <Callout variant="info">
                        ARGUS supports 27+ LLM providers including OpenAI, Anthropic, Google, Ollama, Cohere, Mistral, Groq, and more.
                    </Callout>

                    <div className="space-y-6">
                        <div>
                            <h3 className="text-xl font-semibold mb-3">Using get_llm Helper</h3>
                            <CodeBlock
                                code={`from argus import get_llm

# OpenAI
llm = get_llm("openai", model="gpt-4o")

# Anthropic
llm = get_llm("anthropic", model="claude-3-5-sonnet-20241022")

# Google Gemini
llm = get_llm("gemini", model="gemini-1.5-pro")

# Ollama (local)
llm = get_llm("ollama", model="llama3.2")

# Groq (fast)
llm = get_llm("groq", model="llama-3.1-70b-versatile")`}
                                language="python"
                            />
                        </div>

                        <div>
                            <h3 className="text-xl font-semibold mb-3">Direct Provider Usage</h3>
                            <CodeBlock
                                code={`from argus.core.llm import OpenAILLM, AnthropicLLM

# OpenAI with custom config
llm = OpenAILLM(
    model="gpt-4o",
    api_key="sk-...",
    temperature=0.7,
    max_tokens=4096,
    top_p=1.0
)

# Generate
response = llm.generate("What is AI?")
print(response)

# Stream
for chunk in llm.stream("Tell me about quantum computing"):
    print(chunk, end="", flush=True)`}
                                language="python"
                            />
                        </div>
                    </div>
                </section>

                {/* Configuration */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Configuration</h2>
                    <CodeBlock
                        code={`from argus import ArgusConfig, get_config

# Create custom config
config = ArgusConfig(
    default_provider="openai",
    default_model="gpt-4o",
    temperature=0.7,
    max_tokens=4096
)

# Get global config
config = get_config()

# Access settings
print(config.default_provider)
print(config.llm.openai_api_key)
print(config.chunking.chunk_size)`}
                        language="python"
                    />
                </section>

                {/* Data Models */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Core Data Models</h2>
                    <CodeBlock
                        code={`from argus.core.models import Document, Chunk, Embedding, Evidence

# Document
doc = Document(
    content="Full document text...",
    metadata={"title": "Paper Title", "source": "arxiv"}
)

# Chunk
chunk = Chunk(
    text="Chunk text...",
    doc_id=doc.id,
    chunk_index=0,
    metadata={"page": 1}
)

# Embedding
embedding = Embedding(
    vector=[0.1, 0.2, ...],
    model="all-MiniLM-L6-v2",
    dimension=384
)

# Evidence
evidence = Evidence(
    text="Supporting evidence...",
    polarity=1,
    confidence=0.9,
    source="https://..."
)`}
                        language="python"
                    />
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a href="/docs/configuration" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Configuration →</h3>
                            <p className="text-sm text-muted-foreground">Configure ARGUS</p>
                        </a>
                        <a href="/docs/llm-providers" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">LLM Providers →</h3>
                            <p className="text-sm text-muted-foreground">All 27+ providers</p>
                        </a>
                        <a href="/docs/modules/embeddings" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Embeddings →</h3>
                            <p className="text-sm text-muted-foreground">Embedding models</p>
                        </a>
                    </div>
                </section>
            </div>
        </div>
    )
}
