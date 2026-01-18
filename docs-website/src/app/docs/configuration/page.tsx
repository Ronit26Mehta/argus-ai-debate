import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'
import { Callout } from '@/components/Callout'

export const metadata: Metadata = {
    title: 'Configuration | ARGUS Documentation',
    description: 'Complete configuration guide for ARGUS - environment variables, config files, and programmatic setup',
}

export default function ConfigurationPage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Configuration
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Configure ARGUS with environment variables, config files, or programmatically.
                    </p>
                </div>

                {/* Environment Variables */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Environment Variables</h2>
                    <p className="text-muted-foreground">
                        The easiest way to configure ARGUS is through environment variables:
                    </p>

                    <CodeBlock
                        code={`# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
COHERE_API_KEY=...
MISTRAL_API_KEY=...
GROQ_API_KEY=gsk_...

# Embedding Providers
VOYAGE_API_KEY=...
JINA_API_KEY=...

# Tool APIs
TAVILY_API_KEY=tvly-...
BRAVE_API_KEY=BSA...
GITHUB_TOKEN=ghp_...

# Local Options
OLLAMA_HOST=http://localhost:11434

# Default Settings
ARGUS_DEFAULT_PROVIDER=openai
ARGUS_DEFAULT_MODEL=gpt-4o
ARGUS_TEMPERATURE=0.7
ARGUS_MAX_TOKENS=4096

# Logging
ARGUS_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR`}
                        language="bash"
                        filename=".env"
                    />

                    <Callout variant="tip" title="Using .env Files">
                        Create a <code>.env</code> file in your project root. ARGUS automatically loads it using python-dotenv.
                    </Callout>
                </section>

                {/* Configuration File */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Configuration File</h2>
                    <p className="text-muted-foreground">
                        For complex setups, use a YAML configuration file at <code>~/.argus/config.yaml</code>:
                    </p>

                    <CodeBlock
                        code={`# Default LLM settings
default_provider: openai
default_model: gpt-4o
temperature: 0.7
max_tokens: 4096

# LLM credentials (prefer env vars for sensitive data)
llm:
  openai_api_key: \${OPENAI_API_KEY}
  anthropic_api_key: \${ANTHROPIC_API_KEY}
  google_api_key: \${GOOGLE_API_KEY}
  ollama_host: http://localhost:11434

# Debate settings
debate:
  max_rounds: 5
  min_evidence: 3
  convergence_threshold: 0.01
  
# Retrieval settings  
retrieval:
  embedding_model: all-MiniLM-L6-v2
  lambda_param: 0.7
  use_reranker: true
  reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
  
# Chunking settings
chunking:
  chunk_size: 512
  chunk_overlap: 50
  strategy: recursive  # sentence, recursive, semantic

# Provenance
provenance:
  enabled: true
  hash_algorithm: sha256
  
# Metrics
metrics:
  enabled: true
  export_format: json  # json, prometheus`}
                        language="yaml"
                        filename="~/.argus/config.yaml"
                    />
                </section>

                {/* Programmatic Configuration */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Programmatic Configuration</h2>
                    <p className="text-muted-foreground">
                        Configure ARGUS directly in your Python code:
                    </p>

                    <CodeBlock
                        code={`from argus import ArgusConfig, get_config

# Create custom config
config = ArgusConfig(
    default_provider="anthropic",
    default_model="claude-3-5-sonnet-20241022",
    temperature=0.5,
    max_tokens=4096,
)

# Or get global config (from env vars and config file)
config = get_config()

# Access nested config
print(config.chunking.chunk_size)
print(config.llm.openai_api_key)`}
                        language="python"
                    />
                </section>

                {/* LLM Provider Configuration */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">LLM Provider Configuration</h2>

                    <div className="space-y-6">
                        <div>
                            <h3 className="text-xl font-semibold mb-3">OpenAI</h3>
                            <CodeBlock
                                code={`from argus.core.llm import OpenAILLM

llm = OpenAILLM(
    model="gpt-4o",
    api_key="sk-...",  # Or use OPENAI_API_KEY env var
    temperature=0.7,
    max_tokens=4096,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
)`}
                                language="python"
                            />
                        </div>

                        <div>
                            <h3 className="text-xl font-semibold mb-3">Anthropic Claude</h3>
                            <CodeBlock
                                code={`from argus.core.llm import AnthropicLLM

llm = AnthropicLLM(
    model="claude-3-5-sonnet-20241022",
    api_key="sk-ant-...",  # Or use ANTHROPIC_API_KEY
    temperature=0.7,
    max_tokens=4096,
)`}
                                language="python"
                            />
                        </div>

                        <div>
                            <h3 className="text-xl font-semibold mb-3">Ollama (Local)</h3>
                            <CodeBlock
                                code={`from argus.core.llm import OllamaLLM

llm = OllamaLLM(
    model="llama3.2",
    host="http://localhost:11434",  # Or use OLLAMA_HOST
    temperature=0.7,
)`}
                                language="python"
                            />
                        </div>
                    </div>
                </section>

                {/* Debate Settings */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Debate Settings</h2>
                    <p className="text-muted-foreground">
                        Configure debate orchestration parameters:
                    </p>

                    <CodeBlock
                        code={`from argus import RDCOrchestrator, get_llm

llm = get_llm("openai", model="gpt-4o")

orchestrator = RDCOrchestrator(
    llm=llm,
    max_rounds=5,              # Maximum debate rounds
    min_evidence=3,            # Minimum evidence per side
    convergence_threshold=0.01, # Stop if posterior changes < 1%
    budget=100,                # Token budget (optional)
    enable_provenance=True,    # Track all operations
)`}
                        language="python"
                    />
                </section>

                {/* Retrieval Settings */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Retrieval Settings</h2>
                    <p className="text-muted-foreground">
                        Configure hybrid retrieval and reranking:
                    </p>

                    <CodeBlock
                        code={`from argus.retrieval import HybridRetriever

retriever = HybridRetriever(
    embedding_model="all-MiniLM-L6-v2",  # Local or cloud
    lambda_param=0.7,          # Weight: 0=BM25 only, 1=dense only
    use_reranker=True,         # Enable cross-encoder reranking
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k=10,                  # Initial retrieval count
    rerank_top_k=5,            # Final reranked count
)`}
                        language="python"
                    />
                </section>

                {/* Chunking Strategies */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Chunking Strategies</h2>
                    <p className="text-muted-foreground">
                        Configure document chunking:
                    </p>

                    <CodeBlock
                        code={`from argus.knowledge import Chunker, ChunkingStrategy

# Recursive chunking (default)
chunker = Chunker(
    chunk_size=512,
    chunk_overlap=50,
    strategy=ChunkingStrategy.RECURSIVE,
)

# Sentence-based chunking
chunker = Chunker(
    chunk_size=512,
    chunk_overlap=50,
    strategy=ChunkingStrategy.SENTENCE,
)

# Semantic chunking (experimental)
chunker = Chunker(
    chunk_size=512,
    chunk_overlap=50,
    strategy=ChunkingStrategy.SEMANTIC,
    embedding_model="all-MiniLM-L6-v2",
)`}
                        language="python"
                    />
                </section>

                {/* Configuration Validation */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Configuration Validation</h2>
                    <p className="text-muted-foreground">
                        Validate your configuration and API keys:
                    </p>

                    <CodeBlock
                        code={`# CLI validation
argus config validate

# Programmatic validation
from argus import get_config

config = get_config()
is_valid, errors = config.validate()

if not is_valid:
    for error in errors:
        print(f"Error: {error}")
else:
    print("Configuration is valid!")`}
                        language="python"
                    />
                </section>

                {/* Best Practices */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Best Practices</h2>

                    <Callout variant="tip" title="Security">
                        <ul className="list-disc list-inside space-y-2">
                            <li>Never commit API keys to version control</li>
                            <li>Use environment variables or secure vaults</li>
                            <li>Rotate keys regularly</li>
                            <li>Use different keys for dev/staging/prod</li>
                        </ul>
                    </Callout>

                    <Callout variant="info" title="Performance">
                        <ul className="list-disc list-inside space-y-2">
                            <li>Use local embeddings for development (free, fast)</li>
                            <li>Enable reranking for better retrieval quality</li>
                            <li>Adjust chunk size based on your documents</li>
                            <li>Set appropriate convergence thresholds</li>
                        </ul>
                    </Callout>

                    <Callout variant="warning" title="Cost Management">
                        <ul className="list-disc list-inside space-y-2">
                            <li>Set token budgets to prevent runaway costs</li>
                            <li>Use cheaper models for development</li>
                            <li>Monitor API usage regularly</li>
                            <li>Consider local models (Ollama) for testing</li>
                        </ul>
                    </Callout>
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-2">
                        <a
                            href="/docs/llm-providers"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">LLM Providers →</h3>
                            <p className="text-sm text-muted-foreground">
                                Explore all 27+ supported LLM providers
                            </p>
                        </a>
                        <a
                            href="/docs/core-concepts/rdc"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">Core Concepts →</h3>
                            <p className="text-sm text-muted-foreground">
                                Learn about RDC, C-DAG, and multi-agent systems
                            </p>
                        </a>
                    </div>
                </section>
            </div>
        </div>
    )
}
