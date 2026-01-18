import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'
import { Callout } from '@/components/Callout'
import { ComparisonTable } from '@/components/ComparisonTable'

export const metadata: Metadata = {
    title: 'Embedding Providers | ARGUS Documentation',
    description: 'Complete guide to 16+ embedding providers in ARGUS - local and cloud options',
}

export default function EmbeddingProvidersPage() {
    const providers = [
        { name: 'SentenceTransformers', models: 'all-MiniLM-L6-v2, all-mpnet-base-v2', dimensions: '384-768', apiKey: 'None (Free)', cost: 'Free' },
        { name: 'FastEmbed', models: 'BAAI/bge-small-en', dimensions: '384-768', apiKey: 'None (Free)', cost: 'Free' },
        { name: 'Ollama', models: 'nomic-embed-text', dimensions: '768-1024', apiKey: 'None (Free)', cost: 'Free' },
        { name: 'OpenAI', models: 'text-embedding-3-small/large', dimensions: '1536-3072', apiKey: 'OPENAI_API_KEY', cost: '$0.02-0.13/1M tokens' },
        { name: 'Cohere', models: 'embed-english-v3.0', dimensions: '1024', apiKey: 'COHERE_API_KEY', cost: '$0.10/1M tokens' },
        { name: 'Voyage', models: 'voyage-3', dimensions: '512-1024', apiKey: 'VOYAGE_API_KEY', cost: '$0.12/1M tokens' },
        { name: 'Mistral', models: 'mistral-embed', dimensions: '1024', apiKey: 'MISTRAL_API_KEY', cost: '$0.10/1M tokens' },
        { name: 'Google', models: 'text-embedding-004', dimensions: '768', apiKey: 'GOOGLE_API_KEY', cost: 'Free tier available' },
        { name: 'HuggingFace', models: 'BAAI/bge-*', dimensions: '384-1024', apiKey: 'HF_TOKEN', cost: 'Free tier available' },
        { name: 'Jina', models: 'jina-embeddings-v3', dimensions: '1024', apiKey: 'JINA_API_KEY', cost: '$0.02/1M tokens' },
        { name: 'Nomic', models: 'nomic-embed-text-v1.5', dimensions: '768', apiKey: 'NOMIC_API_KEY', cost: 'Free tier available' },
    ]

    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Embedding Providers
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        ARGUS supports 16+ embedding providers for semantic search, RAG, and similarity computations.
                    </p>
                </div>

                {/* Provider Comparison */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Provider Comparison</h2>

                    <div className="overflow-x-auto">
                        <table className="w-full border-collapse rounded-lg overflow-hidden text-sm">
                            <thead className="bg-muted">
                                <tr>
                                    <th className="px-4 py-3 text-left font-semibold">Provider</th>
                                    <th className="px-4 py-3 text-left font-semibold">Models</th>
                                    <th className="px-4 py-3 text-left font-semibold">Dimensions</th>
                                    <th className="px-4 py-3 text-left font-semibold">API Key</th>
                                    <th className="px-4 py-3 text-left font-semibold">Cost</th>
                                </tr>
                            </thead>
                            <tbody>
                                {providers.map((provider, idx) => (
                                    <tr key={idx} className={`border-t ${idx % 2 === 0 ? 'bg-muted/30' : ''}`}>
                                        <td className="px-4 py-3 font-semibold">{provider.name}</td>
                                        <td className="px-4 py-3 font-mono text-xs">{provider.models}</td>
                                        <td className="px-4 py-3">{provider.dimensions}</td>
                                        <td className="px-4 py-3 font-mono text-xs">{provider.apiKey}</td>
                                        <td className="px-4 py-3">{provider.cost}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </section>

                {/* Local Providers */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Local Providers (Free)</h2>
                    <p className="text-muted-foreground">
                        Run embeddings locally without API keys or costs:
                    </p>

                    <div className="space-y-6">
                        <div>
                            <h3 className="text-xl font-semibold mb-3">SentenceTransformers (Most Popular)</h3>
                            <CodeBlock
                                code={`from argus.embeddings import get_embedding

# Local, free, no API key needed
embedder = get_embedding("sentence_transformers", model="all-MiniLM-L6-v2")

# Embed documents
texts = ["Hello world", "Machine learning", "Python programming"]
vectors = embedder.embed_documents(texts)

# Embed query
query_vec = embedder.embed_query("What is AI?")

print(f"Dimension: {len(vectors[0])}")  # 384`}
                                language="python"
                            />
                        </div>

                        <div>
                            <h3 className="text-xl font-semibold mb-3">FastEmbed (Fastest)</h3>
                            <CodeBlock
                                code={`from argus.embeddings import get_embedding

# Optimized for speed
embedder = get_embedding("fastembed", model="BAAI/bge-small-en-v1.5")

vectors = embedder.embed_documents(texts)  # Very fast!`}
                                language="python"
                            />
                        </div>

                        <div>
                            <h3 className="text-xl font-semibold mb-3">Ollama (Local LLM Embeddings)</h3>
                            <CodeBlock
                                code={`from argus.embeddings import get_embedding

# Requires Ollama running locally
embedder = get_embedding("ollama", model="nomic-embed-text")

vectors = embedder.embed_documents(texts)`}
                                language="python"
                            />
                            <Callout variant="info" className="mt-3">
                                Install Ollama from <a href="https://ollama.ai" className="text-primary hover:underline">ollama.ai</a> and run: <code>ollama pull nomic-embed-text</code>
                            </Callout>
                        </div>
                    </div>
                </section>

                {/* Cloud Providers */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Cloud Providers</h2>

                    <div className="space-y-6">
                        <div>
                            <h3 className="text-xl font-semibold mb-3">OpenAI</h3>
                            <CodeBlock
                                code={`from argus.embeddings import get_embedding

embedder = get_embedding(
    "openai",
    model="text-embedding-3-small",  # or text-embedding-3-large
    api_key="sk-...",  # or use OPENAI_API_KEY env var
)

vectors = embedder.embed_documents(texts)
print(f"Dimension: {len(vectors[0])}")  # 1536 or 3072`}
                                language="python"
                            />
                        </div>

                        <div>
                            <h3 className="text-xl font-semibold mb-3">Cohere</h3>
                            <CodeBlock
                                code={`from argus.embeddings import get_embedding

embedder = get_embedding(
    "cohere",
    model="embed-english-v3.0",
    api_key="...",  # or use COHERE_API_KEY
)

vectors = embedder.embed_documents(texts)`}
                                language="python"
                            />
                        </div>

                        <div>
                            <h3 className="text-xl font-semibold mb-3">Voyage AI</h3>
                            <CodeBlock
                                code={`from argus.embeddings import get_embedding

embedder = get_embedding(
    "voyage",
    model="voyage-3",
    api_key="...",  # or use VOYAGE_API_KEY
)

vectors = embedder.embed_documents(texts)`}
                                language="python"
                            />
                        </div>
                    </div>
                </section>

                {/* Batch Processing */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Batch Processing</h2>
                    <p className="text-muted-foreground">
                        Efficiently process large document collections:
                    </p>

                    <CodeBlock
                        code={`from argus.embeddings import get_embedding

embedder = get_embedding("sentence_transformers", model="all-MiniLM-L6-v2")

# Large document collection
documents = [f"Document {i}" for i in range(10000)]

# Batch processing with progress
vectors = embedder.embed_documents(
    documents,
    batch_size=32,  # Process 32 at a time
    show_progress=True,
)

print(f"Embedded {len(vectors)} documents")`}
                        language="python"
                    />
                </section>

                {/* Similarity Search */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Similarity Search</h2>
                    <CodeBlock
                        code={`import numpy as np
from argus.embeddings import get_embedding

embedder = get_embedding("sentence_transformers", model="all-MiniLM-L6-v2")

# Embed corpus
corpus = [
    "Machine learning is a subset of AI",
    "Python is a programming language",
    "Neural networks are inspired by the brain",
]
corpus_vectors = embedder.embed_documents(corpus)

# Embed query
query = "What is AI?"
query_vector = embedder.embed_query(query)

# Compute cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Find most similar
similarities = [cosine_similarity(query_vector, vec) for vec in corpus_vectors]
best_idx = np.argmax(similarities)

print(f"Most similar: {corpus[best_idx]}")
print(f"Similarity: {similarities[best_idx]:.3f}")`}
                        language="python"
                    />
                </section>

                {/* Best Practices */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Best Practices</h2>

                    <div className="space-y-4">
                        <Callout variant="tip" title="Development vs Production">
                            <ul className="list-disc list-inside space-y-1">
                                <li><strong>Development:</strong> Use local models (SentenceTransformers, FastEmbed) - free and fast</li>
                                <li><strong>Production:</strong> Consider cloud providers for better quality and scalability</li>
                            </ul>
                        </Callout>

                        <Callout variant="info" title="Model Selection">
                            <ul className="list-disc list-inside space-y-1">
                                <li><strong>General purpose:</strong> all-MiniLM-L6-v2 (384d, fast)</li>
                                <li><strong>High quality:</strong> all-mpnet-base-v2 (768d, better quality)</li>
                                <li><strong>Code search:</strong> voyage-code-3</li>
                                <li><strong>Multilingual:</strong> BAAI/bge-m3</li>
                            </ul>
                        </Callout>

                        <Callout variant="warning" title="Cost Optimization">
                            <ul className="list-disc list-inside space-y-1">
                                <li>Cache embeddings to avoid recomputation</li>
                                <li>Use smaller models for development</li>
                                <li>Batch process to reduce API calls</li>
                                <li>Consider local models for high-volume use cases</li>
                            </ul>
                        </Callout>
                    </div>
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a
                            href="/docs/modules/embeddings"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">Embeddings Module →</h3>
                            <p className="text-sm text-muted-foreground">
                                Deep dive into the embeddings module
                            </p>
                        </a>
                        <a
                            href="/docs/modules/retrieval"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">Retrieval →</h3>
                            <p className="text-sm text-muted-foreground">
                                Hybrid retrieval with embeddings
                            </p>
                        </a>
                        <a
                            href="/tutorials"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">Tutorials →</h3>
                            <p className="text-sm text-muted-foreground">
                                Practical embedding examples
                            </p>
                        </a>
                    </div>
                </section>
            </div>
        </div>
    )
}
