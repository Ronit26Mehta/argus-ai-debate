import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'

export const metadata: Metadata = {
    title: 'Embeddings Module | ARGUS Documentation',
    description: '16+ embedding model integrations for semantic search and RAG',
}

export default function EmbeddingsModulePage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Embeddings Module
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        16+ embedding model integrations for semantic search, RAG, and similarity computations.
                    </p>
                </div>

                {/* Quick Start */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Quick Start</h2>
                    <CodeBlock
                        code={`from argus.embeddings import get_embedding

# Local (free)
embedder = get_embedding("sentence_transformers", model="all-MiniLM-L6-v2")

# Embed documents
texts = ["Hello world", "Machine learning"]
vectors = embedder.embed_documents(texts)

# Embed query
query_vec = embedder.embed_query("What is AI?")

print(f"Dimension: {len(vectors[0])}")`}
                        language="python"
                    />
                </section>

                {/* Providers */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Available Providers</h2>
                    <div className="grid gap-4 md:grid-cols-2">
                        <div className="p-4 rounded-lg border bg-muted/30">
                            <h4 className="font-semibold mb-2">Local (Free)</h4>
                            <ul className="text-sm space-y-1">
                                <li>• SentenceTransformers</li>
                                <li>• FastEmbed</li>
                                <li>• Ollama</li>
                            </ul>
                        </div>
                        <div className="p-4 rounded-lg border bg-muted/30">
                            <h4 className="font-semibold mb-2">Cloud APIs</h4>
                            <ul className="text-sm space-y-1">
                                <li>• OpenAI</li>
                                <li>• Cohere</li>
                                <li>• Voyage</li>
                                <li>• Google</li>
                                <li>• +9 more</li>
                            </ul>
                        </div>
                    </div>
                </section>

                {/* Batch Processing */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Batch Processing</h2>
                    <CodeBlock
                        code={`# Process large collections
documents = [f"Document {i}" for i in range(10000)]

vectors = embedder.embed_documents(
    documents,
    batch_size=32,
    show_progress=True
)`}
                        language="python"
                    />
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a href="/docs/embedding-providers" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">All Providers →</h3>
                            <p className="text-sm text-muted-foreground">16+ providers</p>
                        </a>
                        <a href="/docs/modules/retrieval" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Retrieval →</h3>
                            <p className="text-sm text-muted-foreground">Hybrid retrieval</p>
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
