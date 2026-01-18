import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'

export const metadata: Metadata = {
    title: 'Retrieval Module | ARGUS Documentation',
    description: 'Hybrid retrieval with BM25, dense embeddings, and reranking',
}

export default function RetrievalModulePage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Retrieval Module
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Hybrid retrieval combining BM25 sparse retrieval, dense embeddings, and cross-encoder reranking.
                    </p>
                </div>

                {/* Hybrid Retrieval */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Hybrid Retrieval</h2>
                    <CodeBlock
                        code={`from argus.retrieval import HybridRetriever

retriever = HybridRetriever(
    embedding_model="all-MiniLM-L6-v2",
    lambda_param=0.7,  # 0=BM25 only, 1=dense only
    use_reranker=True,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Index chunks
retriever.index_chunks(chunks)

# Retrieve
results = retriever.retrieve(
    query="machine learning applications",
    top_k=10,
    rerank_top_k=5
)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Text: {result.chunk.text[:100]}...")`}
                        language="python"
                    />
                </section>

                {/* BM25 Retrieval */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">BM25 (Sparse) Retrieval</h2>
                    <CodeBlock
                        code={`from argus.retrieval import BM25Retriever

retriever = BM25Retriever()
retriever.index_chunks(chunks)

results = retriever.retrieve("query", top_k=10)`}
                        language="python"
                    />
                </section>

                {/* Dense Retrieval */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Dense (Embedding) Retrieval</h2>
                    <CodeBlock
                        code={`from argus.retrieval import DenseRetriever

retriever = DenseRetriever(
    embedding_model="all-MiniLM-L6-v2"
)
retriever.index_chunks(chunks)

results = retriever.retrieve("query", top_k=10)`}
                        language="python"
                    />
                </section>

                {/* Reranking */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Cross-Encoder Reranking</h2>
                    <CodeBlock
                        code={`from argus.retrieval import Reranker

reranker = Reranker(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Rerank results
reranked = reranker.rerank(
    query="machine learning",
    results=initial_results,
    top_k=5
)`}
                        language="python"
                    />
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a href="/docs/modules/embeddings" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Embeddings →</h3>
                            <p className="text-sm text-muted-foreground">Embedding models</p>
                        </a>
                        <a href="/docs/modules/knowledge" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Knowledge →</h3>
                            <p className="text-sm text-muted-foreground">Document loading</p>
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
