import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'

export const metadata: Metadata = {
    title: 'Knowledge Module | ARGUS Documentation',
    description: 'Document loading, chunking, and knowledge base management',
}

export default function KnowledgeModulePage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Knowledge Module
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Document loading, chunking strategies, and knowledge base management.
                    </p>
                </div>

                {/* Document Loading */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Document Loading</h2>
                    <CodeBlock
                        code={`from argus.knowledge import DocumentLoader

loader = DocumentLoader()

# Load single file
doc = loader.load("paper.pdf")

# Load directory
docs = loader.load_directory("./papers/")

# Load with metadata
doc = loader.load("paper.pdf", metadata={"source": "arxiv"})`}
                        language="python"
                    />
                </section>

                {/* Chunking */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Chunking Strategies</h2>
                    <CodeBlock
                        code={`from argus.knowledge import Chunker, ChunkingStrategy

# Recursive chunking (default)
chunker = Chunker(
    chunk_size=512,
    chunk_overlap=50,
    strategy=ChunkingStrategy.RECURSIVE
)

chunks = chunker.chunk(doc)

# Sentence-based
chunker = Chunker(
    chunk_size=512,
    strategy=ChunkingStrategy.SENTENCE
)

# Semantic chunking
chunker = Chunker(
    chunk_size=512,
    strategy=ChunkingStrategy.SEMANTIC,
    embedding_model="all-MiniLM-L6-v2"
)`}
                        language="python"
                    />
                </section>

                {/* Knowledge Base */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Knowledge Base</h2>
                    <CodeBlock
                        code={`from argus.knowledge import KnowledgeBase

kb = KnowledgeBase()

# Add documents
kb.add_document(doc)
kb.add_documents(docs)

# Query
results = kb.query("machine learning", top_k=5)

# Get statistics
print(f"Documents: {kb.num_documents}")
print(f"Chunks: {kb.num_chunks}")`}
                        language="python"
                    />
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a href="/docs/connectors" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Connectors →</h3>
                            <p className="text-sm text-muted-foreground">External sources</p>
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
