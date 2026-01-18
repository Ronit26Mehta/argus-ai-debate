import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'
import { Callout } from '@/components/Callout'

export const metadata: Metadata = {
    title: 'Connectors | ARGUS Documentation',
    description: 'External connectors for Web, arXiv, and CrossRef in ARGUS',
}

export default function ConnectorsPage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        External Connectors
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Connect to external data sources for document ingestion and retrieval.
                    </p>
                </div>

                {/* Overview */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Available Connectors</h2>

                    <div className="grid gap-4 md:grid-cols-3">
                        <div className="p-6 rounded-xl border bg-card">
                            <h3 className="text-lg font-semibold mb-2">🌐 Web Connector</h3>
                            <p className="text-sm text-muted-foreground">
                                Crawl websites with robots.txt compliance
                            </p>
                        </div>
                        <div className="p-6 rounded-xl border bg-card">
                            <h3 className="text-lg font-semibold mb-2">📄 arXiv Connector</h3>
                            <p className="text-sm text-muted-foreground">
                                Fetch academic papers from arXiv
                            </p>
                        </div>
                        <div className="p-6 rounded-xl border bg-card">
                            <h3 className="text-lg font-semibold mb-2">📚 CrossRef Connector</h3>
                            <p className="text-sm text-muted-foreground">
                                Access citation metadata
                            </p>
                        </div>
                    </div>
                </section>

                {/* Web Connector */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Web Connector</h2>
                    <p className="text-muted-foreground">
                        Crawl and index web content with robots.txt compliance:
                    </p>

                    <CodeBlock
                        code={`from argus.knowledge.connectors import WebConnector

# Initialize connector
connector = WebConnector(
    respect_robots_txt=True,
    max_depth=2,
    max_pages=100,
)

# Crawl website
documents = connector.fetch(
    url="https://example.com",
    include_patterns=["*/blog/*"],
    exclude_patterns=["*/admin/*"],
)

print(f"Fetched {len(documents)} documents")

for doc in documents[:5]:
    print(f"Title: {doc.metadata['title']}")
    print(f"URL: {doc.metadata['url']}")
    print(f"Content: {doc.content[:100]}...")`}
                        language="python"
                    />

                    <Callout variant="info" title="Robots.txt Compliance">
                        The Web Connector automatically respects robots.txt files and implements rate limiting to be a good web citizen.
                    </Callout>
                </section>

                {/* arXiv Connector */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">arXiv Connector</h2>
                    <p className="text-muted-foreground">
                        Fetch academic papers from arXiv for research debates:
                    </p>

                    <CodeBlock
                        code={`from argus.knowledge.connectors import ArxivConnector

# Initialize connector
connector = ArxivConnector()

# Search papers
documents = connector.fetch(
    query="transformer attention mechanism",
    max_results=10,
    sort_by="relevance",  # or "lastUpdatedDate", "submittedDate"
)

for doc in documents:
    print(f"Title: {doc.metadata['title']}")
    print(f"Authors: {', '.join(doc.metadata['authors'])}")
    print(f"Published: {doc.metadata['published']}")
    print(f"PDF: {doc.metadata['pdf_url']}")
    print(f"Abstract: {doc.content[:200]}...")

# Fetch specific paper by ID
doc = connector.fetch_by_id("2103.14030")  # arXiv ID`}
                        language="python"
                    />

                    <div className="p-6 rounded-xl border bg-muted/30">
                        <h4 className="font-semibold mb-3">Metadata Available</h4>
                        <ul className="text-sm space-y-1">
                            <li>• <code>title</code> - Paper title</li>
                            <li>• <code>authors</code> - List of authors</li>
                            <li>• <code>published</code> - Publication date</li>
                            <li>• <code>updated</code> - Last update date</li>
                            <li>• <code>pdf_url</code> - PDF download link</li>
                            <li>• <code>categories</code> - arXiv categories</li>
                            <li>• <code>doi</code> - DOI if available</li>
                        </ul>
                    </div>
                </section>

                {/* CrossRef Connector */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">CrossRef Connector</h2>
                    <p className="text-muted-foreground">
                        Access citation metadata from CrossRef:
                    </p>

                    <CodeBlock
                        code={`from argus.knowledge.connectors import CrossRefConnector

# Initialize connector
connector = CrossRefConnector()

# Search by DOI
doc = connector.fetch_by_doi("10.1038/nature12373")

print(f"Title: {doc.metadata['title']}")
print(f"Authors: {', '.join(doc.metadata['authors'])}")
print(f"Journal: {doc.metadata['journal']}")
print(f"Year: {doc.metadata['year']}")
print(f"Citations: {doc.metadata['citation_count']}")

# Search papers
documents = connector.fetch(
    query="machine learning healthcare",
    max_results=20,
    filter_params={
        "from-pub-date": "2020",
        "type": "journal-article",
    }
)

for doc in documents:
    print(f"📄 {doc.metadata['title']}")
    print(f"   DOI: {doc.metadata['doi']}")
    print(f"   Citations: {doc.metadata['citation_count']}")`}
                        language="python"
                    />
                </section>

                {/* Connector Registry */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Connector Registry</h2>
                    <p className="text-muted-foreground">
                        Register and manage custom connectors:
                    </p>

                    <CodeBlock
                        code={`from argus.knowledge.connectors import ConnectorRegistry, BaseConnector

# List available connectors
connectors = ConnectorRegistry.list_connectors()
print(connectors)  # ['web', 'arxiv', 'crossref']

# Get connector by name
connector = ConnectorRegistry.get_connector("arxiv")

# Register custom connector
class MyConnector(BaseConnector):
    def fetch(self, **kwargs):
        # Your implementation
        pass

ConnectorRegistry.register("my_connector", MyConnector)`}
                        language="python"
                    />
                </section>

                {/* Custom Connector */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Creating Custom Connectors</h2>
                    <CodeBlock
                        code={`from argus.knowledge.connectors import BaseConnector
from argus.knowledge import Document

class MyCustomConnector(BaseConnector):
    """Custom connector implementation."""
    
    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
    
    def fetch(self, query: str, max_results: int = 10, **kwargs):
        """Fetch documents from external source."""
        documents = []
        
        # Your implementation here
        # 1. Make API calls
        # 2. Parse responses
        # 3. Create Document objects
        
        for item in results:
            doc = Document(
                content=item["content"],
                metadata={
                    "title": item["title"],
                    "source": "my_source",
                    "url": item["url"],
                },
            )
            documents.append(doc)
        
        return documents
    
    def fetch_by_id(self, doc_id: str):
        """Fetch specific document by ID."""
        # Your implementation
        pass

# Use custom connector
connector = MyCustomConnector(api_key="...")
docs = connector.fetch(query="machine learning")`}
                        language="python"
                    />
                </section>

                {/* Integration with Retrieval */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Integration with Retrieval</h2>
                    <p className="text-muted-foreground">
                        Use connectors with the retrieval system:
                    </p>

                    <CodeBlock
                        code={`from argus.knowledge.connectors import ArxivConnector
from argus.knowledge import Chunker
from argus.retrieval import HybridRetriever

# Fetch papers
connector = ArxivConnector()
documents = connector.fetch(
    query="transformer models",
    max_results=20
)

# Chunk documents
chunker = Chunker(chunk_size=512, chunk_overlap=50)
chunks = []
for doc in documents:
    chunks.extend(chunker.chunk(doc))

# Create retriever
retriever = HybridRetriever(
    embedding_model="all-MiniLM-L6-v2",
    use_reranker=True,
)
retriever.index_chunks(chunks)

# Search
results = retriever.retrieve("attention mechanism", top_k=5)
for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Text: {result.chunk.text[:100]}...")
    print(f"Source: {result.chunk.metadata['title']}")`}
                        language="python"
                    />
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a
                            href="/docs/modules/knowledge"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">Knowledge Module →</h3>
                            <p className="text-sm text-muted-foreground">
                                Document loading and processing
                            </p>
                        </a>
                        <a
                            href="/docs/modules/retrieval"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">Retrieval →</h3>
                            <p className="text-sm text-muted-foreground">
                                Hybrid retrieval system
                            </p>
                        </a>
                        <a
                            href="/tutorials"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">Tutorials →</h3>
                            <p className="text-sm text-muted-foreground">
                                Practical connector examples
                            </p>
                        </a>
                    </div>
                </section>
            </div>
        </div>
    )
}
