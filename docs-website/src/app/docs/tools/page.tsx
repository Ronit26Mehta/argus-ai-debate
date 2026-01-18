import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'
import { Callout } from '@/components/Callout'

export const metadata: Metadata = {
    title: 'Tools | ARGUS Documentation',
    description: 'Complete guide to 19+ pre-built tools in ARGUS for search, web, productivity, database, and finance',
}

export default function ToolsPage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Tools
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        ARGUS provides 19+ pre-built tools that agents can use for search, data retrieval, code execution, and more.
                    </p>
                </div>

                {/* Tools Overview */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Available Tools (19)</h2>

                    <div className="grid gap-4 md:grid-cols-2">
                        <div className="p-4 rounded-lg border bg-muted/30">
                            <h4 className="font-semibold mb-2">🔍 Search Tools (6)</h4>
                            <ul className="text-sm space-y-1">
                                <li>• DuckDuckGo (free)</li>
                                <li>• Wikipedia (free)</li>
                                <li>• ArXiv (free)</li>
                                <li>• Tavily</li>
                                <li>• Brave</li>
                                <li>• Exa</li>
                            </ul>
                        </div>
                        <div className="p-4 rounded-lg border bg-muted/30">
                            <h4 className="font-semibold mb-2">🌐 Web Tools (4)</h4>
                            <ul className="text-sm space-y-1">
                                <li>• HTTP Requests</li>
                                <li>• Web Scraper</li>
                                <li>• Jina Reader</li>
                                <li>• YouTube</li>
                            </ul>
                        </div>
                        <div className="p-4 rounded-lg border bg-muted/30">
                            <h4 className="font-semibold mb-2">⚙️ Productivity (5)</h4>
                            <ul className="text-sm space-y-1">
                                <li>• FileSystem</li>
                                <li>• Python REPL</li>
                                <li>• Shell</li>
                                <li>• GitHub</li>
                                <li>• JSON</li>
                            </ul>
                        </div>
                        <div className="p-4 rounded-lg border bg-muted/30">
                            <h4 className="font-semibold mb-2">💾 Database & Finance (4)</h4>
                            <ul className="text-sm space-y-1">
                                <li>• SQL</li>
                                <li>• Pandas DataFrame</li>
                                <li>• Yahoo Finance</li>
                                <li>• Weather</li>
                            </ul>
                        </div>
                    </div>
                </section>

                {/* Search Tools */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Search Tools</h2>

                    <div className="space-y-6">
                        <div>
                            <h3 className="text-xl font-semibold mb-3">DuckDuckGo (Free)</h3>
                            <CodeBlock
                                code={`from argus.tools.integrations import DuckDuckGoTool

search = DuckDuckGoTool()
result = search(query="latest AI research 2024", max_results=10)

for item in result.data["results"]:
    print(f"📰 {item['title']}")
    print(f"   {item['url']}")
    print(f"   {item['snippet'][:100]}...")`}
                                language="python"
                            />
                        </div>

                        <div>
                            <h3 className="text-xl font-semibold mb-3">Wikipedia (Free)</h3>
                            <CodeBlock
                                code={`from argus.tools.integrations import WikipediaTool

wiki = WikipediaTool()

# Search
result = wiki(query="Artificial Intelligence", action="search")

# Get summary
result = wiki(query="Machine Learning", action="summary", sentences=5)
print(result.data["summary"])

# Get full page
result = wiki(query="Neural Network", action="page")
print(result.data["content"])`}
                                language="python"
                            />
                        </div>

                        <div>
                            <h3 className="text-xl font-semibold mb-3">ArXiv (Free)</h3>
                            <CodeBlock
                                code={`from argus.tools.integrations import ArxivTool

arxiv = ArxivTool()
result = arxiv(query="transformer attention mechanism", max_results=5)

for paper in result.data["results"]:
    print(f"📄 {paper['title']}")
    print(f"   Authors: {', '.join(paper['authors'])}")
    print(f"   PDF: {paper['pdf_url']}")`}
                                language="python"
                            />
                        </div>
                    </div>
                </section>

                {/* Web Tools */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Web Tools</h2>

                    <div className="space-y-6">
                        <div>
                            <h3 className="text-xl font-semibold mb-3">HTTP Requests</h3>
                            <CodeBlock
                                code={`from argus.tools.integrations import RequestsTool

http = RequestsTool()
result = http(
    url="https://api.github.com/repos/python/cpython",
    method="GET"
)

print(f"Stars: {result.data['content']['stargazers_count']}")`}
                                language="python"
                            />
                        </div>

                        <div>
                            <h3 className="text-xl font-semibold mb-3">Web Scraper</h3>
                            <CodeBlock
                                code={`from argus.tools.integrations import WebScraperTool

scraper = WebScraperTool()

# Extract text
result = scraper(url="https://example.com", extract="text")

# Extract links
result = scraper(url="https://example.com", extract="links")

# CSS selector
result = scraper(
    url="https://example.com",
    extract="selector",
    selector="h1.title"
)`}
                                language="python"
                            />
                        </div>
                    </div>
                </section>

                {/* Productivity Tools */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Productivity Tools</h2>

                    <div className="space-y-6">
                        <div>
                            <h3 className="text-xl font-semibold mb-3">Python REPL</h3>
                            <CodeBlock
                                code={`from argus.tools.integrations import PythonReplTool

repl = PythonReplTool()

# Execute code
result = repl(code="x = 10; y = 20; print(x + y)")
print(result.data["output"])  # 30

# Complex computation
result = repl(code="""
import math
primes = [n for n in range(2, 100) if all(n % i != 0 for i in range(2, int(math.sqrt(n))+1))]
print(f"Found {len(primes)} primes")
""")`}
                                language="python"
                            />
                        </div>

                        <div>
                            <h3 className="text-xl font-semibold mb-3">FileSystem</h3>
                            <CodeBlock
                                code={`from argus.tools.integrations import FileSystemTool

fs = FileSystemTool(base_dir="./data")

# List directory
result = fs(action="list", path=".")

# Read file
result = fs(action="read", path="config.json")

# Write file
result = fs(action="write", path="output.txt", content="Hello")`}
                                language="python"
                            />
                        </div>
                    </div>
                </section>

                {/* Database & Finance */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Database & Finance Tools</h2>

                    <div className="space-y-6">
                        <div>
                            <h3 className="text-xl font-semibold mb-3">Yahoo Finance (Free)</h3>
                            <CodeBlock
                                code={`from argus.tools.integrations import YahooFinanceTool

yf = YahooFinanceTool()

# Get quote
result = yf(symbol="AAPL", action="quote")
print(f"Apple: \${result.data['price']}")

# Historical data
result = yf(symbol="MSFT", action="history", period="1mo")

# News
result = yf(symbol="NVDA", action="news")`}
                                language="python"
                            />
                        </div>

                        <div>
                            <h3 className="text-xl font-semibold mb-3">SQL Database</h3>
                            <CodeBlock
                                code={`from argus.tools.integrations import SqlTool

sql = SqlTool(connection_string="sqlite:///mydb.db")

# Query
result = sql(query="SELECT * FROM users LIMIT 10")

# With parameters
result = sql(
    query="SELECT * FROM users WHERE age > :min_age",
    params={"min_age": 25}
)`}
                                language="python"
                            />
                        </div>
                    </div>
                </section>

                {/* Custom Tools */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Creating Custom Tools</h2>
                    <CodeBlock
                        code={`from argus.tools import BaseTool, ToolResult, ToolCategory

class MyCustomTool(BaseTool):
    """My custom tool implementation."""
    
    name = "my_tool"
    description = "Does something useful"
    category = ToolCategory.UTILITY
    
    def execute(self, query: str, **kwargs) -> ToolResult:
        try:
            result = {"data": f"Processed: {query}"}
            return ToolResult.from_data(result)
        except Exception as e:
            return ToolResult.from_error(str(e))

# Use custom tool
tool = MyCustomTool()
result = tool(query="hello world")`}
                        language="python"
                    />
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a
                            href="/docs/modules/tools"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">Tools Module →</h3>
                            <p className="text-sm text-muted-foreground">
                                Deep dive into the tools module
                            </p>
                        </a>
                        <a
                            href="/docs/core-concepts/agents"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">Agents →</h3>
                            <p className="text-sm text-muted-foreground">
                                Use tools with agents
                            </p>
                        </a>
                        <a
                            href="/tutorials"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">Tutorials →</h3>
                            <p className="text-sm text-muted-foreground">
                                Practical tool examples
                            </p>
                        </a>
                    </div>
                </section>
            </div>
        </div>
    )
}
