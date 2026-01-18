import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'

export const metadata: Metadata = {
    title: 'Tools Module | ARGUS Documentation',
    description: '19+ pre-built tools for search, web, productivity, database, and finance',
}

export default function ToolsModulePage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Tools Module
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        19+ pre-built tools that agents can use for search, data retrieval, code execution, and more.
                    </p>
                </div>

                {/* Quick Start */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Quick Start</h2>
                    <CodeBlock
                        code={`from argus.tools.integrations import DuckDuckGoTool, WikipediaTool

# Web search (free)
search = DuckDuckGoTool()
result = search(query="AI trends 2024", max_results=10)

# Wikipedia
wiki = WikipediaTool()
result = wiki(query="Machine Learning", action="summary")`}
                        language="python"
                    />
                </section>

                {/* Tool Categories */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Tool Categories</h2>
                    <div className="grid gap-4 md:grid-cols-2">
                        <div className="p-4 rounded-lg border bg-muted/30">
                            <h4 className="font-semibold mb-2">🔍 Search (6)</h4>
                            <ul className="text-sm space-y-1">
                                <li>• DuckDuckGo</li>
                                <li>• Wikipedia</li>
                                <li>• ArXiv</li>
                                <li>• Tavily</li>
                                <li>• Brave</li>
                                <li>• Exa</li>
                            </ul>
                        </div>
                        <div className="p-4 rounded-lg border bg-muted/30">
                            <h4 className="font-semibold mb-2">🌐 Web (4)</h4>
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
                                <li>• Pandas</li>
                                <li>• Yahoo Finance</li>
                                <li>• Weather</li>
                            </ul>
                        </div>
                    </div>
                </section>

                {/* Custom Tools */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Creating Custom Tools</h2>
                    <CodeBlock
                        code={`from argus.tools import BaseTool, ToolResult

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"
    
    def execute(self, query: str, **kwargs) -> ToolResult:
        result = {"data": f"Processed: {query}"}
        return ToolResult.from_data(result)

# Use tool
tool = MyTool()
result = tool(query="hello")`}
                        language="python"
                    />
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a href="/docs/tools" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">All Tools →</h3>
                            <p className="text-sm text-muted-foreground">19+ tools</p>
                        </a>
                        <a href="/docs/modules/agents" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Agents →</h3>
                            <p className="text-sm text-muted-foreground">Use tools with agents</p>
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
