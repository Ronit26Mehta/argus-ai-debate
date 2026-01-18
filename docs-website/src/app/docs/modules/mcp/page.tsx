import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'

export const metadata: Metadata = {
    title: 'MCP Module | ARGUS Documentation',
    description: 'Model Context Protocol integration',
}

export default function MCPModulePage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        MCP Module
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Model Context Protocol (MCP) integration for standardized tool and resource access.
                    </p>
                </div>

                {/* MCP Client */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">MCP Client</h2>
                    <CodeBlock
                        code={`from argus.mcp import MCPClient

# Connect to MCP server
client = MCPClient(server_url="http://localhost:8000")

# List available tools
tools = client.list_tools()

# Use tool
result = client.call_tool("search", query="AI research")`}
                        language="python"
                    />
                </section>

                {/* MCP Server */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">MCP Server</h2>
                    <CodeBlock
                        code={`from argus.mcp import MCPServer

# Create MCP server
server = MCPServer(port=8000)

# Register tools
server.register_tool(my_tool)

# Start server
server.start()`}
                        language="python"
                    />
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a href="/docs/modules/tools" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Tools →</h3>
                            <p className="text-sm text-muted-foreground">19+ tools</p>
                        </a>
                        <a href="/docs/modules/agents" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Agents →</h3>
                            <p className="text-sm text-muted-foreground">Multi-agent system</p>
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
