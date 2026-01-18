import { Metadata } from 'next'
import Link from 'next/link'

export const metadata: Metadata = {
    title: 'API Reference | ARGUS Documentation',
    description: 'Complete API reference for all ARGUS modules and classes',
}

export default function APIReferencePage() {
    const modules = [
        { name: 'Core', href: '/api-reference/core', description: 'LLM providers, configuration, and base classes' },
        { name: 'Agents', href: '/api-reference/agents', description: 'Multi-agent system classes' },
        { name: 'C-DAG', href: '/api-reference/cdag', description: 'Graph structure and operations' },
        { name: 'Decision', href: '/api-reference/decision', description: 'Bayesian updating and VoI' },
        { name: 'Embeddings', href: '/api-reference/embeddings', description: 'Embedding providers' },
        { name: 'Evaluation', href: '/api-reference/evaluation', description: 'Metrics and evaluation' },
        { name: 'Knowledge', href: '/api-reference/knowledge', description: 'Document loading and chunking' },
        { name: 'Retrieval', href: '/api-reference/retrieval', description: 'Hybrid retrieval system' },
        { name: 'Tools', href: '/api-reference/tools', description: 'Tool integrations' },
        { name: 'Outputs', href: '/api-reference/outputs', description: 'Visualization and reporting' },
        { name: 'Provenance', href: '/api-reference/provenance', description: 'Audit trail and integrity' },
        { name: 'Orchestrator', href: '/api-reference/orchestrator', description: 'RDC orchestration' },
        { name: 'Memory', href: '/api-reference/memory', description: 'Conversation memory' },
        { name: 'HITL', href: '/api-reference/hitl', description: 'Human-in-the-loop' },
        { name: 'MCP', href: '/api-reference/mcp', description: 'Model Context Protocol' },
        { name: 'Durable', href: '/api-reference/durable', description: 'State persistence' },
        { name: 'Metrics', href: '/api-reference/metrics', description: 'Performance metrics' },
    ]

    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        API Reference
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Complete API documentation for all ARGUS modules, classes, and functions.
                    </p>
                </div>

                {/* Quick Links */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Quick Links</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a href="#core-classes" className="block p-4 rounded-lg border bg-muted/30 hover:bg-muted/50 transition-colors">
                            <h3 className="font-semibold">Core Classes</h3>
                            <p className="text-sm text-muted-foreground">RDCOrchestrator, CDAG, BaseLLM</p>
                        </a>
                        <a href="#agents" className="block p-4 rounded-lg border bg-muted/30 hover:bg-muted/50 transition-colors">
                            <h3 className="font-semibold">Agents</h3>
                            <p className="text-sm text-muted-foreground">Moderator, Specialist, Refuter, Jury</p>
                        </a>
                        <a href="#utilities" className="block p-4 rounded-lg border bg-muted/30 hover:bg-muted/50 transition-colors">
                            <h3 className="font-semibold">Utilities</h3>
                            <p className="text-sm text-muted-foreground">get_llm, get_embedding, helpers</p>
                        </a>
                    </div>
                </section>

                {/* All Modules */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">All Modules</h2>
                    <div className="grid gap-4">
                        {modules.map((module) => (
                            <Link
                                key={module.name}
                                href={module.href}
                                className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                            >
                                <h3 className="text-xl font-semibold mb-2">{module.name}</h3>
                                <p className="text-sm text-muted-foreground">{module.description}</p>
                            </Link>
                        ))}
                    </div>
                </section>

                {/* Usage */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">How to Use This Reference</h2>
                    <div className="p-6 rounded-xl border bg-muted/30">
                        <ul className="space-y-2 text-muted-foreground">
                            <li>• <strong>Browse by module:</strong> Click on any module above to see its classes and functions</li>
                            <li>• <strong>Search:</strong> Use Ctrl+F to search for specific classes or methods</li>
                            <li>• <strong>Examples:</strong> Each class includes usage examples</li>
                            <li>• <strong>Type hints:</strong> All parameters and return types are documented</li>
                        </ul>
                    </div>
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a href="/docs/quick-start" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Quick Start →</h3>
                            <p className="text-sm text-muted-foreground">Get started with ARGUS</p>
                        </a>
                        <a href="/docs/modules" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Modules →</h3>
                            <p className="text-sm text-muted-foreground">Module documentation</p>
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
