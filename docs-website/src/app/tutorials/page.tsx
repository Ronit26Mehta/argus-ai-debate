import { Metadata } from 'next'
import { Callout } from '@/components/Callout'

export const metadata: Metadata = {
    title: 'Tutorials | ARGUS Documentation',
    description: 'Step-by-step tutorials for real-world ARGUS use cases',
}

export default function TutorialsPage() {
    const tutorials = [
        {
            title: 'Clinical Evidence Evaluation',
            description: 'Evaluate medical treatment claims using multi-agent debates with clinical literature',
            difficulty: 'Intermediate',
            time: '30 minutes',
            topics: ['Medical', 'RAG', 'Multi-Agent'],
            href: '/tutorials/clinical-evidence',
        },
        {
            title: 'Research Claim Verification',
            description: 'Verify scientific claims by fetching papers from arXiv and running structured debates',
            difficulty: 'Intermediate',
            time: '25 minutes',
            topics: ['Research', 'arXiv', 'C-DAG'],
            href: '/tutorials/research-verification',
        },
        {
            title: 'Custom Agent Pipeline',
            description: 'Build a custom multi-agent pipeline with different LLMs and provenance tracking',
            difficulty: 'Advanced',
            time: '45 minutes',
            topics: ['Agents', 'Provenance', 'Multi-Model'],
            href: '/tutorials/custom-agent-pipeline',
        },
        {
            title: 'SEC Filing Analysis',
            description: 'Analyze SEC filings with multi-specialist debates and generate visualizations',
            difficulty: 'Advanced',
            time: '40 minutes',
            topics: ['Finance', 'Visualization', 'Tools'],
            href: '/tutorials/sec-debate',
        },
    ]

    const getDifficultyColor = (difficulty: string) => {
        switch (difficulty) {
            case 'Beginner':
                return 'bg-green-100 dark:bg-green-950/30 text-green-700 dark:text-green-300'
            case 'Intermediate':
                return 'bg-amber-100 dark:bg-amber-950/30 text-amber-700 dark:text-amber-300'
            case 'Advanced':
                return 'bg-red-100 dark:bg-red-950/30 text-red-700 dark:text-red-300'
            default:
                return 'bg-muted text-muted-foreground'
        }
    }

    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Tutorials
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Step-by-step guides for real-world ARGUS use cases. Learn by building practical applications.
                    </p>
                </div>

                <Callout variant="info" title="Prerequisites">
                    Before starting these tutorials, make sure you have:
                    <ul className="list-disc list-inside mt-2 space-y-1">
                        <li>ARGUS installed (<code>pip install argus-debate-ai</code>)</li>
                        <li>Basic Python knowledge</li>
                        <li>API keys for cloud providers (or use Ollama for free)</li>
                    </ul>
                </Callout>

                {/* Tutorial Cards */}
                <section className="space-y-6">
                    <h2 className="text-3xl font-semibold">Available Tutorials</h2>

                    <div className="grid gap-6">
                        {tutorials.map((tutorial, idx) => (
                            <a
                                key={idx}
                                href={tutorial.href}
                                className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                            >
                                <div className="flex items-start justify-between mb-3">
                                    <h3 className="text-2xl font-semibold">{tutorial.title}</h3>
                                    <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getDifficultyColor(tutorial.difficulty)}`}>
                                        {tutorial.difficulty}
                                    </span>
                                </div>

                                <p className="text-muted-foreground mb-4">
                                    {tutorial.description}
                                </p>

                                <div className="flex items-center gap-4 text-sm text-muted-foreground">
                                    <span>⏱️ {tutorial.time}</span>
                                    <span className="flex gap-2">
                                        {tutorial.topics.map((topic, topicIdx) => (
                                            <span
                                                key={topicIdx}
                                                className="px-2 py-1 rounded bg-muted text-xs"
                                            >
                                                {topic}
                                            </span>
                                        ))}
                                    </span>
                                </div>
                            </a>
                        ))}
                    </div>
                </section>

                {/* Coming Soon */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Coming Soon</h2>
                    <div className="grid gap-4 md:grid-cols-2">
                        <div className="p-6 rounded-xl border bg-muted/30">
                            <h3 className="text-lg font-semibold mb-2">Policy Analysis</h3>
                            <p className="text-sm text-muted-foreground">
                                Evaluate policy proposals using multi-agent debates
                            </p>
                        </div>
                        <div className="p-6 rounded-xl border bg-muted/30">
                            <h3 className="text-lg font-semibold mb-2">Legal Document Review</h3>
                            <p className="text-sm text-muted-foreground">
                                Analyze legal documents with specialist agents
                            </p>
                        </div>
                        <div className="p-6 rounded-xl border bg-muted/30">
                            <h3 className="text-lg font-semibold mb-2">Product Review Analysis</h3>
                            <p className="text-sm text-muted-foreground">
                                Aggregate and verify product reviews
                            </p>
                        </div>
                        <div className="p-6 rounded-xl border bg-muted/30">
                            <h3 className="text-lg font-semibold mb-2">News Fact-Checking</h3>
                            <p className="text-sm text-muted-foreground">
                                Verify news claims with evidence-based debates
                            </p>
                        </div>
                    </div>
                </section>

                {/* Learning Path */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Recommended Learning Path</h2>

                    <div className="space-y-4">
                        <div className="p-6 rounded-xl border bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950/20 dark:to-purple-950/20">
                            <h3 className="text-xl font-semibold mb-3">1. Start with Basics</h3>
                            <ul className="space-y-2 text-sm">
                                <li>• Complete the <a href="/docs/quick-start" className="text-primary hover:underline">Quick Start</a> guide</li>
                                <li>• Understand <a href="/docs/core-concepts/rdc" className="text-primary hover:underline">RDC</a> and <a href="/docs/core-concepts/cdag" className="text-primary hover:underline">C-DAG</a></li>
                                <li>• Learn about <a href="/docs/core-concepts/agents" className="text-primary hover:underline">Multi-Agent Systems</a></li>
                            </ul>
                        </div>

                        <div className="p-6 rounded-xl border bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-950/20 dark:to-pink-950/20">
                            <h3 className="text-xl font-semibold mb-3">2. Build Your First Application</h3>
                            <ul className="space-y-2 text-sm">
                                <li>• Try <strong>Research Claim Verification</strong> tutorial</li>
                                <li>• Experiment with different LLM providers</li>
                                <li>• Add document retrieval to your debates</li>
                            </ul>
                        </div>

                        <div className="p-6 rounded-xl border bg-gradient-to-r from-pink-50 to-red-50 dark:from-pink-950/20 dark:to-red-950/20">
                            <h3 className="text-xl font-semibold mb-3">3. Advanced Topics</h3>
                            <ul className="space-y-2 text-sm">
                                <li>• Build <strong>Custom Agent Pipelines</strong></li>
                                <li>• Implement provenance tracking</li>
                                <li>• Create domain-specific tools</li>
                                <li>• Generate visualizations and reports</li>
                            </ul>
                        </div>
                    </div>
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a
                            href="/docs/quick-start"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">Quick Start →</h3>
                            <p className="text-sm text-muted-foreground">
                                5-minute introduction to ARGUS
                            </p>
                        </a>
                        <a
                            href="/docs/modules"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">Modules →</h3>
                            <p className="text-sm text-muted-foreground">
                                Explore all 17 ARGUS modules
                            </p>
                        </a>
                        <a
                            href="/api-reference"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-lg font-semibold mb-2">API Reference →</h3>
                            <p className="text-sm text-muted-foreground">
                                Complete API documentation
                            </p>
                        </a>
                    </div>
                </section>
            </div>
        </div>
    )
}
