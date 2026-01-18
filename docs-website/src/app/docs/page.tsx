import { Metadata } from 'next'
import Link from 'next/link'
import { ArrowRight, BookOpen, Code, Rocket, Zap } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

export const metadata: Metadata = {
    title: 'Documentation | ARGUS',
    description: 'Complete documentation for ARGUS - Agentic Research & Governance Unified System',
}

export default function DocsIndexPage() {
    return (
        <div className="space-y-12">
            {/* Hero */}
            <div className="text-center space-y-4">
                <h1 className="text-5xl font-bold gradient-text">ARGUS Documentation</h1>
                <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                    Everything you need to build evidence-based AI systems with multi-agent debates
                </p>
            </div>

            {/* Quick Links */}
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                <Card className="hover:shadow-lg transition-all card-hover">
                    <CardHeader>
                        <Rocket className="h-10 w-10 text-primary mb-2" />
                        <CardTitle>Getting Started</CardTitle>
                        <CardDescription>
                            Install ARGUS and run your first debate in minutes
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <Link href="/docs/getting-started">
                            <Button variant="ghost" className="w-full justify-between">
                                Start Here
                                <ArrowRight className="h-4 w-4" />
                            </Button>
                        </Link>
                    </CardContent>
                </Card>

                <Card className="hover:shadow-lg transition-all card-hover">
                    <CardHeader>
                        <BookOpen className="h-10 w-10 text-primary mb-2" />
                        <CardTitle>Core Concepts</CardTitle>
                        <CardDescription>
                            Learn about RDC, C-DAG, and multi-agent systems
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <Link href="/docs/core-concepts">
                            <Button variant="ghost" className="w-full justify-between">
                                Learn Concepts
                                <ArrowRight className="h-4 w-4" />
                            </Button>
                        </Link>
                    </CardContent>
                </Card>

                <Card className="hover:shadow-lg transition-all card-hover">
                    <CardHeader>
                        <Code className="h-10 w-10 text-primary mb-2" />
                        <CardTitle>API Reference</CardTitle>
                        <CardDescription>
                            Detailed API documentation for all modules
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <Link href="/api-reference">
                            <Button variant="ghost" className="w-full justify-between">
                                View API
                                <ArrowRight className="h-4 w-4" />
                            </Button>
                        </Link>
                    </CardContent>
                </Card>

                <Card className="hover:shadow-lg transition-all card-hover">
                    <CardHeader>
                        <Zap className="h-10 w-10 text-primary mb-2" />
                        <CardTitle>Quick Start</CardTitle>
                        <CardDescription>
                            Jump straight into code with practical examples
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <Link href="/docs/quick-start">
                            <Button variant="ghost" className="w-full justify-between">
                                Quick Start
                                <ArrowRight className="h-4 w-4" />
                            </Button>
                        </Link>
                    </CardContent>
                </Card>

                <Card className="hover:shadow-lg transition-all card-hover">
                    <CardHeader>
                        <BookOpen className="h-10 w-10 text-primary mb-2" />
                        <CardTitle>Tutorials</CardTitle>
                        <CardDescription>
                            Step-by-step guides for real-world use cases
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <Link href="/tutorials">
                            <Button variant="ghost" className="w-full justify-between">
                                Browse Tutorials
                                <ArrowRight className="h-4 w-4" />
                            </Button>
                        </Link>
                    </CardContent>
                </Card>

                <Card className="hover:shadow-lg transition-all card-hover">
                    <CardHeader>
                        <Code className="h-10 w-10 text-primary mb-2" />
                        <CardTitle>Modules</CardTitle>
                        <CardDescription>
                            Explore all 17 ARGUS modules in detail
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <Link href="/docs/modules">
                            <Button variant="ghost" className="w-full justify-between">
                                View Modules
                                <ArrowRight className="h-4 w-4" />
                            </Button>
                        </Link>
                    </CardContent>
                </Card>
            </div>

            {/* Popular Pages */}
            <section className="space-y-4">
                <h2 className="text-3xl font-semibold">Popular Pages</h2>
                <div className="grid md:grid-cols-2 gap-4">
                    <Link href="/docs/installation" className="block p-4 rounded-xl border bg-card hover:shadow-lg transition-all">
                        <h3 className="font-semibold mb-1">Installation</h3>
                        <p className="text-sm text-muted-foreground">Install ARGUS via PyPI or from source</p>
                    </Link>
                    <Link href="/docs/configuration" className="block p-4 rounded-xl border bg-card hover:shadow-lg transition-all">
                        <h3 className="font-semibold mb-1">Configuration</h3>
                        <p className="text-sm text-muted-foreground">Configure LLMs, embeddings, and more</p>
                    </Link>
                    <Link href="/docs/llm-providers" className="block p-4 rounded-xl border bg-card hover:shadow-lg transition-all">
                        <h3 className="font-semibold mb-1">LLM Providers</h3>
                        <p className="text-sm text-muted-foreground">27+ supported LLM providers</p>
                    </Link>
                    <Link href="/docs/core-concepts/rdc" className="block p-4 rounded-xl border bg-card hover:shadow-lg transition-all">
                        <h3 className="font-semibold mb-1">Research Debate Chain</h3>
                        <p className="text-sm text-muted-foreground">The core of ARGUS reasoning</p>
                    </Link>
                </div>
            </section>
        </div>
    )
}
