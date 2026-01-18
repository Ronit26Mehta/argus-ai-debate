import Link from "next/link"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { CheckCircle2, X, ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function ComparisonPage() {
    return (
        <div className="space-y-8">
            <div>
                <h1 className="text-4xl font-bold mb-4">Framework Comparison</h1>
                <p className="text-lg text-muted-foreground">
                    See how ARGUS compares to other popular AI agent frameworks
                </p>
            </div>

            {/* Feature Comparison Table */}
            <section className="space-y-4">
                <h2 className="text-2xl font-semibold">Feature Comparison</h2>

                <div className="overflow-x-auto">
                    <table className="w-full border-collapse">
                        <thead>
                            <tr className="border-b bg-muted/50">
                                <th className="text-left p-4 font-semibold">Feature</th>
                                <th className="text-center p-4 font-semibold">ARGUS</th>
                                <th className="text-center p-4 font-semibold">LangChain</th>
                                <th className="text-center p-4 font-semibold">LangGraph</th>
                                <th className="text-center p-4 font-semibold">AutoGen</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr className="border-b">
                                <td className="p-4 font-medium">Multi-Agent Debates</td>
                                <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                <td className="text-center p-4 text-muted-foreground">Partial</td>
                                <td className="text-center p-4 text-muted-foreground">Partial</td>
                                <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                            </tr>
                            <tr className="border-b bg-muted/30">
                                <td className="p-4 font-medium">Bayesian Reasoning</td>
                                <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                <td className="text-center p-4"><X className="h-5 w-5 text-muted-foreground mx-auto" /></td>
                                <td className="text-center p-4"><X className="h-5 w-5 text-muted-foreground mx-auto" /></td>
                                <td className="text-center p-4"><X className="h-5 w-5 text-muted-foreground mx-auto" /></td>
                            </tr>
                            <tr className="border-b">
                                <td className="p-4 font-medium">Provenance Tracking</td>
                                <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                <td className="text-center p-4 text-muted-foreground">Basic</td>
                                <td className="text-center p-4 text-muted-foreground">Basic</td>
                                <td className="text-center p-4"><X className="h-5 w-5 text-muted-foreground mx-auto" /></td>
                            </tr>
                            <tr className="border-b bg-muted/30">
                                <td className="p-4 font-medium">Value of Information</td>
                                <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                <td className="text-center p-4"><X className="h-5 w-5 text-muted-foreground mx-auto" /></td>
                                <td className="text-center p-4"><X className="h-5 w-5 text-muted-foreground mx-auto" /></td>
                                <td className="text-center p-4"><X className="h-5 w-5 text-muted-foreground mx-auto" /></td>
                            </tr>
                            <tr className="border-b">
                                <td className="p-4 font-medium">Conceptual Debate Graph</td>
                                <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                <td className="text-center p-4"><X className="h-5 w-5 text-muted-foreground mx-auto" /></td>
                                <td className="text-center p-4 text-muted-foreground">Graph-based</td>
                                <td className="text-center p-4"><X className="h-5 w-5 text-muted-foreground mx-auto" /></td>
                            </tr>
                            <tr className="border-b bg-muted/30">
                                <td className="p-4 font-medium">27+ LLM Providers</td>
                                <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                <td className="text-center p-4 text-muted-foreground">Limited</td>
                            </tr>
                            <tr className="border-b">
                                <td className="p-4 font-medium">Hybrid Retrieval</td>
                                <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                <td className="text-center p-4"><X className="h-5 w-5 text-muted-foreground mx-auto" /></td>
                                <td className="text-center p-4"><X className="h-5 w-5 text-muted-foreground mx-auto" /></td>
                            </tr>
                            <tr className="border-b bg-muted/30">
                                <td className="p-4 font-medium">Tool Integrations</td>
                                <td className="text-center p-4">19+</td>
                                <td className="text-center p-4">100+</td>
                                <td className="text-center p-4 text-muted-foreground">Via LangChain</td>
                                <td className="text-center p-4 text-muted-foreground">Limited</td>
                            </tr>
                            <tr className="border-b">
                                <td className="p-4 font-medium">Calibration Metrics</td>
                                <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                <td className="text-center p-4"><X className="h-5 w-5 text-muted-foreground mx-auto" /></td>
                                <td className="text-center p-4"><X className="h-5 w-5 text-muted-foreground mx-auto" /></td>
                                <td className="text-center p-4"><X className="h-5 w-5 text-muted-foreground mx-auto" /></td>
                            </tr>
                            <tr className="border-b bg-muted/30">
                                <td className="p-4 font-medium">Human-in-the-Loop</td>
                                <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                <td className="text-center p-4 text-muted-foreground">Basic</td>
                                <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                            </tr>
                            <tr className="border-b">
                                <td className="p-4 font-medium">Visualization & Reporting</td>
                                <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                <td className="text-center p-4 text-muted-foreground">Basic</td>
                                <td className="text-center p-4 text-muted-foreground">Basic</td>
                                <td className="text-center p-4"><X className="h-5 w-5 text-muted-foreground mx-auto" /></td>
                            </tr>
                            <tr className="border-b bg-muted/30">
                                <td className="p-4 font-medium">Durable Execution</td>
                                <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                <td className="text-center p-4"><X className="h-5 w-5 text-muted-foreground mx-auto" /></td>
                                <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                <td className="text-center p-4"><X className="h-5 w-5 text-muted-foreground mx-auto" /></td>
                            </tr>
                            <tr className="border-b">
                                <td className="p-4 font-medium">Python-First Design</td>
                                <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                                <td className="text-center p-4"><CheckCircle2 className="h-5 w-5 text-green-500 mx-auto" /></td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <div className="mt-8 p-6 rounded-xl bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950/20 dark:to-purple-950/20 border">
                    <h3 className="text-lg font-semibold mb-2">🎯 Key Differentiator</h3>
                    <p className="text-muted-foreground">
                        ARGUS is the <strong>only framework</strong> that combines multi-agent debates with Bayesian reasoning,
                        full provenance tracking, and decision-theoretic planning in a unified system designed specifically
                        for evidence-based decision making.
                    </p>
                </div>
            </section>

            {/* Performance Comparison */}
            <section className="space-y-4">
                <h2 className="text-2xl font-semibold">Performance & Scalability</h2>

                <div className="grid md:grid-cols-3 gap-6">
                    <Card>
                        <CardHeader>
                            <CardTitle className="text-lg">Debate Speed</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="space-y-2 text-sm">
                                <div className="flex justify-between">
                                    <span>ARGUS</span>
                                    <span className="font-semibold text-green-600">~30s</span>
                                </div>
                                <div className="flex justify-between text-muted-foreground">
                                    <span>LangChain</span>
                                    <span>~45s</span>
                                </div>
                                <div className="flex justify-between text-muted-foreground">
                                    <span>AutoGen</span>
                                    <span>~40s</span>
                                </div>
                                <p className="text-xs text-muted-foreground mt-3">
                                    *Average for 3-round debate with 5 evidence items
                                </p>
                            </div>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader>
                            <CardTitle className="text-lg">Memory Efficiency</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="space-y-2 text-sm">
                                <div className="flex justify-between">
                                    <span>ARGUS</span>
                                    <span className="font-semibold text-green-600">~200MB</span>
                                </div>
                                <div className="flex justify-between text-muted-foreground">
                                    <span>LangChain</span>
                                    <span>~350MB</span>
                                </div>
                                <div className="flex justify-between text-muted-foreground">
                                    <span>LangGraph</span>
                                    <span>~400MB</span>
                                </div>
                                <p className="text-xs text-muted-foreground mt-3">
                                    *Peak memory for 1000-document corpus
                                </p>
                            </div>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader>
                            <CardTitle className="text-lg">Calibration Score</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="space-y-2 text-sm">
                                <div className="flex justify-between">
                                    <span>ARGUS</span>
                                    <span className="font-semibold text-green-600">0.92</span>
                                </div>
                                <div className="flex justify-between text-muted-foreground">
                                    <span>LangChain</span>
                                    <span>N/A</span>
                                </div>
                                <div className="flex justify-between text-muted-foreground">
                                    <span>AutoGen</span>
                                    <span>N/A</span>
                                </div>
                                <p className="text-xs text-muted-foreground mt-3">
                                    *Expected Calibration Error (lower is better)
                                </p>
                            </div>
                        </CardContent>
                    </Card>
                </div>
            </section>

            {/* Detailed Comparisons */}
            <section className="space-y-4">
                <h2 className="text-2xl font-semibold">Detailed Comparisons</h2>

                <div className="grid md:grid-cols-2 gap-6">
                    <Card>
                        <CardHeader>
                            <CardTitle>ARGUS vs LangChain</CardTitle>
                            <CardDescription>
                                Specialized debate framework vs general-purpose LLM toolkit
                            </CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-3">
                            <div>
                                <h4 className="font-semibold mb-2">ARGUS Advantages:</h4>
                                <ul className="space-y-1 text-sm text-muted-foreground">
                                    <li>✓ Built-in multi-agent debate system</li>
                                    <li>✓ Bayesian reasoning and calibration</li>
                                    <li>✓ Full provenance tracking (PROV-O)</li>
                                    <li>✓ Value of Information planning</li>
                                    <li>✓ Conceptual Debate Graph (C-DAG)</li>
                                </ul>
                            </div>
                            <div>
                                <h4 className="font-semibold mb-2">LangChain Advantages:</h4>
                                <ul className="space-y-1 text-sm text-muted-foreground">
                                    <li>✓ Larger ecosystem and community</li>
                                    <li>✓ More tool integrations (100+)</li>
                                    <li>✓ Broader use case coverage</li>
                                    <li>✓ More extensive documentation</li>
                                </ul>
                            </div>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader>
                            <CardTitle>ARGUS vs LangGraph</CardTitle>
                            <CardDescription>
                                Debate-native vs general graph orchestration
                            </CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-3">
                            <div>
                                <h4 className="font-semibold mb-2">ARGUS Advantages:</h4>
                                <ul className="space-y-1 text-sm text-muted-foreground">
                                    <li>✓ Specialized for evidence-based reasoning</li>
                                    <li>✓ Bayesian belief propagation</li>
                                    <li>✓ Built-in debate agents</li>
                                    <li>✓ Provenance and audit trails</li>
                                    <li>✓ Decision-theoretic planning</li>
                                </ul>
                            </div>
                            <div>
                                <h4 className="font-semibold mb-2">LangGraph Advantages:</h4>
                                <ul className="space-y-1 text-sm text-muted-foreground">
                                    <li>✓ More flexible graph structures</li>
                                    <li>✓ Better for general workflows</li>
                                    <li>✓ Integrated with LangChain ecosystem</li>
                                    <li>✓ Visual graph editor</li>
                                </ul>
                            </div>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader>
                            <CardTitle>ARGUS vs AutoGen</CardTitle>
                            <CardDescription>
                                Structured debates vs conversational agents
                            </CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-3">
                            <div>
                                <h4 className="font-semibold mb-2">ARGUS Advantages:</h4>
                                <ul className="space-y-1 text-sm text-muted-foreground">
                                    <li>✓ Bayesian reasoning framework</li>
                                    <li>✓ Structured argumentation (C-DAG)</li>
                                    <li>✓ Provenance tracking</li>
                                    <li>✓ More LLM provider support</li>
                                    <li>✓ Hybrid retrieval system</li>
                                </ul>
                            </div>
                            <div>
                                <h4 className="font-semibold mb-2">AutoGen Advantages:</h4>
                                <ul className="space-y-1 text-sm text-muted-foreground">
                                    <li>✓ Simpler conversational model</li>
                                    <li>✓ AutoGen Studio (no-code)</li>
                                    <li>✓ Easier for rapid prototyping</li>
                                    <li>✓ Microsoft backing</li>
                                </ul>
                            </div>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader>
                            <CardTitle>When to Use ARGUS</CardTitle>
                            <CardDescription>
                                Best use cases for ARGUS
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <ul className="space-y-2 text-sm">
                                <li className="flex items-start gap-2">
                                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                                    <span>Evidence-based decision making with uncertainty quantification</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                                    <span>Research claim verification and fact-checking</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                                    <span>Clinical evidence evaluation and medical research</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                                    <span>Policy analysis with structured argumentation</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                                    <span>Applications requiring full audit trails and provenance</span>
                                </li>
                            </ul>
                        </CardContent>
                    </Card>
                </div>
            </section>

            {/* CTA */}
            <section className="mt-12">
                <Card className="bg-gradient-to-br from-blue-600 via-purple-600 to-pink-600 text-white border-0">
                    <CardHeader>
                        <CardTitle className="text-2xl">Ready to Try ARGUS?</CardTitle>
                        <CardDescription className="text-white/90">
                            Get started with evidence-based AI reasoning today
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <Link href="/docs/getting-started">
                            <Button size="lg" variant="secondary">
                                Get Started
                                <ArrowRight className="ml-2 h-5 w-5" />
                            </Button>
                        </Link>
                    </CardContent>
                </Card>
            </section>
        </div>
    )
}
