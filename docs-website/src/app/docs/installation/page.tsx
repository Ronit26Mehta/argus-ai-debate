import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'
import { Callout } from '@/components/Callout'

export const metadata: Metadata = {
    title: 'Installation | ARGUS Documentation',
    description: 'Complete installation guide for ARGUS - Agentic Research & Governance Unified System',
}

export default function InstallationPage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Installation
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        Get started with ARGUS in minutes. Choose your preferred installation method below.
                    </p>
                </div>

                {/* From PyPI */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">From PyPI (Recommended)</h2>
                    <p className="text-muted-foreground">
                        The easiest way to install ARGUS is via pip from PyPI:
                    </p>

                    <CodeBlock
                        code="pip install argus-debate-ai"
                        language="bash"
                        filename="terminal"
                    />

                    <Callout variant="success" title="Latest Version">
                        This installs ARGUS v1.4.2 with all core dependencies including 27+ LLM providers,
                        16 embedding models, and the complete debate framework.
                    </Callout>
                </section>

                {/* From Source */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">From Source (Development)</h2>
                    <p className="text-muted-foreground">
                        For development or to get the latest features:
                    </p>

                    <CodeBlock
                        code={`git clone https://github.com/Ronit26Mehta/argus-ai-debate.git
cd argus
pip install -e ".[dev]"`}
                        language="bash"
                        filename="terminal"
                    />

                    <p className="text-sm text-muted-foreground">
                        The <code>-e</code> flag installs in editable mode, perfect for development.
                    </p>
                </section>

                {/* Optional Dependencies */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Optional Dependencies</h2>
                    <p className="text-muted-foreground">
                        Install additional features based on your needs:
                    </p>

                    <div className="space-y-6">
                        <div>
                            <h3 className="text-xl font-semibold mb-3">All Features</h3>
                            <CodeBlock
                                code="pip install argus-debate-ai[all]"
                                language="bash"
                            />
                        </div>

                        <div>
                            <h3 className="text-xl font-semibold mb-3">LLM Providers</h3>
                            <CodeBlock
                                code={`# Local LLM support
pip install argus-debate-ai[ollama]

# Cloud providers
pip install argus-debate-ai[cohere]
pip install argus-debate-ai[mistral]
pip install argus-debate-ai[groq]`}
                                language="bash"
                            />
                        </div>

                        <div>
                            <h3 className="text-xl font-semibold mb-3">Tools & Integrations</h3>
                            <CodeBlock
                                code="pip install argus-debate-ai[tools]"
                                language="bash"
                            />
                            <p className="text-sm text-muted-foreground mt-2">
                                Includes 19+ pre-built tools: DuckDuckGo, Wikipedia, ArXiv, GitHub, and more.
                            </p>
                        </div>

                        <div>
                            <h3 className="text-xl font-semibold mb-3">Embeddings</h3>
                            <CodeBlock
                                code="pip install argus-debate-ai[embeddings]"
                                language="bash"
                            />
                            <p className="text-sm text-muted-foreground mt-2">
                                Additional embedding providers: FastEmbed, Voyage, Nomic, and more.
                            </p>
                        </div>

                        <div>
                            <h3 className="text-xl font-semibold mb-3">Visualization</h3>
                            <CodeBlock
                                code="pip install argus-debate-ai[plotting]"
                                language="bash"
                            />
                            <p className="text-sm text-muted-foreground mt-2">
                                Publication-quality plots with matplotlib, seaborn, and plotly.
                            </p>
                        </div>
                    </div>
                </section>

                {/* System Requirements */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">System Requirements</h2>

                    <div className="overflow-x-auto">
                        <table className="w-full border-collapse rounded-lg overflow-hidden">
                            <thead className="bg-muted">
                                <tr>
                                    <th className="px-4 py-3 text-left font-semibold">Requirement</th>
                                    <th className="px-4 py-3 text-left font-semibold">Minimum</th>
                                    <th className="px-4 py-3 text-left font-semibold">Recommended</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr className="border-t">
                                    <td className="px-4 py-3 font-medium">Python</td>
                                    <td className="px-4 py-3">3.11+</td>
                                    <td className="px-4 py-3">3.12+</td>
                                </tr>
                                <tr className="border-t bg-muted/30">
                                    <td className="px-4 py-3 font-medium">RAM</td>
                                    <td className="px-4 py-3">4 GB</td>
                                    <td className="px-4 py-3">16 GB</td>
                                </tr>
                                <tr className="border-t">
                                    <td className="px-4 py-3 font-medium">Storage</td>
                                    <td className="px-4 py-3">1 GB</td>
                                    <td className="px-4 py-3">10 GB (with embeddings)</td>
                                </tr>
                                <tr className="border-t bg-muted/30">
                                    <td className="px-4 py-3 font-medium">GPU</td>
                                    <td className="px-4 py-3">None</td>
                                    <td className="px-4 py-3">CUDA-compatible (for local embeddings)</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </section>

                {/* Virtual Environment */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Virtual Environment Setup</h2>
                    <p className="text-muted-foreground">
                        We strongly recommend using a virtual environment:
                    </p>

                    <div className="space-y-4">
                        <div>
                            <h3 className="text-lg font-semibold mb-2">Using venv</h3>
                            <CodeBlock
                                code={`# Create virtual environment
python -m venv argus-env

# Activate (Linux/Mac)
source argus-env/bin/activate

# Activate (Windows)
argus-env\\Scripts\\activate

# Install ARGUS
pip install argus-debate-ai`}
                                language="bash"
                            />
                        </div>

                        <div>
                            <h3 className="text-lg font-semibold mb-2">Using conda</h3>
                            <CodeBlock
                                code={`# Create conda environment
conda create -n argus python=3.12

# Activate
conda activate argus

# Install ARGUS
pip install argus-debate-ai`}
                                language="bash"
                            />
                        </div>
                    </div>
                </section>

                {/* Verification */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Verify Installation</h2>
                    <p className="text-muted-foreground">
                        Confirm ARGUS is installed correctly:
                    </p>

                    <CodeBlock
                        code={`# Check version
argus --version

# List available providers
argus providers

# Run a simple test
python -c "from argus import CDAG; print('ARGUS installed successfully!')"`}
                        language="bash"
                    />
                </section>

                {/* Troubleshooting */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Troubleshooting</h2>

                    <Callout variant="warning" title="Common Issues">
                        <ul className="list-disc list-inside space-y-2">
                            <li><strong>Python version:</strong> Ensure you're using Python 3.11 or higher</li>
                            <li><strong>pip outdated:</strong> Run <code>pip install --upgrade pip</code></li>
                            <li><strong>Permission errors:</strong> Use <code>pip install --user</code> or a virtual environment</li>
                            <li><strong>Dependency conflicts:</strong> Try installing in a fresh virtual environment</li>
                        </ul>
                    </Callout>
                </section>

                {/* Next Steps */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">Next Steps</h2>
                    <div className="grid gap-4 md:grid-cols-2">
                        <a
                            href="/docs/quick-start"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-xl font-semibold mb-2">Quick Start →</h3>
                            <p className="text-sm text-muted-foreground">
                                Get up and running with your first debate in 5 minutes
                            </p>
                        </a>
                        <a
                            href="/docs/configuration"
                            className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover"
                        >
                            <h3 className="text-xl font-semibold mb-2">Configuration →</h3>
                            <p className="text-sm text-muted-foreground">
                                Set up API keys and configure ARGUS for your needs
                            </p>
                        </a>
                    </div>
                </section>
            </div>
        </div>
    )
}
