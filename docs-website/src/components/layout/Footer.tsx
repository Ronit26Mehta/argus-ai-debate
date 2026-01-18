import Link from "next/link"
import { Github, Twitter, Package } from "lucide-react"

export function Footer() {
    const currentYear = new Date().getFullYear()

    return (
        <footer className="border-t bg-muted/50">
            <div className="container py-12">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
                    {/* About */}
                    <div className="space-y-3">
                        <h3 className="text-lg font-semibold">ARGUS</h3>
                        <p className="text-sm text-muted-foreground">
                            Production-ready multi-agent AI debate framework for evidence-based reasoning.
                        </p>
                        <div className="flex space-x-2">
                            <Link
                                href="https://github.com/Ronit26Mehta/argus-ai-debate"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-muted-foreground hover:text-foreground transition-colors"
                            >
                                <Github className="h-5 w-5" />
                            </Link>
                            <Link
                                href="https://pypi.org/project/argus-debate-ai/"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-muted-foreground hover:text-foreground transition-colors"
                            >
                                <Package className="h-5 w-5" />
                            </Link>
                        </div>
                    </div>

                    {/* Documentation */}
                    <div className="space-y-3">
                        <h3 className="text-sm font-semibold">Documentation</h3>
                        <ul className="space-y-2 text-sm text-muted-foreground">
                            <li>
                                <Link href="/docs/getting-started" className="hover:text-foreground transition-colors">
                                    Getting Started
                                </Link>
                            </li>
                            <li>
                                <Link href="/docs/core-concepts" className="hover:text-foreground transition-colors">
                                    Core Concepts
                                </Link>
                            </li>
                            <li>
                                <Link href="/docs/modules" className="hover:text-foreground transition-colors">
                                    Modules
                                </Link>
                            </li>
                            <li>
                                <Link href="/api-reference" className="hover:text-foreground transition-colors">
                                    API Reference
                                </Link>
                            </li>
                        </ul>
                    </div>

                    {/* Resources */}
                    <div className="space-y-3">
                        <h3 className="text-sm font-semibold">Resources</h3>
                        <ul className="space-y-2 text-sm text-muted-foreground">
                            <li>
                                <Link href="/tutorials" className="hover:text-foreground transition-colors">
                                    Tutorials
                                </Link>
                            </li>
                            <li>
                                <Link href="/comparison" className="hover:text-foreground transition-colors">
                                    Comparison
                                </Link>
                            </li>
                            <li>
                                <Link href="https://github.com/Ronit26Mehta/argus-ai-debate/issues" className="hover:text-foreground transition-colors">
                                    Issues
                                </Link>
                            </li>
                            <li>
                                <Link href="https://github.com/Ronit26Mehta/argus-ai-debate/discussions" className="hover:text-foreground transition-colors">
                                    Discussions
                                </Link>
                            </li>
                        </ul>
                    </div>

                    {/* Community */}
                    <div className="space-y-3">
                        <h3 className="text-sm font-semibold">Community</h3>
                        <ul className="space-y-2 text-sm text-muted-foreground">
                            <li>
                                <Link href="https://github.com/Ronit26Mehta/argus-ai-debate" className="hover:text-foreground transition-colors">
                                    GitHub
                                </Link>
                            </li>
                            <li>
                                <Link href="https://pypi.org/project/argus-debate-ai/" className="hover:text-foreground transition-colors">
                                    PyPI
                                </Link>
                            </li>
                            <li>
                                <Link href="https://github.com/Ronit26Mehta/argus-ai-debate/blob/main/CONTRIBUTING.md" className="hover:text-foreground transition-colors">
                                    Contributing
                                </Link>
                            </li>
                            <li>
                                <Link href="https://github.com/Ronit26Mehta/argus-ai-debate/blob/main/CODE_OF_CONDUCT.md" className="hover:text-foreground transition-colors">
                                    Code of Conduct
                                </Link>
                            </li>
                        </ul>
                    </div>
                </div>

                <div className="mt-8 pt-8 border-t text-center text-sm text-muted-foreground">
                    <p>
                        © {currentYear} ARGUS Team. Released under the{" "}
                        <Link
                            href="https://github.com/Ronit26Mehta/argus-ai-debate/blob/main/LICENSE"
                            className="hover:text-foreground transition-colors"
                        >
                            MIT License
                        </Link>
                        . Version 1.4.2
                    </p>
                </div>
            </div>
        </footer>
    )
}
