"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { ChevronRight } from "lucide-react"
import { ScrollArea } from "@/components/ui/scroll-area"

const navigation = [
    {
        title: "Getting Started",
        items: [
            { title: "Introduction", href: "/docs/getting-started" },
            { title: "Installation", href: "/docs/installation" },
            { title: "Quick Start", href: "/docs/quick-start" },
            { title: "Configuration", href: "/docs/configuration" },
        ],
    },
    {
        title: "Core Concepts",
        items: [
            { title: "Overview", href: "/docs/core-concepts" },
            { title: "Research Debate Chain", href: "/docs/core-concepts/rdc" },
            { title: "C-DAG Architecture", href: "/docs/core-concepts/cdag" },
            { title: "Multi-Agent System", href: "/docs/core-concepts/agents" },
        ],
    },
    {
        title: "Modules",
        items: [
            { title: "Agents", href: "/docs/modules/agents" },
            { title: "C-DAG", href: "/docs/modules/cdag" },
            { title: "Core", href: "/docs/modules/core" },
            { title: "Decision", href: "/docs/modules/decision" },
            { title: "Durable", href: "/docs/modules/durable" },
            { title: "Embeddings", href: "/docs/modules/embeddings" },
            { title: "Evaluation", href: "/docs/modules/evaluation" },
            { title: "HITL", href: "/docs/modules/hitl" },
            { title: "Knowledge", href: "/docs/modules/knowledge" },
            { title: "MCP", href: "/docs/modules/mcp" },
            { title: "Memory", href: "/docs/modules/memory" },
            { title: "Metrics", href: "/docs/modules/metrics" },
            { title: "Orchestrator", href: "/docs/modules/orchestrator" },
            { title: "Outputs", href: "/docs/modules/outputs" },
            { title: "Provenance", href: "/docs/modules/provenance" },
            { title: "Retrieval", href: "/docs/modules/retrieval" },
            { title: "Tools", href: "/docs/modules/tools" },
        ],
    },
    {
        title: "Integrations",
        items: [
            { title: "LLM Providers", href: "/docs/llm-providers" },
            { title: "Embedding Providers", href: "/docs/embedding-providers" },
            { title: "Tool Integrations", href: "/docs/tools" },
            { title: "External Connectors", href: "/docs/connectors" },
        ],
    },
    {
        title: "Tutorials",
        items: [
            { title: "Overview", href: "/tutorials" },
            { title: "Clinical Evidence", href: "/tutorials/clinical-evidence" },
            { title: "Research Verification", href: "/tutorials/research-verification" },
            { title: "Custom Agent Pipeline", href: "/tutorials/custom-agent-pipeline" },
            { title: "SEC Filing Analysis", href: "/tutorials/sec-debate" },
        ],
    },
    {
        title: "API Reference",
        items: [
            { title: "Overview", href: "/api-reference" },
            { title: "Core", href: "/api-reference/core" },
            { title: "Agents", href: "/api-reference/agents" },
            { title: "C-DAG", href: "/api-reference/cdag" },
        ],
    },
]

export function DocsSidebar() {
    const pathname = usePathname()

    return (
        <aside className="w-64 border-r bg-muted/50 hidden md:block">
            <ScrollArea className="h-[calc(100vh-4rem)] py-6 px-4">
                <nav className="space-y-6">
                    {navigation.map((section) => (
                        <div key={section.title}>
                            <h4 className="mb-2 text-sm font-semibold">{section.title}</h4>
                            <ul className="space-y-1">
                                {section.items.map((item) => (
                                    <li key={item.href}>
                                        <Link
                                            href={item.href}
                                            className={`block rounded-md px-3 py-2 text-sm transition-colors hover:bg-accent ${pathname === item.href
                                                ? "bg-accent text-accent-foreground font-medium"
                                                : "text-muted-foreground"
                                                }`}
                                        >
                                            {item.title}
                                        </Link>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    ))}
                </nav>
            </ScrollArea>
        </aside>
    )
}
