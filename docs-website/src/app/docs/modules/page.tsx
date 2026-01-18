import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

const modules = [
    { name: "Evaluation", desc: "Benchmarking and metrics for debate quality", path: "evaluation" },
    { name: "Memory", desc: "Conversation memory and context management", path: "memory" },
    { name: "MCP", desc: "Model Context Protocol integration", path: "mcp" },
    { name: "Durable", desc: "Durable execution and checkpointing", path: "durable" },
    { name: "HITL", desc: "Human-in-the-loop workflows", path: "hitl" },
    { name: "Knowledge", desc: "Document processing and chunking", path: "knowledge" },
    { name: "Metrics", desc: "Observability and tracing", path: "metrics" },
    { name: "Outputs", desc: "Report generation and visualization", path: "outputs" },
    { name: "Core", desc: "Configuration and data models", path: "core" },
    { name: "Orchestrator", desc: "RDC debate orchestration", path: "orchestrator" },
]

export default function ModulesPage() {
    return (
        <div className="space-y-8">
            <div>
                <h1 className="text-4xl font-bold mb-4">All Modules</h1>
                <p className="text-lg text-muted-foreground">
                    Explore all 17 core modules of the ARGUS framework.
                </p>
            </div>

            <div className="grid md:grid-cols-2 gap-4">
                {modules.map((mod) => (
                    <Card key={mod.path} className="hover:border-primary transition-colors">
                        <CardHeader>
                            <CardTitle>{mod.name}</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <p className="text-sm text-muted-foreground">{mod.desc}</p>
                        </CardContent>
                    </Card>
                ))}
            </div>
        </div>
    )
}
