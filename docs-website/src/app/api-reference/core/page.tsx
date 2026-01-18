import { Metadata } from 'next'
import { CodeBlock } from '@/components/CodeBlock'

export const metadata: Metadata = {
    title: 'Core API Reference | ARGUS',
    description: 'API reference for core module - LLM providers, configuration, and base classes',
}

export default function CoreAPIPage() {
    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <div className="space-y-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight gradient-text mb-4">
                        Core API Reference
                    </h1>
                    <p className="text-lg text-muted-foreground">
                        LLM providers, configuration, and base classes.
                    </p>
                </div>

                {/* get_llm */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold" id="get_llm">get_llm()</h2>
                    <div className="p-6 rounded-xl border bg-muted/30">
                        <p className="text-muted-foreground mb-4">Get an LLM instance by provider name.</p>
                        <CodeBlock
                            code={`def get_llm(
    provider: str,
    model: str,
    api_key: str | None = None,
    **kwargs
) -> BaseLLM:
    """
    Get an LLM instance.
    
    Args:
        provider: Provider name (e.g., "openai", "anthropic")
        model: Model name
        api_key: Optional API key (uses env var if not provided)
        **kwargs: Additional provider-specific arguments
    
    Returns:
        BaseLLM instance
    
    Example:
        >>> llm = get_llm("openai", model="gpt-4o")
        >>> response = llm.generate("Hello")
    """`}
                            language="python"
                        />
                    </div>
                </section>

                {/* BaseLLM */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold" id="BaseLLM">BaseLLM</h2>
                    <div className="p-6 rounded-xl border bg-muted/30">
                        <p className="text-muted-foreground mb-4">Base class for all LLM providers.</p>
                        <CodeBlock
                            code={`class BaseLLM:
    """Base LLM interface."""
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        pass
    
    def stream(
        self,
        prompt: str,
        **kwargs
    ) -> Iterator[str]:
        """Stream text generation."""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass`}
                            language="python"
                        />
                    </div>
                </section>

                {/* ArgusConfig */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold" id="ArgusConfig">ArgusConfig</h2>
                    <div className="p-6 rounded-xl border bg-muted/30">
                        <p className="text-muted-foreground mb-4">Global configuration class.</p>
                        <CodeBlock
                            code={`class ArgusConfig:
    """Global configuration."""
    
    default_provider: str = "openai"
    default_model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    
    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Retrieval
    top_k: int = 10
    lambda_param: float = 0.7
    
    @classmethod
    def from_yaml(cls, path: str) -> "ArgusConfig":
        """Load from YAML file."""
        pass`}
                            language="python"
                        />
                    </div>
                </section>

                {/* Next */}
                <section className="space-y-4">
                    <h2 className="text-3xl font-semibold">See Also</h2>
                    <div className="grid gap-4 md:grid-cols-3">
                        <a href="/api-reference/agents" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Agents →</h3>
                            <p className="text-sm text-muted-foreground">Agent classes</p>
                        </a>
                        <a href="/api-reference/cdag" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">C-DAG →</h3>
                            <p className="text-sm text-muted-foreground">Graph classes</p>
                        </a>
                        <a href="/docs/modules/core" className="block p-6 rounded-xl border bg-card hover:shadow-lg transition-all card-hover">
                            <h3 className="text-lg font-semibold mb-2">Core Docs →</h3>
                            <p className="text-sm text-muted-foreground">Module documentation</p>
                        </a>
                    </div>
                </section>
            </div>
        </div>
    )
}
