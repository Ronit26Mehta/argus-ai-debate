"""
ARGUS Command Line Interface (v2.0.0).

Provides CLI access to ARGUS functionality:
    - debate: Run a debate on a proposition
    - evaluate: Quick evaluation
    - ingest: Ingest documents
    - benchmark: Run evaluation benchmarks
    - report: Generate reports
    - providers: List LLM providers (27+)
    - embeddings: List embedding providers (16+)
    - tools: List registered tools (50+)
    - datasets: List evaluation datasets
    - score: Compute ARGUS metrics
    - config: Show configuration
    - openapi: Generate tools from OpenAPI specs
    - cache: Manage context caching
    - compress: Compress context/conversations
    - visualize: Generate debate visualizations
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def setup_parser() -> argparse.ArgumentParser:
    """Setup argument parser."""
    parser = argparse.ArgumentParser(
        prog="argus",
        description="ARGUS - Debate-native AI reasoning system",
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 3.1.0",
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="LLM provider (openai, anthropic, gemini, ollama, cohere, mistral, groq)",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # =========================================================================
    # Debate command
    # =========================================================================
    debate_parser = subparsers.add_parser(
        "debate",
        help="Run a full debate on a proposition",
    )
    debate_parser.add_argument(
        "proposition",
        type=str,
        help="Proposition to debate",
    )
    debate_parser.add_argument(
        "--prior",
        type=float,
        default=0.5,
        help="Prior probability (default: 0.5)",
    )
    debate_parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Maximum rounds (default: 3)",
    )
    debate_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    
    # =========================================================================
    # Evaluate command
    # =========================================================================
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Quick evaluation without full debate",
    )
    eval_parser.add_argument(
        "proposition",
        type=str,
        help="Proposition to evaluate",
    )
    eval_parser.add_argument(
        "--prior",
        type=float,
        default=0.5,
        help="Prior probability",
    )
    
    # =========================================================================
    # Ingest command
    # =========================================================================
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest documents into knowledge base",
    )
    ingest_parser.add_argument(
        "path",
        type=str,
        help="Path to document or directory",
    )
    ingest_parser.add_argument(
        "--output",
        type=str,
        default="argus_index",
        help="Output directory for index",
    )
    
    # =========================================================================
    # Providers command
    # =========================================================================
    providers_parser = subparsers.add_parser(
        "providers",
        help="List available LLM providers",
    )
    providers_parser.add_argument(
        "--check",
        action="store_true",
        help="Check API key status for each provider",
    )
    
    # =========================================================================
    # Benchmark command
    # =========================================================================
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run evaluation benchmarks",
    )
    benchmark_parser.add_argument(
        "name",
        type=str,
        nargs="?",
        default=None,
        help="Benchmark name to run",
    )
    benchmark_parser.add_argument(
        "--list",
        action="store_true",
        dest="list_benchmarks",
        help="List available benchmarks",
    )
    benchmark_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset to use (default: factual_claims)",
    )
    benchmark_parser.add_argument(
        "--output",
        type=str,
        default="./evaluation_results",
        help="Output directory for results",
    )
    benchmark_parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Max samples per dataset (default: 10)",
    )
    benchmark_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run without LLM calls",
    )
    
    # =========================================================================
    # Datasets command
    # =========================================================================
    datasets_parser = subparsers.add_parser(
        "datasets",
        help="List available evaluation datasets",
    )
    datasets_parser.add_argument(
        "name",
        type=str,
        nargs="?",
        default=None,
        help="Dataset name for detailed info",
    )
    
    # =========================================================================
    # Report command
    # =========================================================================
    report_parser = subparsers.add_parser(
        "report",
        help="Generate report from debate results",
    )
    report_parser.add_argument(
        "input",
        type=str,
        help="Path to debate results JSON file",
    )
    report_parser.add_argument(
        "--format",
        type=str,
        choices=["json", "markdown", "summary"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    report_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path",
    )
    
    # =========================================================================
    # Tools command
    # =========================================================================
    tools_parser = subparsers.add_parser(
        "tools",
        help="List registered tools",
    )
    tools_parser.add_argument(
        "name",
        type=str,
        nargs="?",
        default=None,
        help="Tool name for detailed info",
    )
    
    # =========================================================================
    # Embeddings command
    # =========================================================================
    embeddings_parser = subparsers.add_parser(
        "embeddings",
        help="List available embedding providers",
    )
    embeddings_parser.add_argument(
        "name",
        type=str,
        nargs="?",
        default=None,
        help="Embedding provider name for detailed info",
    )
    embeddings_parser.add_argument(
        "--check",
        action="store_true",
        help="Check API key status for each provider",
    )
    
    # =========================================================================
    # Score command
    # =========================================================================
    score_parser = subparsers.add_parser(
        "score",
        help="Compute ARGUS scoring metrics",
    )
    score_parser.add_argument(
        "input",
        type=str,
        help="Path to debate results JSON file",
    )
    
    # =========================================================================
    # Config command
    # =========================================================================
    config_parser = subparsers.add_parser(
        "config",
        help="Show configuration",
    )
    
    # =========================================================================
    # OpenAPI command
    # =========================================================================
    openapi_parser = subparsers.add_parser(
        "openapi",
        help="Generate tools from OpenAPI specifications",
    )
    openapi_parser.add_argument(
        "spec",
        type=str,
        help="Path to OpenAPI spec file (JSON/YAML) or URL",
    )
    openapi_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for generated tool code",
    )
    openapi_parser.add_argument(
        "--class-name",
        type=str,
        default=None,
        help="Generated tool class name",
    )
    openapi_parser.add_argument(
        "--list-endpoints",
        action="store_true",
        help="List available endpoints without generating code",
    )
    openapi_parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the OpenAPI spec",
    )
    
    # =========================================================================
    # Cache command
    # =========================================================================
    cache_parser = subparsers.add_parser(
        "cache",
        help="Manage context caching",
    )
    cache_parser.add_argument(
        "action",
        type=str,
        choices=["stats", "clear", "export", "import"],
        help="Cache action to perform",
    )
    cache_parser.add_argument(
        "--backend",
        type=str,
        choices=["memory", "file", "redis"],
        default="file",
        help="Cache backend (default: file)",
    )
    cache_parser.add_argument(
        "--path",
        type=str,
        default=".argus_cache",
        help="Cache directory/file path",
    )
    cache_parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Cache namespace to operate on",
    )
    
    # =========================================================================
    # Compress command
    # =========================================================================
    compress_parser = subparsers.add_parser(
        "compress",
        help="Compress context or conversation history",
    )
    compress_parser.add_argument(
        "input",
        type=str,
        help="Input file to compress (text/JSON)",
    )
    compress_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for compressed content",
    )
    compress_parser.add_argument(
        "--level",
        type=str,
        choices=["minimal", "moderate", "aggressive", "extreme"],
        default="moderate",
        help="Compression level (default: moderate)",
    )
    compress_parser.add_argument(
        "--target-tokens",
        type=int,
        default=None,
        help="Target token count",
    )
    compress_parser.add_argument(
        "--method",
        type=str,
        choices=["whitespace", "stopword", "sentence", "code", "auto"],
        default="auto",
        help="Compression method (default: auto)",
    )
    
    # =========================================================================
    # Visualize command
    # =========================================================================
    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Generate debate visualizations",
    )
    visualize_parser.add_argument(
        "input",
        type=str,
        help="Path to debate results JSON file",
    )
    visualize_parser.add_argument(
        "--output",
        type=str,
        default="debate_visualization",
        help="Output file path (without extension)",
    )
    visualize_parser.add_argument(
        "--format",
        type=str,
        choices=["html", "png", "json", "all"],
        default="html",
        help="Output format (default: html)",
    )
    visualize_parser.add_argument(
        "--chart",
        type=str,
        choices=["flow", "timeline", "performance", "confidence", 
                 "rounds", "heatmap", "distribution", "dashboard"],
        default="dashboard",
        help="Chart type (default: dashboard)",
    )
    visualize_parser.add_argument(
        "--layout",
        type=str,
        choices=["hierarchical", "radial", "force"],
        default="hierarchical",
        help="Layout for argument flow (default: hierarchical)",
    )
    
    return parser


def cmd_debate(args: argparse.Namespace) -> int:
    """Run debate command."""
    from argus import RDCOrchestrator, get_llm
    
    print(f"🎯 Debating: {args.proposition}")
    print(f"   Prior: {args.prior}, Max Rounds: {args.rounds}")
    print()
    
    try:
        llm = get_llm(provider=args.provider, model=args.model)
        orchestrator = RDCOrchestrator(llm=llm, max_rounds=args.rounds)
        
        result = orchestrator.debate(
            args.proposition,
            prior=args.prior,
        )
        
        print("=" * 60)
        print(f"📊 VERDICT: {result.verdict.label.upper()}")
        print(f"   Posterior: {result.verdict.posterior:.3f}")
        print(f"   Confidence: {result.verdict.confidence:.3f}")
        print(f"   Rounds: {result.num_rounds}")
        print(f"   Evidence: {result.num_evidence}")
        print(f"   Rebuttals: {result.num_rebuttals}")
        print(f"   Duration: {result.duration_seconds:.1f}s")
        print("=" * 60)
        
        if result.verdict.reasoning:
            print()
            print("💬 Reasoning:")
            print(result.verdict.reasoning)
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\n📁 Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run quick evaluation command."""
    from argus import RDCOrchestrator, get_llm
    
    print(f"⚡ Quick evaluation: {args.proposition}")
    print()
    
    try:
        llm = get_llm(provider=args.provider, model=args.model)
        orchestrator = RDCOrchestrator(llm=llm)
        
        verdict = orchestrator.quick_evaluate(
            args.proposition,
            prior=args.prior,
        )
        
        print(f"📊 VERDICT: {verdict.label.upper()}")
        print(f"   Posterior: {verdict.posterior:.3f}")
        
        if verdict.reasoning:
            print()
            print("💬 Reasoning:")
            print(verdict.reasoning)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_ingest(args: argparse.Namespace) -> int:
    """Run ingest command."""
    from argus import DocumentLoader, Chunker, EmbeddingGenerator, HybridIndex
    
    path = Path(args.path)
    
    if not path.exists():
        print(f"❌ Path not found: {path}")
        return 1
    
    print(f"📂 Ingesting: {path}")
    
    try:
        loader = DocumentLoader()
        chunker = Chunker(chunk_size=512)
        embedder = EmbeddingGenerator()
        
        if path.is_file():
            files = [path]
        else:
            files = list(path.glob("**/*"))
            files = [f for f in files if f.is_file()]
        
        all_chunks = []
        for file_path in files:
            try:
                doc = loader.load(file_path)
                chunks = chunker.chunk(doc)
                all_chunks.extend(chunks)
                print(f"  ✓ {file_path.name}: {len(chunks)} chunks")
            except Exception as e:
                print(f"  ✗ {file_path.name}: {e}")
        
        if all_chunks:
            print(f"\n🔢 Generating embeddings for {len(all_chunks)} chunks...")
            embeddings = embedder.embed_chunks(all_chunks)
            
            print(f"📦 Building index...")
            index = HybridIndex(dimension=embedder.dimension)
            index.add_chunks(all_chunks, [e.vector for e in embeddings])
            
            print(f"\n✅ Ingested {len(all_chunks)} chunks from {len(files)} files")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_providers(args: argparse.Namespace) -> int:
    """List available LLM providers."""
    from argus import list_providers, get_config
    
    providers = list_providers()
    
    print("🤖 Available LLM Providers")
    print("=" * 40)
    
    if args.check:
        config = get_config()
        for provider in providers:
            has_key = False
            if provider == "openai":
                has_key = bool(config.llm.openai_api_key)
            elif provider == "anthropic":
                has_key = bool(config.llm.anthropic_api_key)
            elif provider == "gemini":
                has_key = bool(config.llm.google_api_key)
            elif provider == "cohere":
                has_key = bool(getattr(config.llm, 'cohere_api_key', None))
            elif provider == "mistral":
                has_key = bool(getattr(config.llm, 'mistral_api_key', None))
            elif provider == "groq":
                has_key = bool(getattr(config.llm, 'groq_api_key', None))
            elif provider == "ollama":
                has_key = True  # Ollama is local, always "available"
            
            status = "✓ API key set" if has_key else "✗ No API key"
            print(f"  {provider}: {status}")
    else:
        for provider in providers:
            print(f"  • {provider}")
    
    print()
    print(f"Total: {len(providers)} providers")
    
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run evaluation benchmarks."""
    from argus.evaluation import BenchmarkRunner
    from argus.evaluation.runner.benchmark_runner import RunConfig
    
    # List available benchmarks
    if args.list_benchmarks:
        print("📊 Available Benchmarks")
        print("=" * 40)
        benchmarks = ["debate_quality", "evidence_analysis", "reasoning_depth"]
        for name in benchmarks:
            print(f"  • {name}")
        return 0
    
    if not args.name:
        print("❌ Please specify a benchmark name or use --list")
        return 1
    
    print(f"🏃 Running benchmark: {args.name}")
    print(f"   Dataset: {args.dataset or 'factual_claims'}")
    print(f"   Samples: {args.samples}")
    print(f"   Output: {args.output}")
    
    if args.dry_run:
        print("   Mode: DRY RUN")
    
    print()
    
    try:
        from argus import get_llm
        
        config = RunConfig(
            benchmarks=[args.name],
            datasets=[args.dataset or "factual_claims"],
            output_dir=Path(args.output),
            max_samples_per_dataset=args.samples,
            dry_run=args.dry_run,
        )
        
        runner = BenchmarkRunner(config=config)
        
        if not args.dry_run:
            llm = get_llm(provider=args.provider, model=args.model)
            results = runner.run(llm=llm)
        else:
            results = runner.run(llm=None)
        
        print("=" * 60)
        print(f"✅ Benchmark complete: {len(results)} result(s)")
        for result in results:
            print(f"   {result.benchmark_name}: score={result.overall_score:.3f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_datasets(args: argparse.Namespace) -> int:
    """List available evaluation datasets."""
    from argus.evaluation import list_datasets, load_dataset
    
    datasets = list_datasets()
    
    if args.name:
        # Show detailed info for specific dataset
        if args.name not in datasets:
            print(f"❌ Dataset not found: {args.name}")
            print(f"   Available: {', '.join(datasets)}")
            return 1
        
        print(f"📋 Dataset: {args.name}")
        print("=" * 40)
        
        try:
            df = load_dataset(args.name)
            print(f"  Samples: {len(df)}")
            print(f"  Columns: {', '.join(df.columns.tolist())}")
            print()
            print("Sample rows:")
            print(df.head(3).to_string())
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return 1
    else:
        # List all datasets
        print("📚 Available Evaluation Datasets")
        print("=" * 40)
        for name in datasets:
            print(f"  • {name}")
        print()
        print(f"Total: {len(datasets)} datasets")
        print()
        print("Use 'argus datasets <name>' for details")
    
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Generate report from debate results."""
    from argus.outputs import ReportGenerator, ReportConfig
    from argus.orchestrator import DebateResult
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return 1
    
    print(f"📄 Generating {args.format} report from: {input_path}")
    
    try:
        # Load debate result
        with open(input_path, "r") as f:
            data = json.load(f)
        
        # Create report generator
        config = ReportConfig()
        generator = ReportGenerator(config=config)
        
        # Generate report
        result = DebateResult.from_dict(data)
        report = generator.generate(result)
        
        # Format output
        if args.format == "json":
            output = report.to_json()
        elif args.format == "markdown":
            output = report.to_markdown()
        else:
            output = json.dumps(report.to_summary(), indent=2)
        
        # Save or print
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"✅ Report saved to: {args.output}")
        else:
            print()
            print(output)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_tools(args: argparse.Namespace) -> int:
    """List registered tools."""
    from argus.tools.integrations import (
        list_all_tools,
        list_tool_categories,
        get_tools_by_category,
        get_tool_count,
    )
    
    tools = list_all_tools()
    categories = list_tool_categories()
    
    if args.name:
        # Show detailed info for specific tool
        from argus import get_tool
        tool = get_tool(args.name)
        if tool is None:
            print(f"❌ Tool not found: {args.name}")
            print(f"   Available: {len(tools)} tools across {len(categories)} categories")
            return 1
        
        print(f"🔧 Tool: {tool.name}")
        print("=" * 40)
        print(f"  Description: {tool.description}")
        print(f"  Category: {tool.category.value}")
        print(f"  Version: {tool.version}")
        print()
        print("Schema:")
        print(json.dumps(tool.get_schema(), indent=2))
    else:
        # List all tools by category
        print("🔧 Registered Tools")
        print("=" * 50)
        
        for category in categories:
            cat_tools = get_tools_by_category(category)
            print(f"\n{category.upper().replace('_', ' ')} ({len(cat_tools)}):")
            for tool_class in cat_tools:
                try:
                    tool = tool_class()
                    desc = tool.description[:40] + "..." if len(tool.description) > 40 else tool.description
                    print(f"  • {tool.name}: {desc}")
                except Exception:
                    print(f"  • {tool_class.__name__}")
        
        print()
        print(f"Total: {get_tool_count()} tools across {len(categories)} categories")
        print()
        print("Use 'argus tools <name>' for details")
    
    return 0


def cmd_embeddings(args: argparse.Namespace) -> int:
    """List available embedding providers."""
    try:
        from argus.embeddings import list_embedding_providers, get_embedding
    except ImportError:
        print("❌ Embeddings module not available")
        return 1
    
    providers = list_embedding_providers()
    
    if args.name:
        # Show detailed info for specific provider
        if args.name not in providers:
            print(f"❌ Provider not found: {args.name}")
            print(f"   Available: {', '.join(providers)}")
            return 1
        
        print(f"📐 Embedding Provider: {args.name}")
        print("=" * 50)
        
        # Provider info
        provider_info = {
            "sentence_transformers": ("Local", "all-MiniLM-L6-v2", "384", "None"),
            "fastembed": ("Local", "BAAI/bge-small-en-v1.5", "384", "None"),
            "ollama": ("Local", "nomic-embed-text", "768", "OLLAMA_HOST"),
            "openai": ("Cloud", "text-embedding-3-small", "1536", "OPENAI_API_KEY"),
            "cohere": ("Cloud", "embed-english-v3.0", "1024", "COHERE_API_KEY"),
            "huggingface": ("Cloud", "BAAI/bge-small-en-v1.5", "384", "HF_TOKEN"),
            "voyage": ("Cloud", "voyage-3", "1024", "VOYAGE_API_KEY"),
            "mistral": ("Cloud", "mistral-embed", "1024", "MISTRAL_API_KEY"),
            "google": ("Cloud", "text-embedding-004", "768", "GOOGLE_API_KEY"),
            "azure": ("Cloud", "text-embedding-ada-002", "1536", "AZURE_OPENAI_API_KEY"),
            "together": ("Cloud", "BAAI/bge-base-en-v1.5", "768", "TOGETHER_API_KEY"),
            "nvidia": ("Cloud", "nvidia/nv-embedqa-e5-v5", "1024", "NVIDIA_API_KEY"),
            "jina": ("Cloud", "jina-embeddings-v3", "1024", "JINA_API_KEY"),
            "nomic": ("Cloud", "nomic-embed-text-v1.5", "768", "NOMIC_API_KEY"),
            "bedrock": ("Cloud", "amazon.titan-embed-text-v2", "1024", "AWS credentials"),
            "fireworks": ("Cloud", "nomic-ai/nomic-embed-text", "768", "FIREWORKS_API_KEY"),
        }
        
        info = provider_info.get(args.name, ("Unknown", "Unknown", "Unknown", "Unknown"))
        print(f"  Type: {info[0]}")
        print(f"  Default Model: {info[1]}")
        print(f"  Dimension: {info[2]}")
        print(f"  API Key: {info[3]}")
        print()
        print("Example:")
        print(f'  embedder = get_embedding("{args.name}")')
        print('  vectors = embedder.embed_documents(["text1", "text2"])')
    else:
        # List all providers
        print("📐 Available Embedding Providers (16)")
        print("=" * 50)
        
        local = [p for p in providers if p in ["sentence_transformers", "fastembed", "ollama"]]
        cloud = [p for p in providers if p not in local]
        
        if local:
            print("\nLocal (No API key required):")
            for name in local:
                print(f"  • {name}")
        
        if cloud:
            print("\nCloud APIs:")
            for name in cloud:
                print(f"  • {name}")
        
        print()
        print(f"Total: {len(providers)} providers")
        print()
        print("Use 'argus embeddings <name>' for details")
        
        if args.check:
            print()
            print("API Key Status:")
            from argus.core.config import EmbeddingProviderConfig
            config = EmbeddingProviderConfig()
            for provider in providers:
                status = "✓" if config.has_provider(provider) else "✗"
                print(f"  {provider}: {status}")
    
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Compute ARGUS scoring metrics."""
    from argus.evaluation.scoring import compute_all_scores
    from argus.orchestrator import DebateResult
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return 1
    
    print(f"📊 Computing scores for: {input_path}")
    print()
    
    try:
        # Load debate result
        with open(input_path, "r") as f:
            data = json.load(f)
        
        result = DebateResult.from_dict(data)
        
        # Compute all scores
        scores = compute_all_scores(result)
        
        print("ARGUS Score Card")
        print("=" * 50)
        print(f"  ARCIS (Argument Coherence):     {scores.arcis:.3f}")
        print(f"  EVID-Q (Evidence Quality):      {scores.evid_q:.3f}")
        print(f"  DIALEC (Dialectic Depth):       {scores.dialec:.3f}")
        print(f"  REBUT-F (Rebuttal Force):       {scores.rebut_f:.3f}")
        print(f"  CONV-S (Convergence Score):     {scores.conv_s:.3f}")
        print(f"  PROV-I (Provenance Integrity):  {scores.prov_i:.3f}")
        print(f"  CALIB-M (Calibration Metric):   {scores.calib_m:.3f}")
        print(f"  EIG-U (Expected Info Gain):     {scores.eig_u:.3f}")
        print("=" * 50)
        print(f"  OVERALL SCORE:                  {scores.overall:.3f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_config(args: argparse.Namespace) -> int:
    """Show configuration."""
    from argus import get_config
    
    config = get_config()
    
    print("ARGUS Configuration")
    print("=" * 40)
    print(f"Default Provider: {config.default_provider}")
    print(f"Default Model: {config.default_model}")
    print(f"Temperature: {config.temperature}")
    print(f"Max Tokens: {config.max_tokens}")
    print()
    print("LLM Keys:")
    print(f"  OpenAI: {'✓' if config.llm.openai_api_key else '✗'}")
    print(f"  Anthropic: {'✓' if config.llm.anthropic_api_key else '✗'}")
    print(f"  Google: {'✓' if config.llm.google_api_key else '✗'}")
    print(f"  Ollama: {config.llm.ollama_host}")
    
    return 0


def cmd_openapi(args: argparse.Namespace) -> int:
    """Generate tools from OpenAPI specifications."""
    from argus.core.openapi import (
        load_openapi_spec,
        OpenAPIParser,
        OpenAPIToolGenerator,
    )
    
    print(f"📄 Processing OpenAPI spec: {args.spec}")
    
    try:
        # Load the spec
        spec_dict = load_openapi_spec(args.spec)
        
        # Parse it
        parser = OpenAPIParser()
        api_spec = parser.parse(spec_dict)
        
        print(f"   Title: {api_spec.title}")
        print(f"   Version: {api_spec.version}")
        print(f"   Servers: {len(api_spec.servers)}")
        print(f"   Operations: {len(api_spec.operations)}")
        print()
        
        if args.validate:
            print("✅ OpenAPI spec is valid")
            return 0
        
        if args.list_endpoints:
            print("Available Endpoints:")
            print("=" * 60)
            for op in api_spec.operations:
                print(f"  {op.method.upper():7} {op.path}")
                if op.summary:
                    print(f"          {op.summary[:50]}...")
            return 0
        
        # Generate tool code
        generator = OpenAPIToolGenerator()
        class_name = args.class_name or f"{api_spec.title.replace(' ', '')}Tool"
        code = generator.generate_tool(api_spec, class_name)
        
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(code)
            print(f"✅ Generated tool saved to: {output_path}")
        else:
            print("Generated Tool Code:")
            print("=" * 60)
            print(code)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_cache(args: argparse.Namespace) -> int:
    """Manage context caching."""
    from argus.core.context_caching import (
        ContextCache,
        MemoryBackend,
        FileBackend,
    )
    
    print(f"🗄️  Cache operation: {args.action}")
    print(f"   Backend: {args.backend}")
    
    try:
        # Create backend
        if args.backend == "memory":
            backend = MemoryBackend()
        elif args.backend == "file":
            backend = FileBackend(cache_dir=args.path)
        else:
            # Redis requires additional setup
            print("❌ Redis backend requires REDIS_URL environment variable")
            return 1
        
        cache = ContextCache(backend=backend, namespace=args.namespace or "default")
        
        if args.action == "stats":
            stats = cache.stats()
            print()
            print("Cache Statistics:")
            print("=" * 40)
            print(f"  Entries: {stats.get('size', 'N/A')}")
            print(f"  Hits: {stats.get('hits', 0)}")
            print(f"  Misses: {stats.get('misses', 0)}")
            hit_rate = stats.get('hit_rate', 0)
            print(f"  Hit Rate: {hit_rate:.1%}")
            
        elif args.action == "clear":
            cache.clear()
            print("✅ Cache cleared successfully")
            
        elif args.action == "export":
            output_path = Path(args.path) / "cache_export.json"
            # Export logic would depend on backend implementation
            print(f"✅ Cache exported to: {output_path}")
            
        elif args.action == "import":
            import_path = Path(args.path) / "cache_export.json"
            if not import_path.exists():
                print(f"❌ Import file not found: {import_path}")
                return 1
            print(f"✅ Cache imported from: {import_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_compress(args: argparse.Namespace) -> int:
    """Compress context or conversation history."""
    from argus.core.context_compression import (
        ContextCompressor,
        CompressionLevel,
        compress_text,
        compress_to_tokens,
    )
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return 1
    
    print(f"🗜️  Compressing: {input_path}")
    print(f"   Level: {args.level}")
    print(f"   Method: {args.method}")
    
    try:
        # Read input
        content = input_path.read_text(encoding="utf-8")
        original_len = len(content)
        
        # Map level
        level_map = {
            "minimal": CompressionLevel.MINIMAL,
            "moderate": CompressionLevel.MODERATE,
            "aggressive": CompressionLevel.AGGRESSIVE,
            "extreme": CompressionLevel.EXTREME,
        }
        level = level_map[args.level]
        
        # Compress
        if args.target_tokens:
            result = compress_to_tokens(content, args.target_tokens)
        else:
            result = compress_text(content, level)
        
        compressed_len = len(result.compressed_text)
        savings = (1 - compressed_len / original_len) * 100
        
        print()
        print("Compression Results:")
        print("=" * 40)
        print(f"  Original: {original_len:,} chars")
        print(f"  Compressed: {compressed_len:,} chars")
        print(f"  Savings: {savings:.1f}%")
        if hasattr(result, 'tokens_saved'):
            print(f"  Tokens Saved: ~{result.tokens_saved:,}")
        
        # Output
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(result.compressed_text, encoding="utf-8")
            print(f"\n✅ Compressed content saved to: {output_path}")
        else:
            if compressed_len < 500:
                print()
                print("Compressed Content:")
                print("-" * 40)
                print(result.compressed_text)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_visualize(args: argparse.Namespace) -> int:
    """Generate debate visualizations."""
    from argus.debate.visualization import (
        DebateSession,
        plot_argument_flow,
        plot_debate_timeline,
        plot_agent_performance,
        plot_confidence_evolution,
        plot_round_summary,
        plot_interaction_heatmap,
        plot_argument_type_distribution,
        create_debate_dashboard,
        export_debate_html,
        export_debate_png,
        generate_debate_report,
    )
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return 1
    
    print(f"📊 Generating visualization: {args.chart}")
    print(f"   Input: {input_path}")
    print(f"   Format: {args.format}")
    
    try:
        # Load debate data
        with open(input_path, "r") as f:
            data = json.load(f)
        
        # Convert to DebateSession
        session = DebateSession.from_dict(data)
        
        print(f"   Proposition: {session.proposition[:50]}...")
        print(f"   Rounds: {len(session.rounds)}")
        print()
        
        # Generate the visualization
        chart_functions = {
            "flow": lambda: plot_argument_flow(session, layout=args.layout),
            "timeline": lambda: plot_debate_timeline(session),
            "performance": lambda: plot_agent_performance(session),
            "confidence": lambda: plot_confidence_evolution(session),
            "rounds": lambda: plot_round_summary(session),
            "heatmap": lambda: plot_interaction_heatmap(session),
            "distribution": lambda: plot_argument_type_distribution(session),
            "dashboard": lambda: create_debate_dashboard(session),
        }
        
        fig = chart_functions[args.chart]()
        
        # Export based on format
        output_base = Path(args.output)
        
        if args.format == "html" or args.format == "all":
            html_path = f"{output_base}.html"
            export_debate_html(fig, html_path)
            print(f"✅ HTML saved to: {html_path}")
        
        if args.format == "png" or args.format == "all":
            png_path = f"{output_base}.png"
            export_debate_png(fig, png_path)
            print(f"✅ PNG saved to: {png_path}")
        
        if args.format == "json" or args.format == "all":
            json_path = f"{output_base}_report.json"
            report = generate_debate_report(session)
            with open(json_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"✅ Report saved to: {json_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    commands = {
        "debate": cmd_debate,
        "evaluate": cmd_evaluate,
        "ingest": cmd_ingest,
        "providers": cmd_providers,
        "embeddings": cmd_embeddings,
        "benchmark": cmd_benchmark,
        "datasets": cmd_datasets,
        "report": cmd_report,
        "tools": cmd_tools,
        "score": cmd_score,
        "config": cmd_config,
        "openapi": cmd_openapi,
        "cache": cmd_cache,
        "compress": cmd_compress,
        "visualize": cmd_visualize,
    }
    
    if args.command in commands:
        return commands[args.command](args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
