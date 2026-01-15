"""
ARGUS Command Line Interface.

Provides CLI access to ARGUS functionality:
    - debate: Run a debate on a proposition
    - evaluate: Quick evaluation
    - ingest: Ingest documents
    - benchmark: Run evaluation benchmarks
    - report: Generate reports
    - providers: List LLM providers
    - tools: List registered tools
    - datasets: List evaluation datasets
    - score: Compute ARGUS metrics
    - config: Show configuration
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
        version="%(prog)s 1.3.0",
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
    from argus import list_tools, get_tool
    
    tools = list_tools()
    
    if args.name:
        # Show detailed info for specific tool
        tool = get_tool(args.name)
        if tool is None:
            print(f"❌ Tool not found: {args.name}")
            print(f"   Available: {', '.join(tools)}")
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
        # List all tools
        print("🔧 Registered Tools")
        print("=" * 40)
        for name in tools:
            tool = get_tool(name)
            desc = tool.description[:50] + "..." if len(tool.description) > 50 else tool.description
            print(f"  • {name}: {desc}")
        print()
        print(f"Total: {len(tools)} tools")
        print()
        print("Use 'argus tools <name>' for details")
    
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
        "benchmark": cmd_benchmark,
        "datasets": cmd_datasets,
        "report": cmd_report,
        "tools": cmd_tools,
        "score": cmd_score,
        "config": cmd_config,
    }
    
    if args.command in commands:
        return commands[args.command](args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
