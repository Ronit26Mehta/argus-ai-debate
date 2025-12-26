"""
ARGUS Command Line Interface.

Provides CLI access to ARGUS functionality:
    - debate: Run a debate on a proposition
    - evaluate: Quick evaluation
    - ingest: Ingest documents
    - query: Query the knowledge base
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
        version="%(prog)s 0.1.0",
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="LLM provider (openai, anthropic, gemini, ollama)",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Debate command
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
    
    # Evaluate command
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
    
    # Ingest command
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
    
    # Config command
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
    
    if args.command == "debate":
        return cmd_debate(args)
    elif args.command == "evaluate":
        return cmd_evaluate(args)
    elif args.command == "ingest":
        return cmd_ingest(args)
    elif args.command == "config":
        return cmd_config(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
