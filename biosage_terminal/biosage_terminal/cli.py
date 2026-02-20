"""BioSage CLI entry point.

Provides the 'biosage' command for launching the TUI application.
"""

import sys
import os
import argparse


def main() -> int:
    """Main entry point for the biosage command."""
    parser = argparse.ArgumentParser(
        prog="biosage",
        description="BioSage - AI-Powered Medical Diagnostic Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  GEMINI_API_KEY       Google Gemini API key (preferred)
  GEMINI_MODEL         Gemini model name (default: gemini-1.5-pro)
  OPENAI_API_KEY       OpenAI API key
  OPENAI_MODEL         OpenAI model name (default: gpt-4)
  ANTHROPIC_API_KEY    Anthropic API key
  ANTHROPIC_MODEL      Anthropic model name (default: claude-3-opus-20240229)
  GROQ_API_KEY         Groq API key
  GROQ_MODEL           Groq model name (default: llama3-70b-8192)
  MISTRAL_API_KEY      Mistral API key
  MISTRAL_MODEL        Mistral model name (default: mistral-large-latest)
  COHERE_API_KEY       Cohere API key
  COHERE_MODEL         Cohere model name (default: command-r-plus)
  OLLAMA_MODEL         Ollama model name (default: llama3)
  BIOSAGE_DATA_DIR     Data storage directory (default: ~/.biosage)

Examples:
  biosage                         Launch the TUI
  biosage --check-api             Check API configuration
  biosage --version               Show version
""",
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 3.1.0",
    )
    
    parser.add_argument(
        "--check-api",
        action="store_true",
        help="Check LLM API availability and exit",
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data storage directory",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging",
    )
    
    args = parser.parse_args()
    
    # Set data directory if provided
    if args.data_dir:
        os.environ["BIOSAGE_DATA_DIR"] = args.data_dir
    
    # Check API mode
    if args.check_api:
        return _check_api()
    
    # Launch the TUI
    return _launch_tui(debug=args.debug)


def _check_api() -> int:
    """Check LLM API availability."""
    try:
        from biosage_terminal.ai import check_llm_availability
        
        print("BioSage - LLM Configuration Check")
        print("=" * 40)
        
        status = check_llm_availability()
        
        if status["recommended"]:
            provider_info = status["recommended"]
            print(f"\n[OK] LLM Provider: {provider_info['provider']}")
            print(f"     Model: {provider_info['model']}")
            print(f"\nARGUS Framework: {'Available' if status['argus_available'] else 'Not Available'}")
            
            if status["available"]:
                print(f"\nAll Available Providers ({len(status['available'])}):")
                for p in status["available"]:
                    print(f"  - {p['provider']}: {p['model']}")
            
            print("\nAPI is configured correctly!")
            return 0
        else:
            print("\n[ERROR] No LLM API configured")
            print("\nPlease set one of the following environment variables:")
            print("  - GEMINI_API_KEY (recommended)")
            print("  - OPENAI_API_KEY")
            print("  - ANTHROPIC_API_KEY")
            print("  - GROQ_API_KEY")
            print("  - MISTRAL_API_KEY")
            print("  - COHERE_API_KEY")
            print("  - OLLAMA_MODEL (for local Ollama)")
            
            if status["unavailable"]:
                print("\nUnavailable Providers:")
                for p in status["unavailable"]:
                    print(f"  - {p['provider']}: {p['reason']}")
            
            return 1
            
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return 1
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1


def _launch_tui(debug: bool = False) -> int:
    """Launch the BioSage TUI application."""
    try:
        from biosage_terminal.app import BioSageApp
        
        app = BioSageApp()
        app.run()
        return 0
        
    except ImportError as e:
        print(f"Error: Failed to import BioSage TUI components: {e}")
        print("Make sure all dependencies are installed: pip install biosage-terminal")
        return 1
    except KeyboardInterrupt:
        print("\nBioSage exited.")
        return 0
    except Exception as e:
        if debug:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
