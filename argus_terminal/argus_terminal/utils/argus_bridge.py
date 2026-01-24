"""
Bridge to Argus package for terminal integration.

Provides utilities for connecting to the argus package functionality.
"""

from typing import Any, Dict, List, Optional
import sys

# Lazy imports to avoid dependencies at startup
_argus_available = None


def check_argus_available() -> bool:
    """Check if argus package is available."""
    global _argus_available
    if _argus_available is None:
        try:
            import argus
            _argus_available = True
        except BaseException:
            # Attempt to add project root to path for local dev
            try:
                from pathlib import Path
                # Assuming structure: root/argus_terminal/argus_terminal/utils/argus_bridge.py
                # We want: root/
                # argus_bridge.py is in argus_terminal/utils
                # So .parent.parent is argus_terminal(source)
                # .parent.parent.parent is argus_terminal(repo)
                # .parent.parent.parent.parent is root (c:\ingester_ops\argus)
                
                current_file = Path(__file__).resolve()
                project_root = current_file.parent.parent.parent.parent
                
                # Also try one level up if structure is flattened
                # c:\ingester_ops\argus\argus_terminal -> c:\ingester_ops\argus
                
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                
                import argus
                # Auto-register integrations
                try:
                    from argus.tools.integrations import get_all_tools
                    from argus.tools import register_tool
                    for tool in get_all_tools():
                        try:
                            register_tool(tool)
                        except ValueError:
                            pass # Already registered
                except Exception:
                    pass
                
                _argus_available = True
            except BaseException:
                # Catch SystemExit and other hard failures from broken environment
                _argus_available = False
                
    return _argus_available


def get_llm(provider: str = "gemini", model: str | None = None) -> Any:
    """
    Get an LLM instance from argus.
    
    Args:
        provider: LLM provider name
        model: Model name (optional)
    
    Returns:
        LLM instance or None if argus not available
    """
    if not check_argus_available():
        return None
    
    try:
        from argus.core.llm import get_llm as argus_get_llm
        return argus_get_llm(provider, model=model)
    except Exception:
        return None


def list_providers() -> List[str]:
    """List available LLM providers."""
    if not check_argus_available():
        # Fallback list matching argus.core.llm.__init__.py
        return [
            # Core
            "openai", "anthropic", "gemini", "ollama", "cohere", "mistral", "groq",
            # OpenAI-compatible
            "deepseek", "xai", "perplexity", "nvidia", "together", "fireworks",
            # Cloud
            "bedrock", "azure", "vertex", "huggingface",
            # Enterprise
            "watsonx", "databricks", "snowflake", "sambanova", "cerebras",
            # Utility
            "litellm", "cloudflare", "replicate", "vllm", "llamacpp"
        ]
    
    try:
        from argus.core.llm import list_providers as argus_list_providers
        return argus_list_providers()
    except Exception:
        # If import fails, return the comprehensive list
        return [
            "openai", "anthropic", "gemini", "ollama", "cohere", "mistral", "groq",
            "deepseek", "xai", "perplexity", "nvidia", "together", "fireworks",
            "bedrock", "azure", "vertex", "huggingface",
            "watsonx", "databricks", "snowflake", "sambanova", "cerebras",
            "litellm", "cloudflare", "replicate", "vllm", "llamacpp"
        ]


def list_tools() -> List[Dict[str, str]]:
    """List available tools."""
    if not check_argus_available():
        return [
            {"name": "echo", "description": "Echo input back", "category": "system"},
            {"name": "calculator", "description": "Basic arithmetic", "category": "utility"},
            {"name": "duckduckgo", "description": "Web search", "category": "Search"},
            {"name": "wikipedia", "description": "Encyclopedia lookup", "category": "Search"},
            {"name": "arxiv", "description": "Academic papers", "category": "Search"},
            {"name": "python_repl", "description": "Execute Python", "category": "Code"},
            {"name": "filesystem", "description": "File operations", "category": "Files"},
        ]
    
    try:
        from argus.tools import list_tools as argus_list_tools
        from argus.tools import get_tool
        
        # argus_list_tools returns list of names (strings)
        tool_names = argus_list_tools()
        tools_data = []
        
        for name in tool_names:
            tool = get_tool(name)
            if tool:
                # Extract data from BaseTool object
                # Handle enum for category safely
                category = "Uncategorized"
                if hasattr(tool, "category"):
                    cat_val = tool.category
                    # check if it's an enum
                    if hasattr(cat_val, "value"):
                        category = cat_val.value
                    else:
                        category = str(cat_val)
                
                tools_data.append({
                    "name": tool.name,
                    "description": tool.description,
                    "category": category
                })
            else:
                # Fallback if tool not found but in list
                tools_data.append({
                    "name": name,
                    "description": "Custom tool",
                    "category": "Uncategorized"
                })
                
        return tools_data
    except Exception as e:
        # Fallback in case of any error
        return [
            {"name": "echo", "description": "Echo input back", "category": "system"},
            {"name": "calculator", "description": "Basic arithmetic", "category": "utility"},
            {"name": "duckduckgo", "description": "Web search", "category": "Search"},
            {"name": "wikipedia", "description": "Encyclopedia lookup", "category": "Search"},
            {"name": "arxiv", "description": "Academic papers", "category": "Search"},
            {"name": "python_repl", "description": "Execute Python", "category": "Code"},
            {"name": "filesystem", "description": "File operations", "category": "Files"},
        ]


def list_datasets() -> List[str]:
    """List available evaluation datasets."""
    if not check_argus_available():
        return ["TruthfulQA", "MMLU", "GSM8K", "HumanEval", "HellaSwag"]
        
    try:
        from argus.evaluation import list_datasets as argus_list_datasets
        return argus_list_datasets()
    except Exception:
        return ["TruthfulQA", "MMLU"]


def list_connectors() -> List[Dict[str, str]]:
    """List available knowledge connectors."""
    if not check_argus_available():
        return [
            {"name": "Web", "description": "Web Scraper", "status": "active"},
            {"name": "ArXiv", "description": "Academic Papers", "status": "active"},
            {"name": "CrossRef", "description": "DOI/Citations", "status": "active"},
        ]
        
    try:
        # Get from registry
        from argus.knowledge.connectors import get_default_registry
        registry = get_default_registry()
        # Assuming registry has a way to list, otherwise return hardcoded based on imports
        return [
            {"name": c.name, "description": c.__doc__.split("\\n")[0] if c.__doc__ else "Connector", "status": "active"} 
            for c in registry.list_connectors()
        ]
    except Exception:
        # Fallback to standard set if registry fails
        return [
            {"name": "Web", "description": "Web Scraper", "status": "active"},
            {"name": "ArXiv", "description": "Academic Papers", "status": "active"},
            {"name": "CrossRef", "description": "DOI/Citations", "status": "active"},
        ]


def run_debate(
    proposition: str,
    prior: float = 0.5,
    max_rounds: int = 5,
    provider: str = "gemini",
    model: str | None = None,
    callback: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Run a debate on a proposition.
    
    Args:
        proposition: The proposition to debate
        prior: Prior probability
        max_rounds: Maximum debate rounds
        provider: LLM provider
        model: Model name
        callback: Optional progress callback
    
    Returns:
        Debate result dictionary
    """
    if not check_argus_available():
        # Return simulated result
        return {
            "verdict": {
                "label": "SUPPORTED",
                "posterior": 0.72,
                "reasoning": "Demo mode - argus package not available",
            },
            "num_evidence": 3,
            "num_rebuttals": 2,
            "rounds": 3,
            "graph": None,
        }
    
    try:
        from argus import RDCOrchestrator, get_llm
        
        llm = get_llm(provider, model=model)
        orchestrator = RDCOrchestrator(llm=llm, max_rounds=max_rounds)
        
        # If callback provided, hook it up (assuming argus supports callbacks)
        # For now, we just run synchronously
        result = orchestrator.debate(proposition, prior=prior)
        
        return {
            "verdict": {
                "label": result.verdict.label,
                "posterior": result.verdict.posterior,
                "reasoning": result.verdict.reasoning,
            },
            "num_evidence": result.num_evidence,
            "num_rebuttals": getattr(result, "num_rebuttals", 0),
            "rounds": getattr(result, "rounds", 0),
            "graph": result.graph if hasattr(result, "graph") else None,
        }
    except Exception as e:
        return {
            "error": str(e),
            "verdict": None,
        }


def execute_tool(name: str, args: str | dict) -> str:
    """
    Execute a tool by name with arguments.
    
    Args:
        name: Tool name (e.g. 'duckduckgo')
        args: Tool arguments (string or dict)
        
    Returns:
        Tool output as string
    """
    # Normalize args
    query = ""
    if isinstance(args, dict):
        # Try common keys
        query = args.get("query") or args.get("q") or args.get("input") or str(args)
    else:
        query = str(args)

    # Alias mapping for user convenience
    aliases = {
        "duckduckgo": "duckduckgo_search",
        "search": "duckduckgo_search",
        "search_web": "duckduckgo_search",
        "google": "duckduckgo_search",
        "wiki": "wikipedia",
        "calc": "calculator",
        "py": "python_repl",
        "python": "python_repl",
        "file": "filesystem",
        "files": "filesystem",
    }
    
    # Try to map alias
    target_name = aliases.get(name.lower(), name)

    if not check_argus_available():
        # Simulation mode
        if target_name == "duckduckgo_search":
            return (
                f"Searching web for: '{query}'...\n\n"
                "1. [Argus Documentation] Argus is a multi-agent AI debate framework...\n"
                "   https://github.com/argus-ai/argus\n\n"
                "2. [PyPI] argus-ai 0.1.0 - Python Package Index\n"
                "   https://pypi.org/project/argus-ai/\n\n"
                "3. [Wikipedia] Argus Panoptes - Greek Mythology\n"
                "   https://en.wikipedia.org/wiki/Argus_Panoptes\n\n"
                "[Simulated Result: Web search active]"
            )
        elif target_name == "calculator":
            try:
                # Safe eval for calc simulation
                allowed = set("0123456789+-*/(). ")
                if all(c in allowed for c in query):
                    return f"Result: {eval(query)}"
                return "Error: Invalid characters in calculation"
            except:
                return "Error: Calculation failed"
        elif target_name == "echo":
            return f"Echo: {query}"
        else:
            return f"Executed tool '{name}' (mapped to {target_name}) with args: {query}\n[Simulation: Tool logic not available offline]"

    try:
        from argus.tools import get_tool
        
        # Try mapped name first, then original
        tool = get_tool(target_name)
        if not tool:
            tool = get_tool(name)
            
        if not tool:
            return f"Error: Tool '{name}' (or '{target_name}') not found in registry."
        
        # In a real scenario we'd use ToolExecutor, but for single tool:
        # Most tools implement __call__ or run()
        if hasattr(tool, "run"):
            return str(tool.run(query))
        elif hasattr(tool, "__call__"):
            return str(tool(query))
        else:
            return f"Error: Tool '{name}' is not executable (no run method)."
            
    except Exception as e:
        return f"Error executing tool: {str(e)}"
