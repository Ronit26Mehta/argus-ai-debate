
import sys
import os
from pathlib import Path

# Ensure we import from the local source
current_file = Path(__file__).resolve()
# c:\ingester_ops\argus\verify_core_tools.py (assumed location)
project_root = current_file.parent
sys.path.insert(0, str(project_root))

try:
    from argus.tools import list_tools, get_tool
    
    print("--- Listing Tools ---")
    tools = list_tools()
    print(tools)
    
    print("\n--- Checking Specific Tools ---")
    expected = ["duckduckgo_search", "wikipedia", "calculator"]
    for name in expected:
        tool = get_tool(name)
        status = "FOUND" if tool else "MISSING"
        print(f"{name}: {status}")

except Exception as e:
    print(f"Error: {e}")
