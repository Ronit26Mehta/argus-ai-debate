
import sys
import os
from pathlib import Path

# Add project root to path so we can import argus_terminal
current_file = Path(__file__).resolve()
# c:\ingester_ops\argus\argus_terminal\verify_tools_final.py
# Root is c:\ingester_ops\argus
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

from argus_terminal.utils.argus_bridge import execute_tool, check_argus_available

print(f"Argus Available: {check_argus_available()}")

# Test 1: Search Web Alias
print("\n--- Test 1: duckduckgo alias ---")
result = execute_tool("duckduckgo", "Argus AI framework")
print(result[:100] + "...") 

# Test 2: Calculator alias
print("\n--- Test 2: calc alias ---")
result_calc = execute_tool("calc", "2 + 2")
print(result_calc)

# Test 3: Unknown tool
print("\n--- Test 3: unknown tool ---")
result_unknown = execute_tool("unknown_tool_xyz", "test")
print(result_unknown)
