
import sys
import os
from pathlib import Path

print(f"Current CWD: {os.getcwd()}")
print(f"Initial sys.path: {sys.path[:3]}...")

# Simulate logic in argus_bridge.py
try:
    current_file = Path("argus_terminal/utils/argus_bridge.py").resolve() # Emulating location
    # Real location if we were running THAT file
    # But here we are running from root argus_terminal/
    
    # Let's try to manually add the root
    # We are in c:\ingester_ops\argus\argus_terminal
    # We want c:\ingester_ops\argus
    
    root_argus = Path("..").resolve()
    print(f"Proposed root: {root_argus}")
    
    if str(root_argus) not in sys.path:
        sys.path.insert(0, str(root_argus))
        print("Added root to sys.path")
    
    import argus
    print(f"SUCCESS: Imported argus from {argus.__file__}")
    
    from argus.core.llm import list_providers
    print(f"Providers: {list_providers()[:3]}...")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
