"""
Build script for creating Argus Terminal executable.

Uses PyInstaller to create a standalone .exe file.
"""

import os
import sys
import shutil
from pathlib import Path

def build():
    """Build the Argus Terminal executable."""
    try:
        import PyInstaller.__main__
    except ImportError:
        print("Error: PyInstaller not installed.")
        print("Install with: pip install pyinstaller")
        sys.exit(1)
    
    # Get project root
    project_root = Path(__file__).parent
    
    # Define paths
    entry_point = project_root / "argus_terminal" / "__main__.py"
    themes_dir = project_root / "argus_terminal" / "themes"
    
    # Build arguments
    args = [
        str(entry_point),
        "--name=argus_sandbox",
        "--onefile",
        "--console",  # Keep console for TUI
        f"--add-data={themes_dir};argus_terminal/themes",
        "--hidden-import=textual",
        "--hidden-import=rich",
        "--hidden-import=argus_terminal",
        "--hidden-import=argus_terminal.app",
        "--hidden-import=argus_terminal.screens",
        "--hidden-import=argus_terminal.widgets",
        "--hidden-import=argus_terminal.utils",
        "--clean",
        "--noconfirm",
    ]
    
    print("=" * 60)
    print("Building Argus Terminal Sandbox")
    print("=" * 60)
    print(f"Entry point: {entry_point}")
    print(f"Themes dir: {themes_dir}")
    print()
    
    # Run PyInstaller
    PyInstaller.__main__.run(args)
    
    print()
    print("=" * 60)
    print("Build complete!")
    print("Executable: dist/argus_sandbox.exe")
    print("=" * 60)


if __name__ == "__main__":
    build()
