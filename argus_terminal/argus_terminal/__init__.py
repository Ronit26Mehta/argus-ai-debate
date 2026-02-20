"""
Argus Terminal Sandbox - Bloomberg-style TUI for AI Debate System.

A comprehensive terminal-style sandbox playground application featuring:
- 1980s Amber phosphor theme (Bloomberg-inspired)
- 1970s Green phosphor theme (classic CRT)
- Full Argus feature coverage
- Non-technical user friendly interface
"""

__version__ = "3.1"
__author__ = "Argus Team"

from argus_terminal.app import ArgusTerminalApp


def main():
    """Launch the Argus Terminal Sandbox application."""
    app = ArgusTerminalApp()
    app.run()


__all__ = ["ArgusTerminalApp", "main", "__version__"]
