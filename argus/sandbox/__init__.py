"""ARGUS Sandbox package entry points."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from argus.sandbox.orchestrator import ArgusSandboxRunner, SandboxConfig

__all__ = [
    "ArgusSandboxRunner",
    "SandboxConfig",
    "launch_sandbox",
]


def launch_sandbox() -> None:
    """Console entry point for launching the Streamlit sandbox UI."""
    app_path = Path(__file__).parent / "interface.py"
    os.execvp(
        sys.executable,
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.headless=false",
            "--browser.gatherUsageStats=false",
        ],
    )
