"""
ARGUS-VIZ: Interactive Debate Visualization Sandbox.

A Streamlit-based web UI for configuring, running, and visualizing
ARGUS multi-agent AI debates with real-time interactive Plotly charts.

Launch:
    argus-viz          (after pip install)
    python -m argus_viz
    streamlit run argus_viz/app.py
"""

__version__ = "3.1"
__app_name__ = "Argus-Viz"


def main():
    """Entry point for the argus-viz console script."""
    import sys
    import os
    from pathlib import Path

    app_path = Path(__file__).parent / "app.py"

    # Launch streamlit with the app
    os.execvp(
        sys.executable,
        [sys.executable, "-m", "streamlit", "run", str(app_path),
         "--server.headless=false",
         "--theme.base=dark",
         "--theme.primaryColor=#00d4ff",
         "--theme.backgroundColor=#0e1117",
         "--theme.secondaryBackgroundColor=#1a1f2e",
         "--theme.textColor=#fafafa",
         "--browser.gatherUsageStats=false"],
    )
