"""
CRUX-VIZ: Interactive CRUX Protocol Visualization Sandbox.

A Streamlit-based web UI for configuring, running, and visualizing
CRUX (Claim-Routed Uncertainty eXchange Protocol) debates with
real-time interactive Plotly charts and full 7-primitive analytics.

Launch:
    crux-viz          (after pip install)
    python -m crux_viz
    streamlit run crux_viz/app.py
"""

__version__ = "4.5.0"
__app_name__ = "CRUX-Viz"


def main():
    """Entry point for the crux-viz console script."""
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
         "--theme.primaryColor=#ff00d4",
         "--theme.backgroundColor=#0e1117",
         "--theme.secondaryBackgroundColor=#1a1f2e",
         "--theme.textColor=#fafafa",
         "--browser.gatherUsageStats=false"],
    )
