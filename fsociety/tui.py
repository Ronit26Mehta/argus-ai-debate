"""
fsociety Terminal User Interface (TUI).

Three-panel layout using Textual:
    ┌──────────────┬───────────────────┬──────────┐
    │ Agent Panel  │  Debate Stream    │  Status  │
    │ (roster)     │  (live findings)  │  Panel   │
    └──────────────┴───────────────────┴──────────┘

Phosphor-green on black aesthetic.
"""

from __future__ import annotations

import logging
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import (
    Header,
    Footer,
    Static,
    RichLog,
    Input,
    Button,
    Label,
)
from textual.css.query import NoMatches
from rich.text import Text
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)


FSOCIETY_CSS = """
Screen {
    background: #0a0a0a;
    color: #00ff41;
}

Header {
    background: #111111;
    color: #00ff41;
}

Footer {
    background: #111111;
    color: #888888;
}

#agent-panel {
    width: 25%;
    height: 100%;
    border: solid #003300;
    background: #0a0a0a;
    padding: 1;
}

#debate-stream {
    width: 50%;
    height: 100%;
    border: solid #003300;
    background: #050505;
    padding: 1;
}

#status-panel {
    width: 25%;
    height: 100%;
    border: solid #003300;
    background: #0a0a0a;
    padding: 1;
}

#input-bar {
    dock: bottom;
    height: 3;
    background: #111111;
    padding: 0 1;
}

#scan-input {
    width: 80%;
    background: #1a1a1a;
    color: #00ff41;
    border: solid #003300;
}

#scan-btn {
    width: 20%;
    background: #003300;
    color: #00ff41;
}

.section-title {
    text-style: bold;
    color: #00ff41;
    text-align: center;
    padding: 1 0;
}

.agent-entry {
    color: #00cc33;
    padding: 0 1;
}

RichLog {
    background: #050505;
    color: #00ff41;
}
"""

BANNER_TEXT = r"""
  ╔═══════════════════════════════════╗
  ║  fsociety VAPT Terminal v1.0.0   ║
  ║  "Hello, friend."                ║
  ╚═══════════════════════════════════╝
"""


class FsocietyTUI(App):
    """fsociety Textual TUI application."""

    CSS = FSOCIETY_CSS
    TITLE = "fsociety VAPT Terminal"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
        ("a", "show_agents", "Agents"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal():
            # Left: Agent roster
            with Vertical(id="agent-panel"):
                yield Label("[ AGENTS ]", classes="section-title")
                yield RichLog(id="agent-log", wrap=True, markup=True)

            # Center: Debate stream
            with Vertical(id="debate-stream"):
                yield Label("[ DEBATE STREAM ]", classes="section-title")
                yield RichLog(id="debate-log", wrap=True, markup=True, auto_scroll=True)

            # Right: Status
            with Vertical(id="status-panel"):
                yield Label("[ STATUS ]", classes="section-title")
                yield RichLog(id="status-log", wrap=True, markup=True)

        with Horizontal(id="input-bar"):
            yield Input(
                placeholder="fsociety ❯ Enter target path or command...",
                id="scan-input",
            )
            yield Button("SCAN", id="scan-btn", variant="success")

        yield Footer()

    def on_mount(self) -> None:
        """Initialize the TUI on mount."""
        debate_log = self.query_one("#debate-log", RichLog)
        debate_log.write(Text(BANNER_TEXT, style="bold green"))
        debate_log.write(Text("Ready. Enter a target path to begin scan.", style="dim green"))

        self._populate_agent_roster()
        self._update_status("IDLE", "Awaiting target...")

    def _populate_agent_roster(self) -> None:
        """Show all agents in the roster panel."""
        agent_log = self.query_one("#agent-log", RichLog)

        agents_info = [
            ("ELLIOT", "Recon & ASM", "CORE"),
            ("MR.ROBOT", "Exploit Chains", "CORE"),
            ("DARLENE", "Auth/Logic Flaws", "CORE"),
            ("WHITEROSE", "Jury/Intel Oracle", "CORE"),
            ("IRVING", "Orchestrator", "CORE"),
            ("ROMERO", "Legacy/Malware", "SPEC"),
            ("MOBLEY", "Network/Cloud", "SPEC"),
            ("TRENTON", "Persistence/APT", "SPEC"),
            ("TYRELL", "Insider Threat", "SPEC"),
            ("ANGELA", "Social Eng.", "SPEC"),
            ("DOM", "Blue Team", "SPEC"),
            ("LEON", "Remediation", "OUT"),
            ("CISCO", "OSINT Enrich", "OUT"),
        ]

        for name, domain, tier in agents_info:
            style = "bold green" if tier == "CORE" else "green" if tier == "SPEC" else "dim green"
            agent_log.write(Text(f"  [{tier}] {name}", style=style))
            agent_log.write(Text(f"        {domain}", style="dim"))

    def _update_status(self, state: str, detail: str) -> None:
        """Update the status panel."""
        try:
            status_log = self.query_one("#status-log", RichLog)
            status_log.clear()
            status_log.write(Text(f"State: {state}", style="bold green"))
            status_log.write(Text(f"\n{detail}", style="dim green"))
        except NoMatches:
            pass

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle scan button press."""
        if event.button.id == "scan-btn":
            inp = self.query_one("#scan-input", Input)
            target = inp.value.strip()
            if target:
                await self._run_scan(target)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle enter key in input."""
        if event.input.id == "scan-input":
            target = event.value.strip()
            if target:
                await self._run_scan(target)

    async def _run_scan(self, target: str) -> None:
        """Execute a scan (blocking for now; async in future)."""
        debate_log = self.query_one("#debate-log", RichLog)
        self._update_status("SCANNING", f"Target: {target}")

        debate_log.write(Text(f"\n▶ Scanning: {target}", style="bold yellow"))
        debate_log.write(Text("  Initializing agents...", style="dim green"))

        try:
            from fsociety.orchestrator import VAPTOrchestrator
            from fsociety.config import FsocietyConfig

            config = FsocietyConfig()
            orch = VAPTOrchestrator(config=config)

            def _round_cb(round_num, posteriors, findings, events):
                p = posteriors[-1] if posteriors else 0.5
                new = sum(1 for f in findings if f.get("round") == round_num)
                try:
                    debate_log.write(Text(
                        f"  Round {round_num} │ Posterior: {p:.1%} │ +{new} findings",
                        style="cyan",
                    ))
                    self._update_status(
                        f"ROUND {round_num}",
                        f"Posterior: {p:.1%}\nFindings: {len(findings)}",
                    )
                except Exception:
                    pass

            result = orch.scan(path=target, round_callback=_round_cb)
            total = len(result.get("findings", []))
            debate_log.write(Text(f"\n✓ Scan complete — {total} findings", style="bold green"))
            self._update_status("COMPLETE", f"Total findings: {total}")

        except Exception as e:
            debate_log.write(Text(f"\n✗ Error: {e}", style="bold red"))
            self._update_status("ERROR", str(e))


if __name__ == "__main__":
    FsocietyTUI().run()
