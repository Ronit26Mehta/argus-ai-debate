"""
fsociety CLI — terminal entry point.

Usage:
    fsociety scan --path /path/to/code
    fsociety agents
    fsociety --help
"""

from __future__ import annotations

import logging
import sys

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

# ASCII banner
BANNER = r"""
    ██╗███████╗ ██████╗  ██████╗██╗███████╗████████╗██╗   ██╗
   ██╔╝██╔════╝██╔════╝ ██╔═══██╗██╔════╝╚══██╔══╝╚██╗ ██╔╝
  ██╔╝ █████╗  ╚█████╗  ██║   ██║██║        ██║    ╚████╔╝
 ██╔╝  ██╔══╝   ╚═══██╗ ██║   ██║██║        ██║     ╚██╔╝
██╔╝   ██║     ██████╔╝ ╚██████╔╝╚██████╗   ██║      ██║
╚═╝    ╚═╝     ╚═════╝   ╚═════╝  ╚═════╝   ╚═╝      ╚═╝
    VAPT Intelligence Terminal — "Hello, friend."
"""


@click.group()
@click.version_option(version="1.0.0", prog_name="fsociety")
@click.option("--debug", is_flag=True, help="Enable debug logging")
def main(debug: bool) -> None:
    """fsociety — Multi-Agent VAPT Intelligence Terminal."""
    if debug:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s | %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")


@main.command()
@click.option("--path", "-p", type=click.Path(exists=True), help="Path to codebase")
@click.option("--repo", "-r", help="GitHub/GitLab repo URL")
@click.option("--url", "-u", help="Live web target URL")
@click.option("--depth", type=click.Choice(["surface", "deep"]), default="surface")
@click.option("--rounds", "-n", type=int, default=6, help="Max debate rounds")
@click.option("--model", "-m", default="local-model", help="Model name for local server")
@click.option("--base-url", default="http://localhost:8080", help="Local LLM server URL")
@click.option("--output", "-o", default="./fsociety_reports", help="Output directory")
def scan(
    path: str | None,
    repo: str | None,
    url: str | None,
    depth: str,
    rounds: int,
    model: str,
    base_url: str,
    output: str,
) -> None:
    """Run a VAPT scan on a target."""
    console.print(BANNER, style="bold green")

    if not any([path, repo, url]):
        console.print("[red]Error: specify at least one of --path, --repo, or --url[/]")
        sys.exit(1)

    from fsociety.config import FsocietyConfig, LLMConfig, ScanConfig
    from fsociety.orchestrator import VAPTOrchestrator

    config = FsocietyConfig(
        llm=LLMConfig(
            base_url=base_url,
            model_name=model,
        ),
        scan=ScanConfig(
            depth=depth,
            max_debate_rounds=rounds,
        ),
        output_dir=output,
    )

    orchestrator = VAPTOrchestrator(config=config)

    # Progress callback
    def _on_round(round_num, posteriors, findings, events):
        p = posteriors[-1] if posteriors else 0.5
        new = sum(1 for f in findings if f.get("round") == round_num)
        console.print(
            f"  [cyan]Round {round_num}[/] │ "
            f"Posterior: [yellow]{p:.1%}[/] │ "
            f"New findings: [green]{new}[/]"
        )

    with console.status("[bold green]Initializing fsociety...", spinner="aesthetic"):
        result = orchestrator.scan(
            path=path,
            repo_url=repo,
            url=url,
            round_callback=_on_round,
        )

    # Summary
    console.print()
    _print_summary(result)


@main.command()
def agents() -> None:
    """List all fsociety agents and their roles."""
    console.print(BANNER, style="bold green")

    from fsociety.agents import ALL_AGENTS

    table = Table(title="fsociety Agent Roster", show_lines=True)
    table.add_column("Persona", style="bold cyan")
    table.add_column("Tier", style="yellow")
    table.add_column("Domain", style="green")
    table.add_column("RDC Role", style="magenta")

    for AgentClass in ALL_AGENTS:
        table.add_row(
            AgentClass.PERSONA_NAME,
            AgentClass.TIER.value,
            AgentClass.VAPT_DOMAIN,
            AgentClass.RDC_ROLE,
        )

    console.print(table)


@main.command()
def tui() -> None:
    """Launch the fsociety Terminal UI."""
    console.print(BANNER, style="bold green")
    console.print("[yellow]Launching TUI...[/]")

    from fsociety.tui import FsocietyTUI
    app = FsocietyTUI()
    app.run()


@main.command()
def sessions() -> None:
    """List past scan sessions."""
    from pathlib import Path

    reports_dir = Path("./fsociety_reports")
    if not reports_dir.exists():
        console.print("[yellow]No sessions found. Run 'fsociety scan' first.[/]")
        return

    table = Table(title="Scan Sessions")
    table.add_column("Target", style="cyan")
    table.add_column("Session ID", style="green")
    table.add_column("Status")

    for target_dir in sorted(reports_dir.iterdir()):
        if target_dir.is_dir():
            for session_dir in sorted(target_dir.iterdir()):
                if session_dir.is_dir():
                    has_report = (session_dir / "report" / "executive_summary.md").exists()
                    status = "✅ Complete" if has_report else "⏳ Partial"
                    table.add_row(target_dir.name, session_dir.name, status)

    console.print(table)


def _print_summary(result: dict) -> None:
    """Print scan result summary."""
    findings = result.get("findings", [])
    posteriors = result.get("posteriors", [])
    report_path = result.get("report_path")

    # Severity counts
    sev_counts = {"P0": 0, "P1": 0, "P2": 0, "P3": 0, "INFO": 0}
    for f in findings:
        sev = f.get("severity", "P2")
        if sev in sev_counts:
            sev_counts[sev] += 1

    table = Table(title="Scan Results Summary", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total findings", str(len(findings)))
    table.add_row("🔴 P0 Critical", str(sev_counts["P0"]))
    table.add_row("🟠 P1 High", str(sev_counts["P1"]))
    table.add_row("🟡 P2 Medium", str(sev_counts["P2"]))
    table.add_row("🟢 P3 Low", str(sev_counts["P3"]))
    table.add_row("Debate rounds", str(len(posteriors) - 1 if posteriors else 0))
    table.add_row("Final posterior", f"{posteriors[-1]:.1%}" if posteriors else "N/A")
    if report_path:
        table.add_row("Report", str(report_path))

    console.print(table)


if __name__ == "__main__":
    main()
