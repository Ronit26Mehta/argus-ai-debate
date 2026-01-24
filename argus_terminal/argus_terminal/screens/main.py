"""
Main Dashboard Screen - Central hub for Argus Terminal.

Features:
- Overview statistics
- Quick access panels
- Recent activity
- System status
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical, Grid, ScrollableContainer
from textual.widgets import Static, Button


class MainScreen(Screen):
    """Main dashboard screen."""
    
    DEFAULT_CSS = """
    MainScreen {
        background: #0a0a0a;
    }
    
    MainScreen #main-scroll {
        width: 100%;
        height: 100%;
    }
    
    MainScreen #main-container {
        width: 100%;
        height: auto;
        padding: 1;
    }
    
    MainScreen #welcome-banner {
        width: 100%;
        height: 5;
        background: #0d0d0d;
        border: solid #665200;
        padding: 1;
        margin-bottom: 1;
    }
    
    MainScreen #banner-title {
        width: 100%;
        height: 1;
        color: #ffcc33;
        text-style: bold;
        content-align: center middle;
    }
    
    MainScreen #banner-subtitle {
        width: 100%;
        height: 1;
        color: #ff8c00;
        content-align: center middle;
    }
    
    MainScreen #stats-row {
        width: 100%;
        height: 4;
        layout: horizontal;
        margin-bottom: 1;
    }
    
    MainScreen .stat-box {
        width: 1fr;
        height: 100%;
        background: #0d0d0d;
        border: solid #4d3800;
        padding: 0 1;
        margin-right: 1;
    }
    
    MainScreen .stat-label {
        color: #ff8c00;
        margin-top: 1;
    }
    
    MainScreen .stat-value {
        color: #ffcc33;
        text-style: bold;
    }
    
    MainScreen #actions-title {
        width: 100%;
        height: 2;
        color: #ffcc33;
        text-style: bold;
    }
    
    MainScreen #actions-grid {
        width: 100%;
        height: auto;
        layout: grid;
        grid-size: 4;
        grid-gutter: 1;
        margin-bottom: 1;
    }
    
    MainScreen .action-btn {
        width: 100%;
        height: 5;
        background: #0d0d0d;
        border: solid #665200;
        color: #ffcc33;
        text-align: center;
    }
    
    MainScreen .action-btn:hover {
        background: #1a1400;
        border: solid #997a00;
    }
    
    MainScreen .action-btn:focus {
        border: solid #ffb000;
    }
    
    MainScreen #activity-panel {
        width: 100%;
        min-height: 10;
        background: #0d0d0d;
        border: solid #665200;
        padding: 1;
    }
    
    MainScreen #activity-title {
        width: 100%;
        height: 1;
        color: #ffcc33;
        text-style: bold;
        margin-bottom: 1;
    }
    
    MainScreen #activity-content {
        width: 100%;
        height: auto;
        color: #ffb000;
    }
    
    MainScreen #nav-hint {
        width: 100%;
        height: 1;
        color: #806000;
        content-align: center middle;
        margin-top: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        """Compose the main dashboard layout."""
        with ScrollableContainer(id="main-scroll"):
            with Container(id="main-container"):
                # Welcome banner
                with Vertical(id="welcome-banner"):
                    yield Static(
                        "[bold #ffcc33][*] ARGUS TERMINAL SANDBOX[/]",
                        id="banner-title"
                    )
                    yield Static(
                        "Agentic Research & Governance Unified System - v2.0.0",
                        id="banner-subtitle"
                    )
                
                # Statistics row
                with Horizontal(id="stats-row"):
                    with Vertical(classes="stat-box"):
                        yield Static("LLM Providers", classes="stat-label")
                        yield Static("[bold]27+[/bold]", classes="stat-value")
                    with Vertical(classes="stat-box"):
                        yield Static("Tools Available", classes="stat-label")
                        yield Static("[bold]19[/bold]", classes="stat-value")
                    with Vertical(classes="stat-box"):
                        yield Static("Embedding Models", classes="stat-label")
                        yield Static("[bold]16[/bold]", classes="stat-value")
                    with Vertical(classes="stat-box"):
                        yield Static("Connectors", classes="stat-label")
                        yield Static("[bold]3[/bold]", classes="stat-value")
                
                # Quick actions
                yield Static("[-] QUICK ACTIONS  [dim](Press number keys 1-8 to navigate)[/dim]", id="actions-title")
                
                with Grid(id="actions-grid"):
                    yield Button("[X] [2] Debate\nRun AI debates", id="btn-debate", classes="action-btn")
                    yield Button("[C] [3] Providers\nConfigure LLMs", id="btn-providers", classes="action-btn")
                    yield Button("[*] [4] Tools\nBrowse tools", id="btn-tools", classes="action-btn")
                    yield Button("[B] [5] Knowledge\nConnectors", id="btn-knowledge", classes="action-btn")
                    yield Button("[#] [6] Benchmarks\nEvaluation", id="btn-benchmark", classes="action-btn")
                    yield Button("[S] [7] Settings\nTheme & config", id="btn-settings", classes="action-btn")
                    yield Button("[?] [8] Help\nDocumentation", id="btn-help", classes="action-btn")
                    yield Button("[>] Quick Start\nTutorial", id="btn-quickstart", classes="action-btn")
                
                # Activity panel
                with Vertical(id="activity-panel"):
                    yield Static("[-] RECENT ACTIVITY", id="activity-title")
                    yield Static(
                        "[#ff8c00]>[/] Welcome to ARGUS Terminal!\n"
                        "[#806000]  Press number keys [bold]1-8[/bold] to switch screens[/]\n"
                        "[#806000]  Press [bold]Tab[/bold] to move between buttons[/]\n"
                        "[#806000]  Press [bold]Enter[/bold] to activate buttons[/]\n"
                        "[#806000]  Press [bold]q[/bold] to quit[/]",
                        id="activity-content"
                    )
                
                yield Static(
                    "[dim]Navigation: 1=Main 2=Debate 3=Providers 4=Tools 5=Knowledge 6=Benchmark 7=Settings 8=Help q=Quit[/dim]",
                    id="nav-hint"
                )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_map = {
            "btn-debate": "debate",
            "btn-providers": "providers",
            "btn-tools": "tools",
            "btn-knowledge": "knowledge",
            "btn-benchmark": "benchmark",
            "btn-settings": "settings",
            "btn-help": "help",
        }
        
        if event.button.id in button_map:
            self.app.switch_mode(button_map[event.button.id])
