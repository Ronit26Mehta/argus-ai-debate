"""
Argus Terminal Sandbox Application.

Main Textual application with Bloomberg-style theme and multi-screen navigation.
"""

from pathlib import Path
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Static

from argus_terminal.config import AppConfig
from argus_terminal.widgets.header import HeaderWidget
from argus_terminal.widgets.footer import FooterWidget
from argus_terminal.widgets.sidebar import SidebarWidget, SidebarItem
from argus_terminal.widgets.command_input import CommandInput

from argus_terminal.screens.main import MainScreen
from argus_terminal.screens.debate import DebateScreen
from argus_terminal.screens.providers import ProvidersScreen
from argus_terminal.screens.tools import ToolsScreen
from argus_terminal.screens.knowledge import KnowledgeScreen
from argus_terminal.screens.benchmark import BenchmarkScreen
from argus_terminal.screens.settings import SettingsScreen
from argus_terminal.screens.help import HelpScreen


# CSS file path
CSS_PATH = Path(__file__).parent / "themes"


class ArgusTerminalApp(App):
    """
    Argus Terminal Sandbox - Bloomberg-style TUI for AI Debate System.
    
    A comprehensive terminal application featuring:
    - 1980s Amber phosphor theme (default)
    - 1970s Green phosphor theme
    - Multi-agent debate system
    - 27+ LLM provider support
    - 19 integrated tools
    - Knowledge connectors
    - Benchmarking suite
    """
    
    # Application title
    TITLE = "ARGUS TERMINAL"
    SUB_TITLE = "Agentic Research & Governance Unified System"
    
    # Enable command palette
    ENABLE_COMMAND_PALETTE = True
    
    # Default CSS - embedded for PyInstaller compatibility
    CSS = """
    /* Base styles embedded for distribution */
    
    $background: #0a0a0a;
    $background-light: #151515;
    $background-panel: #0d0d0d;
    
    $primary: #ffb000;
    $primary-dim: #cc8c00;
    $primary-bright: #ffcc33;
    
    $secondary: #ff8c00;
    $accent: #9400d3;
    $accent-bright: #b833ff;
    
    $border-dim: #4d3800;
    $border-normal: #665200;
    $border-bright: #997a00;
    
    $text-primary: #ffb000;
    $text-secondary: #ff8c00;
    $text-dim: #806000;
    
    $success: #33ff33;
    $warning: #ffcc00;
    $error: #ff3333;
    
    Screen {
        background: $background;
        color: $text-primary;
    }
    
    /* Header */
    #app-header {
        dock: top;
        height: 3;
        width: 100%;
        background: $background-panel;
        border-bottom: solid $border-normal;
    }
    
    /* Footer */
    #app-footer {
        dock: bottom;
        height: 3;
        width: 100%;
        background: $background-panel;
        border-top: solid $border-normal;
    }
    
    /* Sidebar */
    #app-sidebar {
        dock: left;
        width: 26;
        height: 100%;
        background: $background-panel;
        border-right: solid $border-normal;
    }
    
    /* Main content */
    #app-main {
        width: 100%;
        height: 100%;
    }
    
    /* Welcome screen styles */
    #welcome-container {
        width: 100%;
        height: 100%;
        align: center middle;
    }
    
    #welcome-box {
        width: 70%;
        height: auto;
        max-width: 80;
        background: $background-panel;
        border: solid $border-bright;
        padding: 2;
    }
    
    #welcome-logo {
        width: 100%;
        height: 5;
        content-align: center middle;
        color: $primary-bright;
        text-style: bold;
    }
    
    #welcome-title {
        width: 100%;
        height: 2;
        content-align: center middle;
        color: $primary;
        text-style: bold;
    }
    
    #welcome-subtitle {
        width: 100%;
        height: 1;
        content-align: center middle;
        color: $secondary;
    }
    
    #welcome-message {
        width: 100%;
        height: auto;
        margin-top: 2;
        content-align: center middle;
        color: $text-dim;
    }
    
    #welcome-shortcuts {
        width: 100%;
        height: auto;
        margin-top: 2;
        color: $accent-bright;
        content-align: center middle;
    }
    
    /* Scrollbar styling */
    * {
        scrollbar-color: #665200 #0a0a0a;
        scrollbar-background: #0a0a0a;
        scrollbar-background-hover: #151515;
        scrollbar-background-active: #1a1400;
        scrollbar-corner-color: #0a0a0a;
        scrollbar-size: 1 1;
    }
    
    ScrollableContainer {
        scrollbar-gutter: stable;
    }
    
    Tree {
        scrollbar-background: #0a0a0a;
        scrollbar-color: #665200;
    }
    
    /* Prevent layout breaks and overflow issues */
    Container, Vertical, Horizontal {
        overflow: hidden;
    }
    
    /* Ensure proper box model for all widgets */
    Static {
        overflow: hidden;
    }
    
    /* Minimum terminal size handling */
    Screen {
        min-width: 80;
        min-height: 24;
    }
    
    /* Fix border rendering */
    .panel, #header-bar {
        box-sizing: border-box;
    }
    """
    
    # Key bindings - F-keys and number keys as alternatives
    BINDINGS = [
        # F-keys
        Binding("f1", "switch_mode('main')", "Dashboard", show=True),
        Binding("f2", "switch_mode('debate')", "Debate", show=True),
        Binding("f3", "switch_mode('providers')", "Providers", show=True),
        Binding("f4", "switch_mode('tools')", "Tools", show=True),
        Binding("f5", "switch_mode('knowledge')", "Knowledge", show=True),
        Binding("f6", "switch_mode('benchmark')", "Benchmark", show=True),
        Binding("f7", "switch_mode('settings')", "Settings", show=True),
        Binding("f8", "switch_mode('help')", "Help", show=True),
        # Number keys as alternatives (for terminals that capture F-keys)
        Binding("1", "switch_mode('main')", "1:Main", show=False),
        Binding("2", "switch_mode('debate')", "2:Debate", show=False),
        Binding("3", "switch_mode('providers')", "3:Providers", show=False),
        Binding("4", "switch_mode('tools')", "4:Tools", show=False),
        Binding("5", "switch_mode('knowledge')", "5:Knowledge", show=False),
        Binding("6", "switch_mode('benchmark')", "6:Benchmark", show=False),
        Binding("7", "switch_mode('settings')", "7:Settings", show=False),
        Binding("8", "switch_mode('help')", "8:Help", show=False),
        # Control keys
        Binding("q", "quit", "Quit", show=True),
        Binding("ctrl+q", "quit", "Quit", show=False),
        Binding("ctrl+d", "toggle_dark", "Toggle Dark", show=False),
        Binding("escape", "switch_mode('main')", "Back", show=False),
    ]
    
    # Register screens as modes
    MODES = {
        "main": MainScreen,
        "debate": DebateScreen,
        "providers": ProvidersScreen,
        "tools": ToolsScreen,
        "knowledge": KnowledgeScreen,
        "benchmark": BenchmarkScreen,
        "settings": SettingsScreen,
        "help": HelpScreen,
    }
    
    # Default mode when app starts
    DEFAULT_MODE = "main"
    
    def __init__(self, theme: str = "amber"):
        """Initialize the Argus Terminal application.

        Args:
            theme: Color theme ("amber" or "green")
        """
        super().__init__()
        # Store our custom theme (not conflicting with Textual's current_theme)
        self._app_theme = theme
        self.config = AppConfig.load()
    
    def set_app_theme(self, theme: str) -> None:
        """Set the application theme.
        
        Args:
            theme: Theme name ("amber" or "green")
        """
        if theme not in ("amber", "green"):
            raise ValueError("Theme must be either 'amber' or 'green'")
        self._app_theme = theme
    
    def get_app_theme(self) -> str:
        """Get the current application theme name."""
        return self._app_theme
    
    def on_mount(self) -> None:
        """Handle application mount."""
        # Log startup
        self.log.info("Argus Terminal started")
    
    def action_switch_mode(self, mode: str) -> None:
        """Switch to a different mode/screen."""
        if mode in self.MODES:
            self.switch_mode(mode)
    
    def action_toggle_dark(self) -> None:
        """Toggle dark mode (themes are both dark, this is a placeholder)."""
        self.dark = not self.dark
    
    def watch_dark(self, dark: bool) -> None:
        """React to dark mode changes."""
        pass  # Both themes are dark, no action needed


class WelcomeScreen(Screen):
    """Initial welcome screen."""
    
    DEFAULT_CSS = """
    WelcomeScreen {
        align: center middle;
    }
    """
    
    LOGO = """
    ╔═══════════════════════════════════════════════════════╗
    ║     █████╗ ██████╗  ██████╗ ██╗   ██╗███████╗         ║
    ║    ██╔══██╗██╔══██╗██╔════╝ ██║   ██║██╔════╝         ║
    ║    ███████║██████╔╝██║  ███╗██║   ██║███████╗         ║
    ║    ██╔══██║██╔══██╗██║   ██║██║   ██║╚════██║         ║
    ║    ██║  ██║██║  ██║╚██████╔╝╚██████╔╝███████║         ║
    ║    ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚══════╝         ║
    ╚═══════════════════════════════════════════════════════╝
    """
    
    def compose(self) -> ComposeResult:
        """Compose the welcome screen."""
        with Container(id="welcome-container"):
            with Vertical(id="welcome-box"):
                yield Static(self.LOGO, id="welcome-logo")
                yield Static(
                    "ARGUS TERMINAL SANDBOX",
                    id="welcome-title"
                )
                yield Static(
                    "Agentic Research & Governance Unified System",
                    id="welcome-subtitle"
                )
                yield Static(
                    "Press any key to continue...",
                    id="welcome-message"
                )
                yield Static(
                    "F1: Dashboard │ F2: Debate │ F8: Help │ Ctrl+Q: Quit",
                    id="welcome-shortcuts"
                )
    
    def on_key(self) -> None:
        """Switch to main screen on any key press."""
        self.app.switch_mode("main")


# Entry point for running directly
if __name__ == "__main__":
    app = ArgusTerminalApp()
    app.run()
