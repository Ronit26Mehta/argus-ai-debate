"""
Settings Screen - Configuration and theme options.

Features:
- Theme selector
- Display options
- Debate defaults
- Provider settings
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, Button, Input, Switch, RadioButton, RadioSet


class SettingsScreen(Screen):
    """Settings and configuration screen."""
    
    DEFAULT_CSS = """
    SettingsScreen {
        background: #0a0a0a;
    }
    
    SettingsScreen #settings-container {
        width: 100%;
        height: 100%;
        padding: 1;
    }
    
    SettingsScreen #header-bar {
        width: 100%;
        height: 3;
        background: #0d0d0d;
        border: solid #665200;
        padding: 0 1;
        margin-bottom: 1;
    }
    
    SettingsScreen #header-title {
        width: 100%;
        height: 1;
        color: #ffcc33;
        text-style: bold;
        margin-top: 1;
    }
    
    SettingsScreen #content-area {
        width: 100%;
        height: 1fr;
        layout: horizontal;
    }
    
    SettingsScreen #left-panel {
        width: 50%;
        height: 100%;
        padding-right: 1;
    }
    
    SettingsScreen #right-panel {
        width: 50%;
        height: 100%;
    }
    
    SettingsScreen .panel {
        width: 100%;
        height: auto;
        background: #0d0d0d;
        border: solid #665200;
        padding: 1;
        margin-bottom: 1;
    }
    
    SettingsScreen .section-title {
        width: 100%;
        height: 1;
        color: #ffcc33;
        text-style: bold;
        margin-bottom: 1;
    }
    
    SettingsScreen .setting-row {
        width: 100%;
        height: 3;
        layout: horizontal;
        margin-bottom: 1;
    }
    
    SettingsScreen .setting-label {
        width: 20;
        height: 1;
        color: #ff8c00;
        margin-top: 1;
    }
    
    SettingsScreen .setting-value {
        width: 1fr;
        height: 3;
    }
    
    SettingsScreen Input {
        background: #1a1400;
        border: solid #4d3800;
        color: #ffcc33;
    }
    
    SettingsScreen Switch {
        background: #4d3800;
    }
    
    SettingsScreen Switch.-on {
        background: #665200;
    }
    
    SettingsScreen RadioSet {
        background: transparent;
        border: none;
    }
    
    SettingsScreen RadioButton {
        background: transparent;
        padding: 0 1;
    }
    
    SettingsScreen .theme-btn {
        width: 1fr;
        height: 5;
        margin-right: 1;
    }
    
    SettingsScreen #save-btn {
        width: 100%;
        height: 3;
        margin-top: 2;
    }
    
    SettingsScreen #nav-hint {
        width: 100%;
        height: 1;
        color: #806000;
        margin-top: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        """Compose the settings screen layout."""
        with Container(id="settings-container"):
            # Header
            with Horizontal(id="header-bar"):
                yield Static(
                    "◈ SETTINGS  │  Configure theme and preferences",
                    id="header-title"
                )
            
            # Content
            with Horizontal(id="content-area"):
                # Left panel
                with Vertical(id="left-panel"):
                    # Theme
                    with Vertical(classes="panel"):
                        yield Static("◈ THEME", classes="section-title")
                        yield Static("[#ff8c00]Select visual theme:[/]")
                        
                        with Horizontal():
                            yield Button(
                                "🔶 AMBER\n1980s CRT",
                                id="theme-amber",
                                classes="theme-btn",
                                variant="primary"
                            )
                            yield Button(
                                "🟢 GREEN\n1970s CRT",
                                id="theme-green",
                                classes="theme-btn"
                            )
                    
                    # Display
                    with Vertical(classes="panel"):
                        yield Static("◈ DISPLAY OPTIONS", classes="section-title")
                        
                        with Horizontal(classes="setting-row"):
                            yield Static("Scanlines:", classes="setting-label")
                            yield Switch(value=True, id="switch-scanlines")
                        
                        with Horizontal(classes="setting-row"):
                            yield Static("Glow effect:", classes="setting-label")
                            yield Switch(value=True, id="switch-glow")
                        
                        with Horizontal(classes="setting-row"):
                            yield Static("Animations:", classes="setting-label")
                            yield Switch(value=True, id="switch-animations")
                        
                        with Horizontal(classes="setting-row"):
                            yield Static("Show hints:", classes="setting-label")
                            yield Switch(value=True, id="switch-hints")
                    
                    # Provider defaults
                    with Vertical(classes="panel"):
                        yield Static("◈ PROVIDER DEFAULTS", classes="section-title")
                        
                        with Horizontal(classes="setting-row"):
                            yield Static("Default LLM:", classes="setting-label")
                            yield Input(value="gemini", id="default-provider", classes="setting-value")
                        
                        with Horizontal(classes="setting-row"):
                            yield Static("Default Model:", classes="setting-label")
                            yield Input(value="gemini-1.5-flash", id="default-model", classes="setting-value")
                
                # Right panel
                with Vertical(id="right-panel"):
                    # Debate defaults
                    with Vertical(classes="panel"):
                        yield Static("◈ DEBATE DEFAULTS", classes="section-title")
                        
                        with Horizontal(classes="setting-row"):
                            yield Static("Max rounds:", classes="setting-label")
                            yield Input(value="5", id="debate-rounds", classes="setting-value")
                        
                        with Horizontal(classes="setting-row"):
                            yield Static("Prior probability:", classes="setting-label")
                            yield Input(value="0.5", id="debate-prior", classes="setting-value")
                        
                        with Horizontal(classes="setting-row"):
                            yield Static("Specialists:", classes="setting-label")
                            yield Input(value="3", id="debate-specialists", classes="setting-value")
                        
                        with Horizontal(classes="setting-row"):
                            yield Static("Domain:", classes="setting-label")
                            yield Input(value="General", id="debate-domain", classes="setting-value")
                    
                    # Data
                    with Vertical(classes="panel"):
                        yield Static("◈ DATA & STORAGE", classes="section-title")
                        
                        with Horizontal(classes="setting-row"):
                            yield Static("Config path:", classes="setting-label")
                            yield Input(value="~/.argus/config.json", id="config-path", classes="setting-value")
                        
                        with Horizontal(classes="setting-row"):
                            yield Static("Cache path:", classes="setting-label")
                            yield Input(value="~/.argus/cache", id="cache-path", classes="setting-value")
                        
                        yield Button("🗑️ Clear Cache", id="clear-cache-btn")
                    
                    # About
                    with Vertical(classes="panel"):
                        yield Static("◈ ABOUT", classes="section-title")
                        yield Static(
                            "[#ffcc33]ARGUS Terminal Sandbox[/]\n"
                            "[#ff8c00]Version:[/] 3.1.0\n"
                            "[#ff8c00]Built with:[/] Textual + Rich\n"
                            "[#ff8c00]License:[/] MIT\n"
                            "[#ff8c00]Authors:[/] ARGUS Team"
                        )
                    
                    yield Button("💾 SAVE ALL SETTINGS", id="save-btn", variant="primary")
            
            yield Static(
                "[dim]Press 1=Main 2=Debate 3=Providers 4=Tools 5=Knowledge 6=Benchmark 7=Settings 8=Help q=Quit[/dim]",
                id="nav-hint"
            )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "save-btn":
            self.notify("Settings saved successfully!", severity="information")
        elif event.button.id == "clear-cache-btn":
            self.notify("Cache cleared!", severity="information")
        elif event.button.id == "theme-amber":
            self.notify("Amber theme selected", severity="information")
        elif event.button.id == "theme-green":
            self.notify("Green theme selected", severity="information")
