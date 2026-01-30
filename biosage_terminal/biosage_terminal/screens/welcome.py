"""
Welcome/Landing screen for BioSage Terminal.
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical, Center
from textual.widgets import Static, Button, Footer
from rich.text import Text

try:
    import pyfiglet
    HAS_PYFIGLET = True
except ImportError:
    HAS_PYFIGLET = False


class WelcomeScreen(Screen):
    """Welcome/Landing screen with hero section and feature cards."""
    
    BINDINGS = [
        ("d", "goto_dashboard", "Dashboard"),
        ("o", "goto_onboarding", "Onboarding"),
        ("l", "goto_cases", "View Cases"),
        ("s", "goto_specialists", "Specialists"),
        ("e", "goto_evidence", "Evidence"),
        ("m", "goto_model_ops", "Model Ops"),
        ("a", "goto_audit", "Audit"),
        ("c", "goto_collaboration", "Collab"),
        ("r", "goto_research", "Research"),
        ("v", "goto_visual", "Visual Dx"),
        ("q", "quit", "Quit"),
    ]
    
    CSS = """
    WelcomeScreen {
        background: #0F172A;
    }
    
    .welcome-container {
        align: center middle;
        padding: 2;
        height: 100%;
    }
    
    .hero-section {
        align: center middle;
        height: auto;
        padding: 2;
    }
    
    .hero-title {
        text-align: center;
        margin-bottom: 1;
    }
    
    .hero-subtitle {
        text-align: center;
        margin-bottom: 2;
    }
    
    .action-buttons {
        layout: horizontal;
        align: center middle;
        height: auto;
        margin: 2 0;
    }
    
    .action-button {
        width: 24;
        height: 3;
        margin: 0 1;
        background: #1E293B;
        border: solid #3B82F6;
        text-style: bold;
    }
    
    .action-button:hover {
        background: #3B82F6;
        color: #FFFFFF;
    }
    
    .action-button:focus {
        background: #2563EB;
        color: #FFFFFF;
    }
    
    .feature-section {
        layout: horizontal;
        align: center middle;
        height: auto;
        margin-top: 2;
    }
    
    .feature-card {
        width: 30;
        height: 12;
        background: #1E293B;
        border: round #334155;
        padding: 1;
        margin: 0 1;
    }
    
    .feature-card:hover {
        border: round #3B82F6;
    }
    
    .llm-status {
        align: center middle;
        margin-top: 2;
        padding: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        with Container(classes="welcome-container"):
            with Vertical(classes="hero-section"):
                yield Static(self._render_hero_title(), classes="hero-title")
                yield Static(self._render_hero_subtitle(), classes="hero-subtitle")
                
                with Horizontal(classes="action-buttons"):
                    yield Button("New Patient", id="btn_onboarding", classes="action-button")
                    yield Button("Dashboard", id="btn_dashboard", classes="action-button")
                    yield Button("View Cases", id="btn_cases", classes="action-button")
                
                with Horizontal(classes="feature-section"):
                    yield Static(self._render_feature_card(
                        "[AI]",
                        "AI Specialists",
                        "6 medical specialists powered by advanced AI reasoning"
                    ), classes="feature-card")
                    yield Static(self._render_feature_card(
                        "[EV]",
                        "Evidence-Based",
                        "ARGUS debate framework for reliable diagnostics"
                    ), classes="feature-card")
                    yield Static(self._render_feature_card(
                        "[SC]",
                        "Secure & Local",
                        "All data stored locally with full audit trail"
                    ), classes="feature-card")
                
                yield Static(self._render_llm_status(), id="llm_status", classes="llm-status")
        
        yield Footer()
    
    def _render_hero_title(self) -> Text:
        """Render the hero title with a clean Unicode-based logo."""
        text = Text(justify="center")
        text.append("=" * 60 + "\n", style="#334155")
        text.append("\n", style="")
        
        # Clean Unicode block-style logo that works in all terminals
        logo_lines = [
            "  ██████╗ ██╗ ██████╗ ███████╗ █████╗  ██████╗ ███████╗",
            "  ██╔══██╗██║██╔═══██╗██╔════╝██╔══██╗██╔════╝ ██╔════╝",
            "  ██████╔╝██║██║   ██║███████╗███████║██║  ███╗█████╗  ",
            "  ██╔══██╗██║██║   ██║╚════██║██╔══██║██║   ██║██╔══╝  ",
            "  ██████╔╝██║╚██████╔╝███████║██║  ██║╚██████╔╝███████╗",
            "  ╚═════╝ ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝",
        ]
        
        # Apply gradient colors
        colors = ["#3B82F6", "#3B82F6", "#60A5FA", "#60A5FA", "#93C5FD", "#BFDBFE"]
        
        for i, line in enumerate(logo_lines):
            color = colors[i % len(colors)]
            text.append(line + "\n", style=f"bold {color}")
        
        text.append("\n", style="")
        text.append("=" * 60, style="#334155")
        return text
    
    def _render_hero_subtitle(self) -> Text:
        """Render the hero subtitle."""
        text = Text(justify="center")
        text.append("AI-Powered Medical Diagnostic Assistant\n", style="bold #F1F5F9")
        text.append("Evidence-Based Reasoning with ARGUS Framework", style="#94A3B8")
        return text
    
    def _render_feature_card(self, icon: str, title: str, description: str) -> Text:
        """Render a feature card."""
        text = Text()
        text.append(f"{icon}\n", style="bold #3B82F6")
        text.append(f"{title}\n\n", style="bold #F1F5F9")
        text.append(description, style="#94A3B8")
        return text
    
    def _render_llm_status(self) -> Text:
        """Render LLM status indicator."""
        from biosage_terminal.ai import check_llm_availability
        
        status = check_llm_availability()
        text = Text(justify="center")
        
        if status["recommended"]:
            provider = status["recommended"]["provider"]
            model = status["recommended"]["model"]
            text.append("[ON] ", style="bold #22C55E")
            text.append(f"LLM Ready: {provider.title()} ({model})", style="#94A3B8")
        else:
            text.append("[--] ", style="bold #EF4444")
            text.append("No LLM configured. Set GEMINI_API_KEY or other API key.", style="#94A3B8")
        
        return text
    
    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        pass
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn_onboarding":
            self.action_goto_onboarding()
        elif button_id == "btn_dashboard":
            self.action_goto_dashboard()
        elif button_id == "btn_cases":
            self.action_goto_cases()
    
    def action_goto_dashboard(self) -> None:
        """Navigate to dashboard."""
        self.app.push_screen("dashboard")
    
    def action_goto_onboarding(self) -> None:
        """Navigate to patient onboarding."""
        self.app.push_screen("onboarding")
    
    def action_goto_cases(self) -> None:
        """Navigate to cases manager."""
        self.app.push_screen("cases_manager")
    
    def action_goto_specialists(self) -> None:
        """Navigate to specialist grid."""
        self.app.push_screen("specialists")
    
    def action_goto_evidence(self) -> None:
        """Navigate to evidence explorer."""
        self.app.push_screen("evidence")
    
    def action_goto_model_ops(self) -> None:
        """Navigate to model operations."""
        self.app.push_screen("model_operations")
    
    def action_goto_audit(self) -> None:
        """Navigate to audit logs."""
        self.app.push_screen("admin_audit")
    
    def action_goto_collaboration(self) -> None:
        """Navigate to collaboration room."""
        self.app.push_screen("collaboration")
    
    def action_goto_research(self) -> None:
        """Navigate to research hub."""
        self.app.push_screen("research")
    
    def action_goto_visual(self) -> None:
        """Navigate to visual diagnosis."""
        self.app.push_screen("visual")
    
    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()
