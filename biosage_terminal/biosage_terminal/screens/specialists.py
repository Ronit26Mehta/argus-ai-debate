"""
Specialist Grid screen for BioSage Terminal.
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Footer
from textual.binding import Binding
from rich.text import Text

from biosage_terminal.models import Specialty, AgentStatus
from biosage_terminal.theme import SPECIALTY_ICONS, SPECIALTY_COLORS, STATUS_COLORS
from biosage_terminal.storage import get_data_store


class SpecialistGridScreen(Screen):
    """Grid view of all AI specialist agents."""
    
    BINDINGS = [
        Binding("escape", "go_back", "Back"),
        Binding("d", "goto_dashboard", "Dashboard"),
        Binding("q", "quit", "Quit"),
    ]
    
    CSS = """
    SpecialistGridScreen {
        background: #0F172A;
    }
    
    .specialist-header {
        dock: top;
        height: 3;
        background: #1E293B;
        border-bottom: solid #334155;
        padding: 0 2;
    }
    
    .specialist-grid {
        layout: grid;
        grid-size: 3 2;
        grid-gutter: 2;
        padding: 2;
        height: 100%;
    }
    
    .specialist-card {
        height: 18;
        background: #1E293B;
        border: solid #334155;
        padding: 1;
    }
    
    .specialist-card:hover {
        border: solid #3B82F6;
    }
    
    .specialist-card.infectious {
        border-left: wide #EF4444;
    }
    
    .specialist-card.cardiology {
        border-left: wide #F97316;
    }
    
    .specialist-card.neurology {
        border-left: wide #A855F7;
    }
    
    .specialist-card.oncology {
        border-left: wide #EC4899;
    }
    
    .specialist-card.autoimmune {
        border-left: wide #EAB308;
    }
    
    .specialist-card.toxicology {
        border-left: wide #22C55E;
    }
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.specialist_stats = {}
        self._load_stats()
    
    def _load_stats(self) -> None:
        """Load specialist statistics from storage."""
        store = get_data_store()
        all_cases = store.cases.list_all_sync()
        
        # Count cases and diagnoses by specialist
        for specialty in Specialty:
            self.specialist_stats[specialty] = {
                "cases": 0,
                "recent_diagnoses": [],
                "avg_confidence": 0.0,
            }
        
        for case in all_cases:
            diagnoses = case.get("diagnoses", [])
            for diag in diagnoses:
                spec = diag.get("specialist", "")
                try:
                    specialty = Specialty(spec)
                    self.specialist_stats[specialty]["cases"] += 1
                    self.specialist_stats[specialty]["recent_diagnoses"].append(
                        (diag.get("diagnosis", ""), diag.get("confidence", 0))
                    )
                except (ValueError, KeyError):
                    pass
        
        # Calculate average confidence
        for specialty in Specialty:
            diagnoses = self.specialist_stats[specialty]["recent_diagnoses"]
            if diagnoses:
                total_conf = sum(d[1] for d in diagnoses)
                self.specialist_stats[specialty]["avg_confidence"] = total_conf / len(diagnoses)
    
    def compose(self) -> ComposeResult:
        yield Static(self._render_header(), classes="specialist-header")
        
        with ScrollableContainer():
            with Container(classes="specialist-grid"):
                for specialty in Specialty:
                    yield Static(
                        self._render_specialist_card(specialty),
                        classes=f"specialist-card {specialty.value}",
                    )
        
        yield Footer()
    
    def _render_header(self) -> Text:
        """Render the header."""
        text = Text()
        text.append("BioSage", style="bold #3B82F6")
        text.append(" | ", style="#334155")
        text.append("AI Specialist Grid", style="bold #F1F5F9")
        return text
    
    def _render_specialist_card(self, specialty: Specialty) -> Text:
        """Render a specialist card."""
        text = Text()
        
        names = {
            Specialty.INFECTIOUS: "Infectious Disease",
            Specialty.CARDIOLOGY: "Cardiology",
            Specialty.NEUROLOGY: "Neurology",
            Specialty.ONCOLOGY: "Oncology",
            Specialty.AUTOIMMUNE: "Autoimmune",
            Specialty.TOXICOLOGY: "Toxicology",
        }
        
        descriptions = {
            Specialty.INFECTIOUS: "Bacterial, viral, fungal, and parasitic infections",
            Specialty.CARDIOLOGY: "Heart and cardiovascular system disorders",
            Specialty.NEUROLOGY: "Brain, spine, and nervous system conditions",
            Specialty.ONCOLOGY: "Cancer detection and tumor analysis",
            Specialty.AUTOIMMUNE: "Immune system disorders and rheumatology",
            Specialty.TOXICOLOGY: "Poisoning, drug effects, and toxic exposure",
        }
        
        icon = SPECIALTY_ICONS.get(specialty.value, "[???]")
        name = names.get(specialty, "Unknown")
        description = descriptions.get(specialty, "")
        color = SPECIALTY_COLORS.get(specialty.value, "#3B82F6")
        
        stats = self.specialist_stats.get(specialty, {})
        cases = stats.get("cases", 0)
        avg_conf = stats.get("avg_confidence", 0)
        recent = stats.get("recent_diagnoses", [])[-3:]
        
        # Header
        text.append(f"{icon} {name}\n", style=f"bold {color}")
        text.append("[ON] Online\n", style="bold #22C55E")
        text.append("-" * 30 + "\n", style="#334155")
        
        # Description
        text.append(f"{description}\n\n", style="#94A3B8")
        
        # Stats
        text.append("Cases: ", style="#64748B")
        text.append(f"{cases}\n", style="#F1F5F9")
        
        text.append("Avg Confidence: ", style="#64748B")
        conf_bar = self._render_mini_progress(avg_conf)
        text.append(f"{conf_bar} {int(avg_conf * 100)}%\n\n", style="#94A3B8")
        
        # Recent diagnoses
        if recent:
            text.append("Recent:\n", style="#64748B")
            for diag, conf in recent:
                diag_short = diag[:18] + "..." if len(diag) > 18 else diag
                text.append(f"  - {diag_short} ({int(conf * 100)}%)\n", style="#94A3B8")
        else:
            text.append("No recent diagnoses\n", style="#64748B")
        
        return text
    
    def _render_mini_progress(self, value: float, width: int = 10) -> str:
        """Render a small progress bar."""
        filled = int(value * width)
        empty = width - filled
        return f"[{'#' * filled}{'-' * empty}]"
    
    def action_go_back(self) -> None:
        """Go back to previous screen."""
        self.app.pop_screen()
    
    def action_goto_dashboard(self) -> None:
        """Navigate to dashboard."""
        self.app.pop_screen()
        self.app.push_screen("dashboard")
    
    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()
