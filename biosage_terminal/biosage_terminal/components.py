"""
Reusable UI components for BioSage Terminal.
"""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Button, ProgressBar, Label, DataTable
from textual.widget import Widget
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from typing import Optional

from biosage_terminal.theme import (
    COLORS,
    SPECIALTY_COLORS,
    STATUS_COLORS,
    PRIORITY_COLORS,
    format_progress_bar,
    format_confidence_bar,
    SPECIALTY_ICONS,
    STATUS_INDICATORS,
)


class HeaderBar(Static):
    """Application header bar with title and navigation hints."""
    
    DEFAULT_CSS = """
    HeaderBar {
        dock: top;
        height: 3;
        background: #1E293B;
        border-bottom: solid #334155;
        padding: 0 2;
    }
    """
    
    def __init__(
        self,
        title: str = "BioSage",
        subtitle: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.title_text = title
        self.subtitle_text = subtitle
    
    def compose(self) -> ComposeResult:
        yield Static(self._render_header())
    
    def _render_header(self) -> Text:
        text = Text()
        text.append("BioSage", style="bold #3B82F6")
        if self.title_text and self.title_text != "BioSage":
            text.append(" | ", style="#334155")
            text.append(self.title_text, style="bold #F1F5F9")
        if self.subtitle_text:
            text.append(" - ", style="#334155")
            text.append(self.subtitle_text, style="#94A3B8")
        return text
    
    def update_title(self, title: str, subtitle: str = "") -> None:
        """Update the header title and subtitle."""
        self.title_text = title
        self.subtitle_text = subtitle
        self.update(self._render_header())


class FooterBar(Static):
    """Application footer bar with key bindings."""
    
    DEFAULT_CSS = """
    FooterBar {
        dock: bottom;
        height: 1;
        background: #1E293B;
        border-top: solid #334155;
        padding: 0 2;
    }
    """
    
    def __init__(self, hints: Optional[list[tuple[str, str]]] = None, **kwargs):
        super().__init__(**kwargs)
        self.hints = hints or [
            ("d", "Dashboard"),
            ("o", "Onboarding"),
            ("s", "Specialists"),
            ("e", "Evidence"),
            ("q", "Quit"),
        ]
    
    def compose(self) -> ComposeResult:
        yield Static(self._render_footer())
    
    def _render_footer(self) -> Text:
        text = Text()
        for i, (key, label) in enumerate(self.hints):
            if i > 0:
                text.append("  ", style="#334155")
            text.append(f"[{key}]", style="bold #3B82F6")
            text.append(f" {label}", style="#94A3B8")
        return text
    
    def update_hints(self, hints: list[tuple[str, str]]) -> None:
        """Update the footer hints."""
        self.hints = hints
        self.update(self._render_footer())


class MetricCard(Static):
    """Dashboard metric card displaying a single value."""
    
    DEFAULT_CSS = """
    MetricCard {
        width: 1fr;
        height: 7;
        background: #0F172A;
        border: double #334155;
        padding: 1;
        text-align: center;
    }
    """
    
    def __init__(
        self,
        label: str,
        value: str,
        icon: str = "",
        color: str = "#3B82F6",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.label = label
        self.value_text = value
        self.icon = icon
        self.color = color
    
    def compose(self) -> ComposeResult:
        yield Static(self._render_card())
    
    def _render_card(self) -> Text:
        text = Text(justify="center")
        if self.icon:
            text.append(f"{self.icon}\n", style=f"bold {self.color}")
        text.append(f"{self.value_text}\n", style=f"bold {self.color}")
        text.append(self.label, style="#94A3B8")
        return text
    
    def update_value(self, value: str) -> None:
        """Update the displayed value."""
        self.value_text = value
        self.update(self._render_card())


class AgentCard(Static):
    """Card displaying specialist agent status."""
    
    DEFAULT_CSS = """
    AgentCard {
        height: 12;
        background: #1E293B;
        border: solid #334155;
        padding: 1;
    }
    
    AgentCard.status-online {
        border: solid #22C55E;
    }
    
    AgentCard.status-degraded {
        border: solid #EAB308;
    }
    
    AgentCard.status-offline {
        border: solid #EF4444;
    }
    """
    
    def __init__(
        self,
        specialty: str,
        name: str,
        status: str = "online",
        confidence: float = 0.0,
        cases: int = 0,
        recent_diagnoses: Optional[list[tuple[str, float]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.specialty = specialty
        self.name = name
        self.status = status
        self.confidence = confidence
        self.cases = cases
        self.recent_diagnoses = recent_diagnoses or []
        self.add_class(f"status-{status}")
    
    def compose(self) -> ComposeResult:
        yield Static(self._render_card())
    
    def _render_card(self) -> Text:
        text = Text()
        
        # Header with icon and status
        icon = SPECIALTY_ICONS.get(self.specialty.lower(), "[???]")
        status_ind = STATUS_INDICATORS.get(self.status, "[??]")
        color = SPECIALTY_COLORS.get(self.specialty.lower(), "#3B82F6")
        status_color = STATUS_COLORS.get(self.status, "#64748B")
        
        text.append(f"{icon} ", style=f"bold {color}")
        text.append(self.name, style="bold #F1F5F9")
        text.append(f" {status_ind}\n", style=f"bold {status_color}")
        
        # Confidence bar
        conf_bar = format_progress_bar(self.confidence, width=15)
        text.append(f"Conf: {conf_bar} {int(self.confidence * 100)}%\n", style="#94A3B8")
        
        # Cases count
        text.append(f"Cases: {self.cases}\n", style="#94A3B8")
        
        # Recent diagnoses
        if self.recent_diagnoses:
            text.append("\nRecent:\n", style="#64748B")
            for diag, conf in self.recent_diagnoses[:2]:
                diag_short = diag[:15] + "..." if len(diag) > 15 else diag
                text.append(f"  - {diag_short} ({int(conf * 100)}%)\n", style="#94A3B8")
        
        return text
    
    def update_status(self, status: str) -> None:
        """Update the agent status."""
        self.remove_class(f"status-{self.status}")
        self.status = status
        self.add_class(f"status-{status}")
        self.update(self._render_card())


class DiagnosisCard(Static):
    """Card displaying a diagnosis suggestion."""
    
    DEFAULT_CSS = """
    DiagnosisCard {
        background: #1E293B;
        border: solid #3B82F6;
        padding: 2;
        margin: 1;
        height: auto;
    }
    """
    
    def __init__(
        self,
        diagnosis: str,
        confidence: float,
        rationale: str = "",
        specialist: str = "",
        is_primary: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.diagnosis = diagnosis
        self.confidence = confidence
        self.rationale = rationale
        self.specialist = specialist
        self.is_primary = is_primary
    
    def compose(self) -> ComposeResult:
        yield Static(self._render_card())
    
    def _render_card(self) -> Text:
        text = Text()
        
        # Title
        if self.is_primary:
            text.append("[PRIMARY] ", style="bold #22C55E")
        text.append(f"{self.diagnosis}\n", style="bold #3B82F6")
        
        # Confidence
        conf_bar = format_confidence_bar(self.confidence)
        text.append(f"{conf_bar}\n", style="#F1F5F9")
        
        # Specialist
        if self.specialist:
            icon = SPECIALTY_ICONS.get(self.specialist.lower(), "")
            color = SPECIALTY_COLORS.get(self.specialist.lower(), "#94A3B8")
            text.append(f"{icon} {self.specialist.title()}\n", style=color)
        
        # Rationale
        if self.rationale:
            text.append(f"\n{self.rationale}", style="#94A3B8")
        
        return text


class TestCard(Static):
    """Card displaying a recommended test."""
    
    DEFAULT_CSS = """
    TestCard {
        background: #1E293B;
        border: solid #334155;
        padding: 1;
        margin: 1 0;
        height: auto;
    }
    """
    
    def __init__(
        self,
        test_name: str,
        rationale: str = "",
        priority: str = "medium",
        cost: Optional[float] = None,
        turnaround: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.test_name = test_name
        self.rationale = rationale
        self.priority = priority
        self.cost = cost
        self.turnaround = turnaround
    
    def compose(self) -> ComposeResult:
        yield Static(self._render_card())
    
    def _render_card(self) -> Text:
        text = Text()
        
        # Test name and priority
        priority_color = PRIORITY_COLORS.get(self.priority.lower(), "#94A3B8")
        text.append(f"{self.test_name}", style="bold #F1F5F9")
        text.append(f" [{self.priority.upper()}]", style=f"bold {priority_color}")
        
        # Cost and turnaround
        if self.cost is not None or self.turnaround:
            text.append("\n")
            if self.cost is not None:
                text.append(f"${self.cost:.0f}", style="#22C55E")
            if self.cost is not None and self.turnaround:
                text.append(" | ", style="#334155")
            if self.turnaround:
                text.append(self.turnaround, style="#64748B")
        
        # Rationale
        if self.rationale:
            text.append(f"\n{self.rationale}", style="#94A3B8")
        
        return text


class PatientInfoBar(Static):
    """Bar displaying current patient information."""
    
    DEFAULT_CSS = """
    PatientInfoBar {
        height: 5;
        background: #1E293B;
        border-bottom: solid #334155;
        padding: 1;
    }
    """
    
    def __init__(
        self,
        name: str = "",
        age: int = 0,
        mrn: str = "",
        case_id: str = "",
        chief_complaint: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patient_name = name
        self.patient_age = age
        self.patient_mrn = mrn
        self.case_id = case_id
        self.chief_complaint = chief_complaint
    
    def compose(self) -> ComposeResult:
        yield Static(self._render_bar())
    
    def _render_bar(self) -> Text:
        text = Text()
        text.append("Name: ", style="bold #94A3B8")
        text.append(f"{self.patient_name}", style="#F1F5F9")
        text.append(" | ", style="#334155")
        text.append("Age: ", style="bold #94A3B8")
        text.append(f"{self.patient_age}", style="#F1F5F9")
        text.append(" | ", style="#334155")
        text.append("MRN: ", style="bold #94A3B8")
        text.append(f"{self.patient_mrn}", style="#F1F5F9")
        text.append(" | ", style="#334155")
        text.append("Case: ", style="bold #94A3B8")
        text.append(f"{self.case_id}\n", style="#F1F5F9")
        text.append("Chief Complaint: ", style="bold #94A3B8")
        text.append(self.chief_complaint, style="#F1F5F9")
        return text
    
    def update_patient(
        self,
        name: str,
        age: int,
        mrn: str,
        case_id: str,
        chief_complaint: str,
    ) -> None:
        """Update patient information."""
        self.patient_name = name
        self.patient_age = age
        self.patient_mrn = mrn
        self.case_id = case_id
        self.chief_complaint = chief_complaint
        self.update(self._render_bar())


class LoadingIndicator(Static):
    """Loading indicator with spinner and message."""
    
    DEFAULT_CSS = """
    LoadingIndicator {
        align: center middle;
        height: 100%;
    }
    """
    
    SPINNER_FRAMES = ["|", "/", "-", "\\"]
    
    def __init__(
        self,
        message: str = "Loading...",
        progress: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.message = message
        self.progress = progress
        self._frame = 0
    
    def compose(self) -> ComposeResult:
        yield Static(self._render_loading())
    
    def _render_loading(self) -> Text:
        text = Text(justify="center")
        
        # Title
        text.append("Running Diagnosis\n\n", style="bold #3B82F6")
        
        # Progress bar
        prog_bar = format_progress_bar(self.progress, width=40)
        text.append(f"{prog_bar} {int(self.progress * 100)}%\n\n", style="#F1F5F9")
        
        # Message
        text.append(f"{self.message}\n\n", style="#94A3B8")
        
        # Spinner
        spinner = self.SPINNER_FRAMES[self._frame % len(self.SPINNER_FRAMES)]
        text.append(spinner, style="bold #06B6D4")
        
        return text
    
    def update_progress(self, progress: float, message: str = "") -> None:
        """Update progress and message."""
        self.progress = progress
        if message:
            self.message = message
        self._frame += 1
        self.update(self._render_loading())


class VitalCard(Static):
    """Card displaying a vital sign."""
    
    DEFAULT_CSS = """
    VitalCard {
        height: 8;
        background: #1E293B;
        border: solid #334155;
        padding: 1;
        text-align: center;
    }
    """
    
    def __init__(
        self,
        label: str,
        value: str,
        unit: str = "",
        status: str = "normal",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.label = label
        self.value_text = value
        self.unit = unit
        self.status = status
    
    def compose(self) -> ComposeResult:
        yield Static(self._render_card())
    
    def _render_card(self) -> Text:
        text = Text(justify="center")
        
        # Status color
        status_colors = {
            "normal": "#22C55E",
            "abnormal": "#EAB308",
            "critical": "#EF4444",
        }
        color = status_colors.get(self.status, "#F1F5F9")
        
        # Value
        text.append(f"{self.value_text}", style=f"bold {color}")
        if self.unit:
            text.append(f" {self.unit}", style="#64748B")
        text.append("\n\n", style="")
        
        # Label
        text.append(self.label, style="#94A3B8")
        
        return text


class TimelineEvent(Static):
    """Timeline event entry."""
    
    DEFAULT_CSS = """
    TimelineEvent {
        padding-left: 4;
        border-left: solid #334155;
        margin-bottom: 1;
        height: auto;
    }
    """
    
    def __init__(
        self,
        date: str,
        title: str,
        content: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.date = date
        self.title = title
        self.content = content
    
    def compose(self) -> ComposeResult:
        yield Static(self._render_event())
    
    def _render_event(self) -> Text:
        text = Text()
        text.append("[*] ", style="bold #3B82F6")
        text.append(f"{self.date}: ", style="bold #F1F5F9")
        text.append(f"{self.title}\n", style="#F1F5F9")
        if self.content:
            text.append(f"    {self.content}", style="#94A3B8")
        return text


class AuditLogEntry(Static):
    """Audit log entry display."""
    
    DEFAULT_CSS = """
    AuditLogEntry {
        padding: 1;
        height: auto;
    }
    
    AuditLogEntry:hover {
        background: #1E293B;
    }
    """
    
    def __init__(
        self,
        timestamp: str,
        user: str,
        action: str,
        status: str = "success",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.timestamp = timestamp
        self.user = user
        self.action = action
        self.status = status
    
    def compose(self) -> ComposeResult:
        yield Static(self._render_entry())
    
    def _render_entry(self) -> Text:
        text = Text()
        
        # Timestamp
        text.append(f"{self.timestamp:20}", style="#64748B")
        
        # User
        text.append(f"{self.user:20}", style="#94A3B8")
        
        # Action
        text.append(f"{self.action:30}", style="#F1F5F9")
        
        # Status
        status_styles = {
            "success": ("bold #22C55E", "[OK]"),
            "warning": ("bold #EAB308", "[!!]"),
            "error": ("bold #EF4444", "[XX]"),
            "info": ("bold #06B6D4", "[ii]"),
        }
        style, indicator = status_styles.get(self.status, ("", "[??]"))
        text.append(indicator, style=style)
        
        return text


class EmptyState(Static):
    """Empty state placeholder with message."""
    
    DEFAULT_CSS = """
    EmptyState {
        align: center middle;
        height: 100%;
        padding: 4;
    }
    """
    
    def __init__(
        self,
        title: str = "No Data",
        message: str = "",
        action_label: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.title = title
        self.message = message
        self.action_label = action_label
    
    def compose(self) -> ComposeResult:
        yield Static(self._render_empty())
    
    def _render_empty(self) -> Text:
        text = Text(justify="center")
        text.append(f"{self.title}\n\n", style="bold #64748B")
        if self.message:
            text.append(f"{self.message}\n\n", style="#94A3B8")
        if self.action_label:
            text.append(f"[{self.action_label}]", style="#3B82F6")
        return text
