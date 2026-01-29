"""
Dashboard screen for BioSage Terminal.
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Button, Footer, DataTable
from textual.binding import Binding
from rich.text import Text

from biosage_terminal.storage import get_data_store
from biosage_terminal.components import MetricCard, AgentCard, HeaderBar
from biosage_terminal.models import Specialty, AgentStatus


class DashboardScreen(Screen):
    """Main dashboard with metrics, agent status, and patient tables."""
    
    BINDINGS = [
        Binding("w", "goto_welcome", "Welcome"),
        Binding("o", "goto_onboarding", "Onboarding"),
        Binding("l", "goto_cases", "Cases"),
        Binding("s", "goto_specialists", "Specialists"),
        Binding("e", "goto_evidence", "Evidence"),
        Binding("r", "refresh", "Refresh"),
        Binding("c", "goto_collaboration", "Collab"),
        Binding("h", "goto_research", "Research"),
        Binding("v", "goto_visual", "Visual Dx"),
        Binding("escape", "goto_welcome", "Back"),
        Binding("q", "quit", "Quit"),
    ]
    
    CSS = """
    DashboardScreen {
        background: #0F172A;
    }
    
    .dashboard-header {
        dock: top;
        height: 3;
        background: #1E293B;
        border-bottom: solid #334155;
        padding: 0 2;
    }
    
    .metrics-bar {
        height: 9;
        layout: horizontal;
        padding: 1;
        background: #1E293B;
        border-bottom: solid #334155;
    }
    
    .metric-card {
        width: 1fr;
        height: 7;
        background: #0F172A;
        border: double #334155;
        padding: 1;
        margin: 0 1;
        text-align: center;
    }
    
    .agent-section {
        height: auto;
        padding: 1;
    }
    
    .agent-grid {
        layout: grid;
        grid-size: 3 2;
        grid-gutter: 1;
        padding: 1;
    }
    
    .section-title {
        padding: 1;
        text-style: bold;
        color: #F1F5F9;
        border-bottom: solid #334155;
    }
    
    .patient-section {
        height: 1fr;
        padding: 1;
    }
    
    .patient-table-container {
        height: 1fr;
        border: solid #334155;
    }
    
    .table-header {
        padding: 1;
        background: #1E293B;
        text-style: bold;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Static(self._render_header(), classes="dashboard-header")
        
        # Metrics bar
        with Horizontal(classes="metrics-bar"):
            yield Static(self._render_metric("Total Patients", "0", "[P]"), 
                        id="metric_patients", classes="metric-card")
            yield Static(self._render_metric("Open Cases", "0", "[C]"), 
                        id="metric_open_cases", classes="metric-card")
            yield Static(self._render_metric("Diagnosed Today", "0", "[D]"), 
                        id="metric_diagnosed", classes="metric-card")
            yield Static(self._render_metric("Agents Online", "6/6", "[A]"), 
                        id="metric_agents", classes="metric-card")
        
        with ScrollableContainer():
            # Agent status section
            with Vertical(classes="agent-section"):
                yield Static("AI Specialist Agents", classes="section-title")
                with Container(classes="agent-grid"):
                    for specialty in Specialty:
                        yield Static(
                            self._render_agent_card(specialty),
                            id=f"agent_{specialty.value}",
                            classes="metric-card"
                        )
            
            # Patient tables section
            with Vertical(classes="patient-section"):
                yield Static("Undiagnosed Patients", classes="section-title")
                with Container(classes="patient-table-container"):
                    table = DataTable(id="undiagnosed_table")
                    table.cursor_type = "row"
                    yield table
                
                yield Static("Recently Diagnosed", classes="section-title")
                with Container(classes="patient-table-container"):
                    table = DataTable(id="diagnosed_table")
                    table.cursor_type = "row"
                    yield table
        
        yield Footer()
    
    def _render_header(self) -> Text:
        """Render the dashboard header."""
        text = Text()
        text.append("BioSage", style="bold #3B82F6")
        text.append(" | ", style="#334155")
        text.append("Dashboard", style="bold #F1F5F9")
        return text
    
    def _render_metric(self, label: str, value: str, icon: str) -> Text:
        """Render a metric card."""
        text = Text(justify="center")
        text.append(f"{icon}\n", style="bold #3B82F6")
        text.append(f"{value}\n", style="bold #3B82F6")
        text.append(label, style="#94A3B8")
        return text
    
    def _render_agent_card(self, specialty: Specialty) -> Text:
        """Render an agent status card."""
        names = {
            Specialty.INFECTIOUS: "Infectious Disease",
            Specialty.CARDIOLOGY: "Cardiology",
            Specialty.NEUROLOGY: "Neurology",
            Specialty.ONCOLOGY: "Oncology",
            Specialty.AUTOIMMUNE: "Autoimmune",
            Specialty.TOXICOLOGY: "Toxicology",
        }
        icons = {
            Specialty.INFECTIOUS: "[INF]",
            Specialty.CARDIOLOGY: "[CAR]",
            Specialty.NEUROLOGY: "[NEU]",
            Specialty.ONCOLOGY: "[ONC]",
            Specialty.AUTOIMMUNE: "[AUT]",
            Specialty.TOXICOLOGY: "[TOX]",
        }
        colors = {
            Specialty.INFECTIOUS: "#EF4444",
            Specialty.CARDIOLOGY: "#F97316",
            Specialty.NEUROLOGY: "#A855F7",
            Specialty.ONCOLOGY: "#EC4899",
            Specialty.AUTOIMMUNE: "#EAB308",
            Specialty.TOXICOLOGY: "#22C55E",
        }
        
        text = Text(justify="center")
        icon = icons.get(specialty, "[???]")
        name = names.get(specialty, "Unknown")
        color = colors.get(specialty, "#3B82F6")
        
        text.append(f"{icon}\n", style=f"bold {color}")
        text.append(f"{name}\n", style="bold #F1F5F9")
        text.append("[ON] Online\n", style="#22C55E")
        text.append("[#######---] 70%", style="#94A3B8")
        return text
    
    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self._load_data()
        self._setup_tables()
    
    def _load_data(self) -> None:
        """Load dashboard data from storage."""
        store = get_data_store()
        
        # Get counts
        patient_count = store.patients.count_sync()
        open_cases = store.cases.find_open_cases_sync()
        
        # Update metrics
        self._update_metric("metric_patients", str(patient_count), "[P]", "Total Patients")
        self._update_metric("metric_open_cases", str(len(open_cases)), "[C]", "Open Cases")
    
    def _update_metric(self, widget_id: str, value: str, icon: str, label: str) -> None:
        """Update a metric card."""
        widget = self.query_one(f"#{widget_id}", Static)
        text = Text(justify="center")
        text.append(f"{icon}\n", style="bold #3B82F6")
        text.append(f"{value}\n", style="bold #3B82F6")
        text.append(label, style="#94A3B8")
        widget.update(text)
    
    def _setup_tables(self) -> None:
        """Setup the data tables."""
        store = get_data_store()
        
        # Undiagnosed table
        undiagnosed_table = self.query_one("#undiagnosed_table", DataTable)
        undiagnosed_table.add_columns("Name", "Age", "MRN", "Chief Complaint", "Created")
        
        # Load open cases
        open_cases = store.cases.find_open_cases_sync()
        for case in open_cases[:10]:
            patient = store.patients.load_sync(case.get("patient_id", ""))
            if patient:
                undiagnosed_table.add_row(
                    patient.get("name", "Unknown"),
                    str(self._calculate_age(patient.get("date_of_birth", ""))),
                    patient.get("mrn", ""),
                    case.get("chief_complaint", "")[:30],
                    case.get("created_at", "")[:10],
                )
        
        # Diagnosed table
        diagnosed_table = self.query_one("#diagnosed_table", DataTable)
        diagnosed_table.add_columns("Name", "Age", "MRN", "Diagnosis", "Confidence")
        
        # Load diagnosed cases
        all_cases = store.cases.list_all_sync()
        diagnosed_cases = [c for c in all_cases if c.get("status") == "diagnosed"]
        for case in diagnosed_cases[:10]:
            patient = store.patients.load_sync(case.get("patient_id", ""))
            if patient:
                diagnosed_table.add_row(
                    patient.get("name", "Unknown"),
                    str(self._calculate_age(patient.get("date_of_birth", ""))),
                    patient.get("mrn", ""),
                    (case.get("primary_diagnosis", "") or "")[:25],
                    f"{int((case.get('primary_confidence', 0) or 0) * 100)}%",
                )
    
    def _calculate_age(self, dob_str: str) -> int:
        """Calculate age from date of birth string."""
        if not dob_str:
            return 0
        try:
            from datetime import date
            parts = dob_str.split("-")
            if len(parts) >= 3:
                dob = date(int(parts[0]), int(parts[1]), int(parts[2]))
                today = date.today()
                return today.year - dob.year - (
                    (today.month, today.day) < (dob.month, dob.day)
                )
        except (ValueError, IndexError):
            pass
        return 0
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in tables."""
        table_id = event.data_table.id
        row_index = event.cursor_row
        
        store = get_data_store()
        
        if table_id == "undiagnosed_table":
            # Open case view for undiagnosed patient
            open_cases = store.cases.find_open_cases_sync()
            if row_index < len(open_cases):
                case = open_cases[row_index]
                case_id = case.get("_id", "")
                from biosage_terminal.screens.case_view import CaseViewScreen
                case_screen = CaseViewScreen(case_id=case_id)
                self.app.push_screen(case_screen)
        elif table_id == "diagnosed_table":
            # Open case view for diagnosed patient
            all_cases = store.cases.list_all_sync()
            diagnosed_cases = [c for c in all_cases if c.get("status") == "diagnosed"]
            if row_index < len(diagnosed_cases):
                case = diagnosed_cases[row_index]
                case_id = case.get("_id", "")
                from biosage_terminal.screens.case_view import CaseViewScreen
                case_screen = CaseViewScreen(case_id=case_id)
                self.app.push_screen(case_screen)
    
    def action_goto_welcome(self) -> None:
        """Navigate to welcome screen."""
        self.app.pop_screen()
    
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
    
    def action_refresh(self) -> None:
        """Refresh dashboard data."""
        self._load_data()
        # Refresh tables
        undiagnosed_table = self.query_one("#undiagnosed_table", DataTable)
        undiagnosed_table.clear()
        diagnosed_table = self.query_one("#diagnosed_table", DataTable)
        diagnosed_table.clear()
        self._setup_tables()
    
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
