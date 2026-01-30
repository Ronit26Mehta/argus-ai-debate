"""
Cases Manager Screen for BioSage Terminal.
View all cases from the database, access them, diagnose, and manage.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import (
    Header, Footer, Static, Button, Input, Label,
    DataTable, TabbedContent, TabPane, Select,
)
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.binding import Binding
from rich.text import Text
from rich.panel import Panel
from rich.table import Table

from biosage_terminal.storage import get_data_store
from biosage_terminal.theme import COLORS


class CaseCard(Static):
    """A card displaying case summary."""
    
    def __init__(
        self,
        case_id: str,
        patient_name: str,
        chief_complaint: str,
        status: str,
        created_at: str,
        priority: str = "medium",
        diagnosis: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.case_id = case_id
        self.patient_name = patient_name
        self.chief_complaint = chief_complaint
        self.status = status
        self.created_at = created_at
        self.priority = priority
        self.diagnosis = diagnosis
        
    def compose(self) -> ComposeResult:
        status_colors = {
            "open": COLORS["warning"],
            "in_progress": COLORS["info"],
            "diagnosed": COLORS["success"],
            "closed": COLORS["text_muted"],
        }
        priority_colors = {
            "high": COLORS["error"],
            "medium": COLORS["warning"],
            "low": COLORS["success"],
        }
        
        status_color = status_colors.get(self.status, COLORS["text_muted"])
        priority_color = priority_colors.get(self.priority, COLORS["text_muted"])
        
        # Format date
        try:
            dt = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d %H:%M")
        except:
            date_str = self.created_at[:16] if len(self.created_at) > 16 else self.created_at
        
        diagnosis_line = ""
        if self.diagnosis:
            diagnosis_line = f"\n[{COLORS['success']}]Dx: {self.diagnosis}[/{COLORS['success']}]"
        
        content = f"""[bold]{self.case_id}[/bold] [{status_color}][{self.status.upper().replace('_', ' ')}][/{status_color}] [{priority_color}][{self.priority.upper()}][/{priority_color}]
[bold]{self.patient_name}[/bold]
[dim]{self.chief_complaint[:60]}{'...' if len(self.chief_complaint) > 60 else ''}[/dim]
[dim]Created: {date_str}[/dim]{diagnosis_line}"""
        
        yield Static(content, classes="case-card-content")


class CasesManagerScreen(Screen):
    """Cases Manager - View, filter, and manage all cases."""
    
    BINDINGS = [
        Binding("escape", "go_back", "Back"),
        Binding("d", "goto_dashboard", "Dashboard"),
        Binding("n", "new_case", "New Case"),
        Binding("enter", "open_case", "Open"),
        Binding("r", "run_diagnosis", "Diagnose"),
        Binding("t", "order_tests", "Tests"),
        Binding("f", "focus_filter", "Filter"),
        Binding("1", "filter_all", "All"),
        Binding("2", "filter_open", "Open"),
        Binding("3", "filter_diagnosed", "Diagnosed"),
        Binding("q", "quit", "Quit"),
    ]
    
    CSS = """
    CasesManagerScreen {
        background: #0F172A;
        layout: vertical;
    }
    
    .cases-header {
        height: 3;
        background: #1E40AF;
        color: #FFFFFF;
        padding: 1 2;
        text-align: center;
        text-style: bold;
        border-bottom: solid #3B82F6;
    }
    
    .toolbar {
        height: 3;
        padding: 0 1;
        background: #1E293B;
        border-bottom: solid #334155;
        layout: horizontal;
    }
    
    .stats-bar {
        height: 3;
        layout: horizontal;
        padding: 0 1;
        background: #1E293B;
        border-bottom: solid #334155;
    }
    
    .stat-box {
        width: 1fr;
        padding: 0 2;
        text-align: center;
        border-right: solid #334155;
    }
    
    .stat-box:last-of-type {
        border-right: none;
    }
    
    .main-content {
        height: 1fr;
        layout: horizontal;
        padding: 1;
    }
    
    .cases-list-container {
        width: 2fr;
        height: 100%;
        margin-right: 1;
    }
    
    .cases-table-container {
        height: 100%;
        border: solid #334155;
        background: #1E293B;
    }
    
    .case-card-content {
        padding: 1;
        margin: 1;
        background: #1E293B;
        border: solid #334155;
    }
    
    .case-card-content:hover {
        border: solid #3B82F6;
    }
    
    .detail-panel {
        width: 1fr;
        height: 100%;
        padding: 1;
        background: #1E293B;
        border: solid #334155;
        layout: vertical;
    }
    
    .detail-content-scroll {
        height: 1fr;
        padding: 1;
    }
    
    .action-buttons {
        height: auto;
        padding: 1;
        layout: vertical;
    }
    
    .action-btn {
        width: 100%;
        margin-bottom: 1;
    }
    
    .filter-input {
        width: 40;
        margin-right: 2;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.store = get_data_store()
        self.all_cases: list[dict] = []
        self.filtered_cases: list[dict] = []
        self.current_filter = "all"
        self.selected_case_id: Optional[str] = None
        self.search_query = ""
        
    def compose(self) -> ComposeResult:
        yield Static(self._render_header(), classes="cases-header")
        
        # Toolbar with filter and search
        with Horizontal(classes="toolbar"):
            yield Input(placeholder="Search cases (patient, complaint, ID)...", 
                       id="search_input", classes="filter-input")
            yield Button("All [1]", id="btn_all", variant="primary")
            yield Button("Open [2]", id="btn_open", variant="default")
            yield Button("Diagnosed [3]", id="btn_diagnosed", variant="default")
            yield Button("+ New Case [n]", id="btn_new", variant="success")
        
        # Stats bar
        with Horizontal(classes="stats-bar"):
            yield Static(self._render_stat("Total Cases", "0"), id="stat_total", classes="stat-box")
            yield Static(self._render_stat("Open", "0"), id="stat_open", classes="stat-box")
            yield Static(self._render_stat("In Progress", "0"), id="stat_progress", classes="stat-box")
            yield Static(self._render_stat("Diagnosed", "0"), id="stat_diagnosed", classes="stat-box")
        
        # Main content area
        with Horizontal(classes="main-content"):
            # Cases list/table
            with Container(classes="cases-list-container"):
                yield DataTable(id="cases_table", classes="cases-table-container")
            
            # Detail panel
            with Container(classes="detail-panel"):
                with ScrollableContainer(classes="detail-content-scroll"):
                    yield Static(self._render_no_selection(), id="detail_content")
                
                with Container(classes="action-buttons"):
                    yield Button("Open Case [Enter]", id="btn_open_case", 
                               variant="primary", classes="action-btn")
                    yield Button("Run Diagnosis [r]", id="btn_diagnose", 
                               variant="success", classes="action-btn")
                    yield Button("Order Tests [t]", id="btn_tests", 
                               variant="default", classes="action-btn")
                    yield Button("View Patient", id="btn_patient", 
                               variant="default", classes="action-btn")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize the cases table when screen is mounted."""
        self._setup_table()
        self._load_cases()
        self._update_stats()
    
    def _setup_table(self) -> None:
        """Setup the DataTable columns."""
        table = self.query_one("#cases_table", DataTable)
        table.cursor_type = "row"
        table.add_columns(
            "Case ID", "Patient", "Chief Complaint", "Status", "Priority", "Created", "Diagnosis"
        )
    
    def _load_cases(self) -> None:
        """Load all cases from storage."""
        self.all_cases = self.store.cases.list_all_sync()
        
        # Enrich with patient data
        for case in self.all_cases:
            patient_id = case.get("patient_id", "")
            if patient_id:
                patient = self.store.patients.load_sync(patient_id)
                if patient:
                    case["patient_name"] = patient.get("name", "Unknown")
                    case["patient_mrn"] = patient.get("mrn", "N/A")
                else:
                    case["patient_name"] = "Unknown"
                    case["patient_mrn"] = "N/A"
            else:
                case["patient_name"] = "Unknown"
                case["patient_mrn"] = "N/A"
        
        # Sort by creation date (newest first)
        self.all_cases.sort(
            key=lambda c: c.get("created_at", c.get("_updated_at", "")), 
            reverse=True
        )
        
        self._apply_filter()
    
    def _apply_filter(self) -> None:
        """Apply current filter to cases."""
        # Start with all cases
        self.filtered_cases = list(self.all_cases)
        
        # Apply status filter
        if self.current_filter == "open":
            self.filtered_cases = [
                c for c in self.filtered_cases 
                if c.get("status") in ["open", "in_progress"]
            ]
        elif self.current_filter == "diagnosed":
            self.filtered_cases = [
                c for c in self.filtered_cases 
                if c.get("status") in ["diagnosed", "closed"]
            ]
        
        # Apply search query
        if self.search_query:
            query = self.search_query.lower()
            self.filtered_cases = [
                c for c in self.filtered_cases
                if query in c.get("patient_name", "").lower()
                or query in c.get("chief_complaint", "").lower()
                or query in c.get("_id", "").lower()
                or query in c.get("case_number", "").lower()
            ]
        
        self._populate_table()
    
    def _populate_table(self) -> None:
        """Populate the table with filtered cases."""
        table = self.query_one("#cases_table", DataTable)
        table.clear()
        
        if not self.filtered_cases:
            # Show no cases message
            table.add_row("--", "No cases found", "--", "--", "--", "--", "--")
            return
        
        for case in self.filtered_cases:
            case_id = case.get("_id", case.get("case_number", "N/A"))
            patient_name = case.get("patient_name", "Unknown")
            complaint = case.get("chief_complaint", "Not specified")
            if len(complaint) > 40:
                complaint = complaint[:37] + "..."
            
            status = case.get("status", "open")
            priority = case.get("priority", "medium")
            
            # Format date
            created = case.get("created_at", case.get("_updated_at", ""))
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                date_str = dt.strftime("%Y-%m-%d")
            except:
                date_str = created[:10] if len(created) > 10 else created
            
            # Get diagnosis if available
            diagnosis = ""
            diagnosis_result = case.get("diagnosis_result", {})
            if diagnosis_result:
                differentials = diagnosis_result.get("differential_diagnoses", [])
                if differentials:
                    diagnosis = differentials[0].get("diagnosis", "")
                    if len(diagnosis) > 25:
                        diagnosis = diagnosis[:22] + "..."
            
            # Color-code status
            status_text = Text()
            status_colors = {
                "open": "#EAB308",
                "in_progress": "#06B6D4",
                "diagnosed": "#22C55E",
                "closed": "#64748B",
            }
            status_text.append(status.upper().replace("_", " "), style=status_colors.get(status, "#64748B"))
            
            # Color-code priority
            priority_text = Text()
            priority_colors = {
                "high": "#EF4444",
                "medium": "#EAB308",
                "low": "#22C55E",
            }
            priority_text.append(priority.upper(), style=priority_colors.get(priority, "#64748B"))
            
            table.add_row(
                case_id,
                patient_name,
                complaint,
                status_text,
                priority_text,
                date_str,
                diagnosis or "--"
            )
    
    def _update_stats(self) -> None:
        """Update the stats bar."""
        total = len(self.all_cases)
        open_count = len([c for c in self.all_cases if c.get("status") == "open"])
        progress_count = len([c for c in self.all_cases if c.get("status") == "in_progress"])
        diagnosed_count = len([c for c in self.all_cases if c.get("status") in ["diagnosed", "closed"]])
        
        try:
            self.query_one("#stat_total", Static).update(
                self._render_stat("Total Cases", str(total))
            )
            self.query_one("#stat_open", Static).update(
                self._render_stat("Open", str(open_count), "#EAB308")
            )
            self.query_one("#stat_progress", Static).update(
                self._render_stat("In Progress", str(progress_count), "#06B6D4")
            )
            self.query_one("#stat_diagnosed", Static).update(
                self._render_stat("Diagnosed", str(diagnosed_count), "#22C55E")
            )
        except Exception:
            pass
    
    def _render_header(self) -> Text:
        """Render the header."""
        text = Text()
        text.append("\n", style="")
        text.append("🏥 BioSage", style="bold #FFFFFF")
        text.append("  │  ", style="bold #60A5FA")
        text.append("Cases Manager", style="bold #60A5FA")
        text.append("  │  ", style="bold #60A5FA")
        text.append("View, diagnose, and manage all cases", style="#93C5FD")
        text.append("\n", style="")
        return text
    
    def _render_stat(self, label: str, value: str, color: str = "#F1F5F9") -> Text:
        """Render a stat box."""
        text = Text()
        text.append(f"{value}\n", style=f"bold {color}")
        text.append(label, style="#94A3B8")
        return text
    
    def _render_no_selection(self) -> Text:
        """Render no selection message."""
        text = Text()
        text.append("No Case Selected\n\n", style="bold #64748B")
        text.append("Select a case from the table\n", style="#64748B")
        text.append("to view details and actions.\n\n", style="#64748B")
        text.append("Keyboard shortcuts:\n", style="#94A3B8")
        text.append("[Enter] Open Case\n", style="#64748B")
        text.append("[r] Run Diagnosis\n", style="#64748B")
        text.append("[t] Order Tests\n", style="#64748B")
        text.append("[n] New Case\n", style="#64748B")
        return text
    
    def _render_case_detail(self, case: dict) -> Text:
        """Render case detail panel."""
        text = Text()
        
        case_id = case.get("_id", case.get("case_number", "N/A"))
        patient_name = case.get("patient_name", "Unknown")
        patient_mrn = case.get("patient_mrn", "N/A")
        complaint = case.get("chief_complaint", "Not specified")
        status = case.get("status", "open")
        priority = case.get("priority", "medium")
        
        # Status color
        status_colors = {
            "open": "#EAB308",
            "in_progress": "#06B6D4",
            "diagnosed": "#22C55E",
            "closed": "#64748B",
        }
        
        text.append(f"{case_id}\n", style="bold #3B82F6")
        text.append(f"Status: ", style="#94A3B8")
        text.append(f"{status.upper().replace('_', ' ')}\n", style=status_colors.get(status, "#64748B"))
        text.append(f"Priority: ", style="#94A3B8")
        text.append(f"{priority.upper()}\n\n", style="#F1F5F9")
        
        text.append("Patient\n", style="bold #8B5CF6")
        text.append(f"{patient_name}\n", style="bold #F1F5F9")
        text.append(f"MRN: {patient_mrn}\n\n", style="#94A3B8")
        
        text.append("Chief Complaint\n", style="bold #8B5CF6")
        text.append(f"{complaint}\n\n", style="#F1F5F9")
        
        # Show diagnosis if available
        diagnosis_result = case.get("diagnosis_result", {})
        if diagnosis_result:
            text.append("Diagnosis Result\n", style="bold #22C55E")
            differentials = diagnosis_result.get("differential_diagnoses", [])
            for i, dx in enumerate(differentials[:3]):
                diagnosis = dx.get("diagnosis", "Unknown")
                confidence = dx.get("confidence", 0)
                text.append(f"{i+1}. {diagnosis}\n", style="#F1F5F9")
                text.append(f"   Confidence: {confidence:.0%}\n", style="#94A3B8")
        
        # Show symptoms
        symptoms = case.get("symptoms", [])
        if symptoms:
            text.append("\nSymptoms\n", style="bold #8B5CF6")
            for symptom in symptoms[:5]:
                if isinstance(symptom, dict):
                    text.append(f"- {symptom.get('name', symptom)}\n", style="#F1F5F9")
                else:
                    text.append(f"- {symptom}\n", style="#F1F5F9")
        
        return text
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in the table."""
        if not self.filtered_cases or event.row_key is None:
            return
        
        try:
            row_index = event.cursor_row
            if 0 <= row_index < len(self.filtered_cases):
                case = self.filtered_cases[row_index]
                self.selected_case_id = case.get("_id")
                
                # Update detail panel
                detail = self.query_one("#detail_content", Static)
                detail.update(self._render_case_detail(case))
        except Exception:
            pass
    
    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Handle row highlight in the table."""
        if not self.filtered_cases or event.row_key is None:
            return
        
        try:
            row_index = event.cursor_row
            if 0 <= row_index < len(self.filtered_cases):
                case = self.filtered_cases[row_index]
                self.selected_case_id = case.get("_id")
                
                # Update detail panel
                detail = self.query_one("#detail_content", Static)
                detail.update(self._render_case_detail(case))
        except Exception:
            pass
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes."""
        if event.input.id == "search_input":
            self.search_query = event.value
            self._apply_filter()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn_all":
            self.action_filter_all()
        elif button_id == "btn_open":
            self.action_filter_open()
        elif button_id == "btn_diagnosed":
            self.action_filter_diagnosed()
        elif button_id == "btn_new":
            self.action_new_case()
        elif button_id == "btn_open_case":
            self.action_open_case()
        elif button_id == "btn_diagnose":
            self.action_run_diagnosis()
        elif button_id == "btn_tests":
            self.action_order_tests()
        elif button_id == "btn_patient":
            self._view_patient()
    
    def _update_filter_buttons(self) -> None:
        """Update filter button variants."""
        try:
            btn_all = self.query_one("#btn_all", Button)
            btn_open = self.query_one("#btn_open", Button)
            btn_diagnosed = self.query_one("#btn_diagnosed", Button)
            
            btn_all.variant = "primary" if self.current_filter == "all" else "default"
            btn_open.variant = "primary" if self.current_filter == "open" else "default"
            btn_diagnosed.variant = "primary" if self.current_filter == "diagnosed" else "default"
        except Exception:
            pass
    
    def action_filter_all(self) -> None:
        """Show all cases."""
        self.current_filter = "all"
        self._update_filter_buttons()
        self._apply_filter()
    
    def action_filter_open(self) -> None:
        """Show only open cases."""
        self.current_filter = "open"
        self._update_filter_buttons()
        self._apply_filter()
    
    def action_filter_diagnosed(self) -> None:
        """Show only diagnosed cases."""
        self.current_filter = "diagnosed"
        self._update_filter_buttons()
        self._apply_filter()
    
    def action_focus_filter(self) -> None:
        """Focus the search input."""
        try:
            search_input = self.query_one("#search_input", Input)
            search_input.focus()
        except Exception:
            pass
    
    def action_new_case(self) -> None:
        """Navigate to create new case."""
        self.app.goto_onboarding()
    
    def action_go_back(self) -> None:
        """Go back to previous screen."""
        self.app.pop_screen()
    
    def action_goto_dashboard(self) -> None:
        """Navigate to dashboard."""
        self.app.goto_dashboard()
    
    def action_open_case(self) -> None:
        """Open the selected case."""
        if self.selected_case_id:
            case = self._get_selected_case()
            if case:
                patient_id = case.get("patient_id", "")
                self.app.goto_case(self.selected_case_id, patient_id)
        else:
            self.notify("Please select a case first", severity="warning")
    
    def action_run_diagnosis(self) -> None:
        """Run diagnosis on selected case."""
        if self.selected_case_id:
            case = self._get_selected_case()
            if case:
                patient_id = case.get("patient_id", "")
                self.app.goto_diagnosis(self.selected_case_id, patient_id)
        else:
            self.notify("Please select a case first", severity="warning")
    
    def action_order_tests(self) -> None:
        """Order tests for selected case."""
        if self.selected_case_id:
            self.app.goto_tests_orders(self.selected_case_id)
        else:
            self.notify("Please select a case first", severity="warning")
    
    def _view_patient(self) -> None:
        """View patient summary for selected case."""
        if self.selected_case_id:
            case = self._get_selected_case()
            if case:
                patient_id = case.get("patient_id", "")
                if patient_id:
                    self.app.goto_patient_summary(patient_id)
                else:
                    self.notify("No patient linked to this case", severity="warning")
        else:
            self.notify("Please select a case first", severity="warning")
    
    def _get_selected_case(self) -> Optional[dict]:
        """Get the currently selected case data."""
        if not self.selected_case_id:
            return None
        
        for case in self.filtered_cases:
            if case.get("_id") == self.selected_case_id:
                return case
        return None
