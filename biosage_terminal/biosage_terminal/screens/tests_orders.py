"""
Tests & Orders screen for BioSage Terminal.
Displays recommended tests and allows ordering.
"""

from datetime import datetime
from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Footer, Input, Button, Checkbox, DataTable
from textual.binding import Binding
from rich.text import Text

from biosage_terminal.storage import get_data_store
from biosage_terminal.theme import PRIORITY_COLORS


class TestsOrdersScreen(Screen):
    """Tests and orders management screen."""
    
    BINDINGS = [
        Binding("escape", "go_back", "Back"),
        Binding("o", "order_selected", "Order Selected"),
        Binding("q", "quit", "Quit"),
    ]
    
    CSS = """
    TestsOrdersScreen {
        background: #0F172A;
    }
    
    .tests-header {
        dock: top;
        height: 3;
        background: #1E293B;
        border-bottom: solid #334155;
        padding: 0 2;
    }
    
    .search-bar {
        height: 5;
        background: #1E293B;
        border-bottom: solid #334155;
        padding: 1;
        layout: horizontal;
    }
    
    .search-input {
        width: 60%;
    }
    
    .action-button {
        margin-left: 2;
    }
    
    .section-title {
        text-style: bold;
        color: #F1F5F9;
        padding: 1;
        border-bottom: solid #334155;
    }
    
    .recommended-section {
        padding: 1;
        border: solid #334155;
        margin: 1;
        height: auto;
        min-height: 5;
        max-height: 15;
        overflow-y: auto;
    }
    
    .test-item {
        padding: 1;
        border-bottom: solid #334155;
    }
    
    .test-item:hover {
        background: #1E293B;
    }
    
    .additional-tests {
        height: auto;
        min-height: 10;
        padding: 1;
    }
    
    .main-scroll {
        height: 1fr;
        overflow-y: auto;
        margin-bottom: 5;
    }
    
    .summary-bar {
        dock: bottom;
        height: 5;
        background: #1E293B;
        border-top: solid #334155;
        padding: 1 2;
        layout: horizontal;
        align: center middle;
    }
    
    .order-button {
        margin-left: 2;
        background: #22C55E;
        min-width: 25;
    }
    """
    
    def __init__(self, case_id: str = "", patient_id: str = "", diagnosis_data: dict = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.case_id = case_id
        self.patient_id = patient_id
        self.diagnosis_data = diagnosis_data or {}
        self.case_data = {}
        self.selected_tests = set()
        self.selected_additional = set()  # For additional tests from table
        self.additional_tests_list = [
            {"test_name": "Complete Blood Count (CBC)", "category": "Hematology", "estimated_cost": 45, "turnaround_time": "4-6 hours"},
            {"test_name": "Comprehensive Metabolic Panel", "category": "Chemistry", "estimated_cost": 55, "turnaround_time": "4-6 hours"},
            {"test_name": "Lipid Panel", "category": "Chemistry", "estimated_cost": 40, "turnaround_time": "6-8 hours"},
            {"test_name": "Thyroid Panel (TSH, T3, T4)", "category": "Endocrine", "estimated_cost": 85, "turnaround_time": "12-24 hours"},
            {"test_name": "Urinalysis", "category": "Microbiology", "estimated_cost": 25, "turnaround_time": "2-4 hours"},
            {"test_name": "C-Reactive Protein (CRP)", "category": "Inflammation", "estimated_cost": 35, "turnaround_time": "4-6 hours"},
            {"test_name": "Erythrocyte Sedimentation Rate", "category": "Inflammation", "estimated_cost": 20, "turnaround_time": "2-4 hours"},
            {"test_name": "Antinuclear Antibody (ANA)", "category": "Immunology", "estimated_cost": 95, "turnaround_time": "24-48 hours"},
            {"test_name": "Rheumatoid Factor", "category": "Immunology", "estimated_cost": 65, "turnaround_time": "24 hours"},
            {"test_name": "Blood Culture", "category": "Microbiology", "estimated_cost": 120, "turnaround_time": "24-72 hours"},
            {"test_name": "Procalcitonin", "category": "Infection", "estimated_cost": 85, "turnaround_time": "4-6 hours"},
            {"test_name": "D-Dimer", "category": "Coagulation", "estimated_cost": 55, "turnaround_time": "2-4 hours"},
            {"test_name": "Troponin I/T", "category": "Cardiac", "estimated_cost": 75, "turnaround_time": "1-2 hours"},
            {"test_name": "BNP/NT-proBNP", "category": "Cardiac", "estimated_cost": 90, "turnaround_time": "2-4 hours"},
            {"test_name": "HbA1c", "category": "Diabetes", "estimated_cost": 50, "turnaround_time": "24 hours"},
        ]
        # Load case data immediately so it's available during compose()
        self._load_case_data()
    
    def compose(self) -> ComposeResult:
        yield Static(self._render_header(), classes="tests-header")
        
        with Container(classes="search-bar"):
            yield Input(
                placeholder="Search tests...",
                id="search_input",
                classes="search-input",
            )
            yield Button("+ Custom Order", id="btn_custom", classes="action-button")
        
        with ScrollableContainer(classes="main-scroll"):
            # Show previously ordered tests if any
            yield Static("ORDERED TESTS", classes="section-title", id="ordered_title")
            with ScrollableContainer(classes="recommended-section", id="ordered_scroll"):
                yield Static(self._render_ordered_tests(), id="ordered_tests")
            
            yield Static("RECOMMENDED TESTS", classes="section-title")
            with Container(classes="additional-tests", id="recommended_container"):
                table = DataTable(id="recommended_table")
                table.cursor_type = "row"
                yield table
            
            yield Static("ADDITIONAL TESTS", classes="section-title")
            with Container(classes="additional-tests"):
                table = DataTable(id="additional_table")
                table.cursor_type = "row"
                yield table
        
        with Horizontal(classes="summary-bar"):
            yield Static(self._render_summary(), id="summary")
            yield Button("Order Selected Tests", id="btn_order", classes="order-button")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Setup table and refresh data when screen is mounted."""
        # Reload data to ensure it's fresh
        self._load_case_data()
        self._setup_recommended_table()
        self._setup_additional_table()
        
        # Debug: show what case_id and data we have
        test_count = len(self.case_data.get("test_recommendations", []))
        self.notify(f"Case: {self.case_id}, Tests: {test_count}", severity="warning", timeout=10)
        
        # Directly update the widgets now that data is loaded
        try:
            ordered_widget = self.query_one("#ordered_tests", Static)
            ordered_widget.update(self._render_ordered_tests())
            ordered_widget.refresh()
            
            summary_widget = self.query_one("#summary", Static)
            summary_widget.update(self._render_summary())
            summary_widget.refresh()
            
            # Force screen refresh
            self.refresh()
        except Exception as e:
            self.notify(f"Update error: {e}", severity="error")
    
    def _load_case_data(self) -> None:
        """Load case data from storage."""
        self.case_data = {}  # Reset first
        
        if self.case_id:
            try:
                store = get_data_store()
                loaded = store.cases.load_sync(self.case_id)
                if loaded:
                    self.case_data = loaded
                else:
                    # Try with different formats
                    alt_id = self.case_id.replace("CASE-", "C-") if "CASE-" in self.case_id else self.case_id
                    loaded = store.cases.load_sync(alt_id)
                    if loaded:
                        self.case_data = loaded
                        self.case_id = alt_id  # Update to correct format
            except Exception as e:
                # Fallback - try to use diagnosis_data if available
                pass
        
        # If case_data is still empty and we have diagnosis_data, try to extract from it
        if not self.case_data and self.diagnosis_data:
            # The diagnosis_data might have recommendations embedded
            self.case_data = self.diagnosis_data
    
    def _render_header(self) -> Text:
        """Render the header."""
        text = Text()
        text.append("BioSage", style="bold #3B82F6")
        text.append(" | ", style="#334155")
        text.append("Diagnostic Tests & Orders", style="bold #F1F5F9")
        return text
    
    def _render_ordered_tests(self) -> Text:
        """Render the already ordered tests section."""
        text = Text()
        
        ordered = self.case_data.get("ordered_tests", [])
        
        if not ordered:
            text.append("No tests ordered yet.", style="#64748B")
            return text
        
        for test in ordered:
            name = test.get("test_name", "Unknown Test")
            status = test.get("status", "ordered")
            ordered_at = test.get("ordered_at", "")
            cost = test.get("estimated_cost", 0)
            source = test.get("source", "recommended")
            
            # Status icons
            status_icon = "⏳" if status == "ordered" else ("✅" if status == "completed" else "🔬")
            status_color = "#EAB308" if status == "ordered" else ("#22C55E" if status == "completed" else "#3B82F6")
            
            text.append(f"{status_icon} ", style=status_color)
            text.append(f"{name}", style="bold #F1F5F9")
            text.append(f" [${cost}]", style="#22C55E")
            text.append(f" ({source})", style="#64748B")
            
            if ordered_at:
                # Format the datetime nicely
                try:
                    dt = datetime.fromisoformat(ordered_at)
                    text.append(f" - {dt.strftime('%Y-%m-%d %H:%M')}", style="#94A3B8")
                except ValueError:
                    pass
            
            text.append("\n", style="")
        
        return text
    
    def _setup_recommended_table(self) -> None:
        """Setup the recommended tests table."""
        try:
            table = self.query_one("#recommended_table", DataTable)
            table.clear()
            table.add_columns("", "Test Name", "Priority", "Cost", "Turnaround")
            
            tests = self.case_data.get("test_recommendations", [])
            
            if not tests:
                # Add a placeholder row if no tests
                table.add_row("  ", "No tests recommended yet", "-", "-", "-")
                return
            
            for i, test in enumerate(tests):
                selected = i in self.selected_tests
                checkbox = "[X]" if selected else "[ ]"
                name = test.get("test_name", "Unknown Test")
                priority = test.get("priority", "medium").upper()
                cost = f"${test.get('estimated_cost', 50)}"
                turnaround = test.get("turnaround_time", "4-6 hours")
                table.add_row(checkbox, name, priority, cost, turnaround)
        except Exception as e:
            pass
    
    def _refresh_recommended_table(self) -> None:
        """Refresh the recommended tests table to show selection state."""
        try:
            table = self.query_one("#recommended_table", DataTable)
            table.clear()
            
            tests = self.case_data.get("test_recommendations", [])
            
            if not tests:
                table.add_row("  ", "No tests recommended yet", "-", "-", "-")
                return
            
            for i, test in enumerate(tests):
                selected = i in self.selected_tests
                checkbox = "[X]" if selected else "[ ]"
                name = test.get("test_name", "Unknown Test")
                priority = test.get("priority", "medium").upper()
                cost = f"${test.get('estimated_cost', 50)}"
                turnaround = test.get("turnaround_time", "4-6 hours")
                table.add_row(checkbox, name, priority, cost, turnaround)
        except Exception:
            pass
    
    def _setup_additional_table(self) -> None:
        """Setup the additional tests table."""
        table = self.query_one("#additional_table", DataTable)
        table.add_columns("", "Test Name", "Category", "Cost", "Turnaround")
        
        for i, test in enumerate(self.additional_tests_list):
            selected = i in self.selected_additional
            checkbox = "[X]" if selected else "[ ]"
            table.add_row(
                checkbox,
                test["test_name"],
                test["category"],
                f"${test['estimated_cost']}",
                test["turnaround_time"]
            )
    
    def _render_summary(self) -> Text:
        """Render the order summary."""
        text = Text()
        
        num_recommended = len(self.selected_tests)
        num_additional = len(self.selected_additional)
        num_selected = num_recommended + num_additional
        
        # Calculate estimated cost from recommended tests
        recommended_tests = self.case_data.get("test_recommendations", [])
        total_cost = sum(
            recommended_tests[i].get("estimated_cost", 50) 
            for i in self.selected_tests 
            if i < len(recommended_tests)
        )
        
        # Add cost from additional tests
        total_cost += sum(
            self.additional_tests_list[i].get("estimated_cost", 50)
            for i in self.selected_additional
            if i < len(self.additional_tests_list)
        )
        
        # Show already ordered tests count
        ordered_count = len(self.case_data.get("ordered_tests", []))
        
        text.append(f"Selected: {num_selected} test(s)", style="#F1F5F9")
        text.append(" | ", style="#334155")
        text.append(f"Estimated Cost: ${total_cost}", style="#22C55E")
        if ordered_count > 0:
            text.append(" | ", style="#334155")
            text.append(f"Previously Ordered: {ordered_count}", style="#94A3B8")
        
        return text
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn_order":
            self._place_order()
        elif button_id == "btn_custom":
            self._add_custom_order()
    
    def _place_order(self) -> None:
        """Place order for selected tests and save to storage."""
        if not self.selected_tests and not self.selected_additional:
            self.notify("No tests selected", severity="warning")
            return
        
        store = get_data_store()
        recommended_tests = self.case_data.get("test_recommendations", [])
        
        # Collect all ordered tests
        ordered_tests = []
        
        # Add selected recommended tests
        for i in self.selected_tests:
            if i < len(recommended_tests):
                test = recommended_tests[i].copy()
                test["status"] = "ordered"
                test["ordered_at"] = datetime.utcnow().isoformat()
                test["source"] = "recommended"
                ordered_tests.append(test)
        
        # Add selected additional tests
        for i in self.selected_additional:
            if i < len(self.additional_tests_list):
                test = self.additional_tests_list[i].copy()
                test["status"] = "ordered"
                test["ordered_at"] = datetime.utcnow().isoformat()
                test["source"] = "additional"
                test["priority"] = "medium"
                test["rationale"] = "Manually ordered by clinician"
                ordered_tests.append(test)
        
        if not ordered_tests:
            self.notify("No valid tests selected", severity="warning")
            return
        
        # Update case data with ordered tests
        existing_ordered = self.case_data.get("ordered_tests", [])
        existing_ordered.extend(ordered_tests)
        self.case_data["ordered_tests"] = existing_ordered
        self.case_data["updated_at"] = datetime.utcnow().isoformat()
        
        # Save case data to storage
        store.cases.save_sync(self.case_id, self.case_data)
        
        # Log the order to audit
        test_names = [t.get("test_name", "") for t in ordered_tests]
        total_cost = sum(t.get("estimated_cost", 0) for t in ordered_tests)
        
        store.audit.log_event(
            event_type="tests_ordered",
            user="system",
            action=f"Ordered {len(ordered_tests)} tests for case {self.case_id}",
            details={
                "tests": test_names,
                "case_id": self.case_id,
                "patient_id": self.patient_id,
                "total_cost": total_cost,
                "ordered_count": len(ordered_tests),
            },
        )
        
        self.notify(f"Ordered {len(ordered_tests)} tests successfully! (${total_cost} estimated)")
        self.selected_tests.clear()
        self.selected_additional.clear()
        self._load_case_data()  # Reload to get updated data
        self._update_displays()
    
    def _add_custom_order(self) -> None:
        """Add a custom test order."""
        self.notify("Custom order dialog would open here")
    
    def _update_displays(self) -> None:
        """Update all display widgets."""
        try:
            self.query_one("#ordered_tests", Static).update(self._render_ordered_tests())
            self.query_one("#summary", Static).update(self._render_summary())
        except Exception:
            pass
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in recommended and additional tests tables."""
        row_index = event.cursor_row
        table_id = event.data_table.id
        
        if table_id == "recommended_table":
            # Toggle selection for recommended tests
            if row_index in self.selected_tests:
                self.selected_tests.discard(row_index)
            else:
                self.selected_tests.add(row_index)
            self._refresh_recommended_table()
        elif table_id == "additional_table":
            # Toggle selection for additional tests
            if row_index in self.selected_additional:
                self.selected_additional.discard(row_index)
            else:
                self.selected_additional.add(row_index)
            self._refresh_additional_table()
        
        self._update_displays()
    
    def _refresh_additional_table(self) -> None:
        """Refresh the additional tests table to show selection state."""
        try:
            table = self.query_one("#additional_table", DataTable)
            table.clear()
            
            for i, test in enumerate(self.additional_tests_list):
                selected = i in self.selected_additional
                checkbox = "[X]" if selected else "[ ]"
                table.add_row(
                    checkbox,
                    test["test_name"],
                    test["category"],
                    f"${test['estimated_cost']}",
                    test["turnaround_time"]
                )
        except Exception:
            pass
    
    def action_order_selected(self) -> None:
        """Order the selected tests."""
        self._place_order()
    
    def action_go_back(self) -> None:
        """Go back to previous screen."""
        self.app.pop_screen()
    
    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()
