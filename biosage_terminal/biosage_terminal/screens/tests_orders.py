"""
Tests & Orders screen for BioSage Terminal.
Displays recommended tests and allows ordering.
"""

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
    }
    
    .test-item {
        padding: 1;
        border-bottom: solid #334155;
    }
    
    .test-item:hover {
        background: #1E293B;
    }
    
    .additional-tests {
        height: 1fr;
        padding: 1;
    }
    
    .summary-bar {
        dock: bottom;
        height: 4;
        background: #1E293B;
        border-top: solid #334155;
        padding: 1;
        layout: horizontal;
        align: center middle;
    }
    
    .order-button {
        dock: right;
        background: #22C55E;
    }
    """
    
    def __init__(self, case_id: str = "", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.case_id = case_id
        self.case_data = {}
        self.selected_tests = set()
    
    def compose(self) -> ComposeResult:
        yield Static(self._render_header(), classes="tests-header")
        
        with Container(classes="search-bar"):
            yield Input(
                placeholder="Search tests...",
                id="search_input",
                classes="search-input",
            )
            yield Button("+ Custom Order", id="btn_custom", classes="action-button")
        
        with ScrollableContainer():
            yield Static("RECOMMENDED TESTS", classes="section-title")
            with Container(classes="recommended-section"):
                yield Static(self._render_recommended_tests(), id="recommended_tests")
            
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
        """Load data when screen is mounted."""
        self._load_case_data()
        self._setup_additional_table()
    
    def _load_case_data(self) -> None:
        """Load case data from storage."""
        if self.case_id:
            store = get_data_store()
            self.case_data = store.cases.load_sync(self.case_id) or {}
    
    def _render_header(self) -> Text:
        """Render the header."""
        text = Text()
        text.append("BioSage", style="bold #3B82F6")
        text.append(" | ", style="#334155")
        text.append("Diagnostic Tests & Orders", style="bold #F1F5F9")
        return text
    
    def _render_recommended_tests(self) -> Text:
        """Render the recommended tests section."""
        text = Text()
        
        tests = self.case_data.get("test_recommendations", [])
        
        if not tests:
            text.append("No tests recommended yet.\n", style="#64748B")
            text.append("Run diagnosis first to get recommendations.", style="#94A3B8")
            return text
        
        for i, test in enumerate(tests):
            name = test.get("test_name", "Unknown Test")
            rationale = test.get("rationale", "")
            priority = test.get("priority", "medium")
            cost = test.get("estimated_cost", 50)
            turnaround = test.get("turnaround_time", "4-6 hours")
            
            priority_color = PRIORITY_COLORS.get(priority.lower(), "#94A3B8")
            
            # Checkbox simulation
            selected = i in self.selected_tests
            checkbox = "[X]" if selected else "[ ]"
            checkbox_color = "#22C55E" if selected else "#64748B"
            
            text.append(f"{checkbox} ", style=f"bold {checkbox_color}")
            text.append(f"{name}", style="bold #F1F5F9")
            text.append(f" [{priority.upper()}] ", style=f"bold {priority_color}")
            text.append(f"${cost}\n", style="#22C55E")
            
            if rationale:
                text.append(f"    {rationale}\n", style="#94A3B8")
            
            if turnaround:
                text.append(f"    Turnaround: {turnaround}\n", style="#64748B")
            
            text.append("\n", style="")
        
        return text
    
    def _setup_additional_table(self) -> None:
        """Setup the additional tests table."""
        table = self.query_one("#additional_table", DataTable)
        table.add_columns("Test Name", "Category", "Cost", "Turnaround")
        
        # Common diagnostic tests
        additional_tests = [
            ("Complete Blood Count (CBC)", "Hematology", 45, "4-6 hours"),
            ("Comprehensive Metabolic Panel", "Chemistry", 55, "4-6 hours"),
            ("Lipid Panel", "Chemistry", 40, "6-8 hours"),
            ("Thyroid Panel (TSH, T3, T4)", "Endocrine", 85, "12-24 hours"),
            ("Urinalysis", "Microbiology", 25, "2-4 hours"),
            ("C-Reactive Protein (CRP)", "Inflammation", 35, "4-6 hours"),
            ("Erythrocyte Sedimentation Rate", "Inflammation", 20, "2-4 hours"),
            ("Antinuclear Antibody (ANA)", "Immunology", 95, "24-48 hours"),
            ("Rheumatoid Factor", "Immunology", 65, "24 hours"),
            ("Blood Culture", "Microbiology", 120, "24-72 hours"),
            ("Procalcitonin", "Infection", 85, "4-6 hours"),
            ("D-Dimer", "Coagulation", 55, "2-4 hours"),
            ("Troponin I/T", "Cardiac", 75, "1-2 hours"),
            ("BNP/NT-proBNP", "Cardiac", 90, "2-4 hours"),
            ("HbA1c", "Diabetes", 50, "24 hours"),
        ]
        
        for name, category, cost, turnaround in additional_tests:
            table.add_row(name, category, f"${cost}", turnaround)
    
    def _render_summary(self) -> Text:
        """Render the order summary."""
        text = Text()
        
        num_selected = len(self.selected_tests)
        
        # Calculate estimated cost
        tests = self.case_data.get("test_recommendations", [])
        total_cost = sum(
            tests[i].get("estimated_cost", 50) 
            for i in self.selected_tests 
            if i < len(tests)
        )
        
        text.append(f"Selected: {num_selected} test(s)", style="#F1F5F9")
        text.append(" | ", style="#334155")
        text.append(f"Estimated Cost: ${total_cost}", style="#22C55E")
        
        return text
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn_order":
            self._place_order()
        elif button_id == "btn_custom":
            self._add_custom_order()
    
    def _place_order(self) -> None:
        """Place order for selected tests."""
        if not self.selected_tests:
            self.notify("No tests selected", severity="warning")
            return
        
        store = get_data_store()
        tests = self.case_data.get("test_recommendations", [])
        ordered_tests = [tests[i].get("test_name", "") for i in self.selected_tests if i < len(tests)]
        
        # Log the order
        store.audit.log_event(
            event_type="tests_ordered",
            user="system",
            action=f"Ordered {len(ordered_tests)} tests for case {self.case_id}",
            details={"tests": ordered_tests, "case_id": self.case_id},
        )
        
        self.notify(f"Ordered {len(ordered_tests)} tests successfully!")
        self.selected_tests.clear()
        self._update_displays()
    
    def _add_custom_order(self) -> None:
        """Add a custom test order."""
        self.notify("Custom order dialog would open here")
    
    def _update_displays(self) -> None:
        """Update all display widgets."""
        try:
            self.query_one("#recommended_tests", Static).update(self._render_recommended_tests())
            self.query_one("#summary", Static).update(self._render_summary())
        except Exception:
            pass
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in additional tests table."""
        row_index = event.cursor_row
        
        # Toggle selection (using different index space for additional tests)
        adjusted_index = row_index + 1000  # Offset to distinguish from recommended
        if adjusted_index in self.selected_tests:
            self.selected_tests.discard(adjusted_index)
        else:
            self.selected_tests.add(adjusted_index)
        
        self._update_displays()
    
    def action_order_selected(self) -> None:
        """Order the selected tests."""
        self._place_order()
    
    def action_go_back(self) -> None:
        """Go back to previous screen."""
        self.app.pop_screen()
    
    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()
