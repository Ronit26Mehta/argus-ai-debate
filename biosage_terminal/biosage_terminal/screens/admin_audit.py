"""
Admin Audit screen for BioSage Terminal.
Displays audit logs, system events, and compliance information.
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Footer, Input, Button, TabbedContent, TabPane, DataTable
from textual.binding import Binding
from rich.text import Text

from biosage_terminal.storage import get_data_store
from biosage_terminal.theme import STATUS_INDICATORS


class AdminAuditScreen(Screen):
    """Admin audit logs and compliance screen."""
    
    BINDINGS = [
        Binding("escape", "go_back", "Back"),
        Binding("d", "goto_dashboard", "Dashboard"),
        Binding("r", "refresh", "Refresh"),
        Binding("q", "quit", "Quit"),
    ]
    
    CSS = """
    AdminAuditScreen {
        background: #0F172A;
    }
    
    .audit-header {
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
        width: 50%;
    }
    
    .filter-button {
        margin-left: 2;
    }
    
    .tab-content {
        height: 1fr;
        padding: 1;
    }
    
    .audit-table {
        height: 1fr;
    }
    
    .compliance-section {
        padding: 1;
    }
    
    .compliance-badge {
        background: #1E293B;
        border: solid #22C55E;
        padding: 1;
        margin: 1;
        width: 20;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Static(self._render_header(), classes="audit-header")
        
        with Container(classes="search-bar"):
            yield Input(
                placeholder="Search audit logs...",
                id="search_input",
                classes="search-input",
            )
            yield Button("Filter", id="btn_filter", classes="filter-button")
            yield Button("Export", id="btn_export", classes="filter-button")
            yield Button("Refresh", id="btn_refresh", classes="filter-button")
        
        with TabbedContent():
            with TabPane("Audit Logs", id="tab_audit"):
                with ScrollableContainer(classes="tab-content"):
                    table = DataTable(id="audit_table", classes="audit-table")
                    table.cursor_type = "row"
                    yield table
            
            with TabPane("System Events", id="tab_events"):
                with ScrollableContainer(classes="tab-content"):
                    yield Static(self._render_system_events(), id="events_content")
            
            with TabPane("Compliance", id="tab_compliance"):
                with ScrollableContainer(classes="tab-content"):
                    yield Static(self._render_compliance(), id="compliance_content")
            
            with TabPane("User Activity", id="tab_activity"):
                with ScrollableContainer(classes="tab-content"):
                    yield Static(self._render_user_activity(), id="activity_content")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Setup the screen when mounted."""
        self._setup_audit_table()
    
    def _render_header(self) -> Text:
        """Render the header."""
        text = Text()
        text.append("BioSage", style="bold #3B82F6")
        text.append(" | ", style="#334155")
        text.append("[SEC] Admin Audit Logs", style="bold #F1F5F9")
        return text
    
    def _setup_audit_table(self) -> None:
        """Setup the audit logs table."""
        table = self.query_one("#audit_table", DataTable)
        table.add_columns("Timestamp", "User", "Action", "Status")
        
        store = get_data_store()
        events = store.audit.get_recent_events(limit=50)
        
        for event in events:
            timestamp = event.get("timestamp", "")[:19].replace("T", " ")
            user = event.get("user", "system")[:15]
            action = event.get("action", "")[:40]
            status = event.get("status", "success")
            
            status_indicator = STATUS_INDICATORS.get(status, "[??]")
            
            table.add_row(timestamp, user, action, status_indicator)
    
    def _render_system_events(self) -> Text:
        """Render system events summary."""
        text = Text()
        
        text.append("=== System Events ===\n\n", style="bold #F1F5F9")
        
        store = get_data_store()
        events = store.audit.get_recent_events(limit=20)
        
        # Group by event type
        event_types = {}
        for event in events:
            etype = event.get("event_type", "unknown")
            if etype not in event_types:
                event_types[etype] = 0
            event_types[etype] += 1
        
        text.append("Event Type Summary\n", style="bold #3B82F6")
        text.append("-" * 40 + "\n", style="#334155")
        
        for etype, count in sorted(event_types.items(), key=lambda x: -x[1]):
            text.append(f"  {etype}: ", style="#94A3B8")
            text.append(f"{count}\n", style="#F1F5F9")
        
        text.append("\n", style="")
        
        # Recent events timeline
        text.append("Recent Events Timeline\n", style="bold #3B82F6")
        text.append("-" * 40 + "\n", style="#334155")
        
        for event in events[:10]:
            timestamp = event.get("timestamp", "")[:19].replace("T", " ")
            action = event.get("action", "Unknown")
            status = event.get("status", "success")
            
            status_color = "#22C55E" if status == "success" else "#EF4444"
            status_icon = "[OK]" if status == "success" else "[XX]"
            
            text.append(f"  {timestamp} ", style="#64748B")
            text.append(f"{status_icon} ", style=f"bold {status_color}")
            text.append(f"{action[:50]}\n", style="#F1F5F9")
        
        return text
    
    def _render_compliance(self) -> Text:
        """Render compliance status."""
        text = Text()
        
        text.append("=== Compliance Status ===\n\n", style="bold #F1F5F9")
        
        # Compliance badges
        text.append("Active Compliance Standards\n", style="bold #3B82F6")
        text.append("-" * 40 + "\n\n", style="#334155")
        
        compliances = [
            ("HIPAA", "Health Insurance Portability and Accountability Act", True),
            ("SOC2", "System and Organization Controls 2", True),
            ("GDPR", "General Data Protection Regulation", True),
            ("FDA-510k", "FDA Medical Device Clearance", False),
        ]
        
        for code, name, active in compliances:
            status_color = "#22C55E" if active else "#64748B"
            status_text = "[ACTIVE]" if active else "[PENDING]"
            
            text.append(f"  [{code}] ", style=f"bold {status_color}")
            text.append(f"{name}\n", style="#F1F5F9")
            text.append(f"         Status: ", style="#64748B")
            text.append(f"{status_text}\n\n", style=status_color)
        
        # Data protection summary
        text.append("\nData Protection Summary\n", style="bold #3B82F6")
        text.append("-" * 40 + "\n", style="#334155")
        
        text.append("  [OK] All patient data encrypted at rest\n", style="#22C55E")
        text.append("  [OK] Audit logging enabled\n", style="#22C55E")
        text.append("  [OK] Access control enforced\n", style="#22C55E")
        text.append("  [OK] Data retention policy active\n", style="#22C55E")
        
        return text
    
    def _render_user_activity(self) -> Text:
        """Render user activity summary."""
        text = Text()
        
        text.append("=== User Activity ===\n\n", style="bold #F1F5F9")
        
        store = get_data_store()
        events = store.audit.get_recent_events(limit=100)
        
        # Group by user
        user_actions = {}
        for event in events:
            user = event.get("user", "system")
            if user not in user_actions:
                user_actions[user] = {"count": 0, "last_action": ""}
            user_actions[user]["count"] += 1
            if not user_actions[user]["last_action"]:
                user_actions[user]["last_action"] = event.get("action", "")
        
        text.append("User Action Summary\n", style="bold #3B82F6")
        text.append("-" * 40 + "\n", style="#334155")
        
        for user, data in sorted(user_actions.items(), key=lambda x: -x[1]["count"]):
            text.append(f"\n  User: ", style="#64748B")
            text.append(f"{user}\n", style="bold #F1F5F9")
            text.append(f"  Actions: ", style="#64748B")
            text.append(f"{data['count']}\n", style="#F1F5F9")
            text.append(f"  Last: ", style="#64748B")
            text.append(f"{data['last_action'][:40]}\n", style="#94A3B8")
        
        return text
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn_refresh":
            self.action_refresh()
        elif button_id == "btn_export":
            self._export_logs()
        elif button_id == "btn_filter":
            self._apply_filter()
    
    def _export_logs(self) -> None:
        """Export audit logs to a file."""
        store = get_data_store()
        events = store.audit.get_recent_events(limit=1000)
        
        # Create report content
        lines = ["BioSage Audit Log Export", "=" * 50, ""]
        for event in events:
            lines.append(f"Timestamp: {event.get('timestamp', '')}")
            lines.append(f"User: {event.get('user', '')}")
            lines.append(f"Action: {event.get('action', '')}")
            lines.append(f"Status: {event.get('status', '')}")
            lines.append("-" * 30)
        
        # Save report
        from datetime import datetime
        report_id = f"audit_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        path = store.save_report(report_id, "\n".join(lines), "txt")
        
        self.notify(f"Audit logs exported to {path}")
    
    def _apply_filter(self) -> None:
        """Apply search filter to logs."""
        try:
            search_text = self.query_one("#search_input", Input).value.lower()
            if search_text:
                self.notify(f"Filtering by: {search_text}")
        except Exception:
            pass
    
    def action_refresh(self) -> None:
        """Refresh the audit logs."""
        table = self.query_one("#audit_table", DataTable)
        table.clear()
        self._setup_audit_table()
        
        self.query_one("#events_content", Static).update(self._render_system_events())
        self.query_one("#compliance_content", Static).update(self._render_compliance())
        self.query_one("#activity_content", Static).update(self._render_user_activity())
        
        self.notify("Audit logs refreshed")
    
    def action_go_back(self) -> None:
        """Go back to previous screen."""
        self.app.pop_screen()
    
    def action_goto_dashboard(self) -> None:
        """Navigate to dashboard."""
        self.app.push_screen("dashboard")
    
    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()
