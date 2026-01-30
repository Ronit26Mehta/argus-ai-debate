"""
BioSage Terminal Application - Main App Class.

AI-Powered Medical Diagnostic Assistant using ARGUS debate framework.
"""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import Screen

from biosage_terminal.screens.welcome import WelcomeScreen
from biosage_terminal.screens.dashboard import DashboardScreen
from biosage_terminal.screens.onboarding import OnboardingScreen
from biosage_terminal.screens.case_view import CaseViewScreen
from biosage_terminal.screens.diagnosis_progress import DiagnosisProgressScreen
from biosage_terminal.screens.specialists import SpecialistGridScreen
from biosage_terminal.screens.evidence import EvidenceExplorerScreen
from biosage_terminal.screens.patient_summary import PatientSummaryScreen
from biosage_terminal.screens.admin_audit import AdminAuditScreen
from biosage_terminal.screens.model_operations import ModelOperationsScreen
from biosage_terminal.screens.tests_orders import TestsOrdersScreen
from biosage_terminal.screens.collaboration_room import CollaborationRoomScreen
from biosage_terminal.screens.research_hub import ResearchHubScreen
from biosage_terminal.screens.visual_diagnosis import VisualDiagnosisScreen
from biosage_terminal.screens.diagnosis_result import DiagnosisResultScreen
from biosage_terminal.screens.cases_manager import CasesManagerScreen
from biosage_terminal.screens.help import HelpScreen


class BioSageApp(App):
    """BioSage TUI Application - Medical Diagnostic Assistant."""
    
    TITLE = "BioSage - AI Medical Diagnosis"
    SUB_TITLE = "Powered by ARGUS Debate Framework"
    
    CSS = """
    Screen {
        background: #0F172A;
    }
    
    .title-bar {
        background: #1E293B;
        height: 3;
        padding: 0 1;
        border-bottom: solid #334155;
    }
    
    .status-bar {
        background: #1E293B;
        height: 1;
        dock: bottom;
        padding: 0 1;
    }
    
    Button {
        background: #1E293B;
        border: solid #3B82F6;
    }
    
    Button:hover {
        background: #3B82F6;
    }
    
    Button:focus {
        background: #2563EB;
    }
    
    Static {
        color: #F1F5F9;
    }
    
    Input {
        background: #1E293B;
        border: solid #334155;
        color: #F1F5F9;
    }
    
    Input:focus {
        border: solid #3B82F6;
    }
    
    TextArea {
        background: #1E293B;
        border: solid #334155;
        color: #F1F5F9;
    }
    
    ListView {
        background: #1E293B;
        border: solid #334155;
    }
    
    ListItem {
        background: #1E293B;
        color: #F1F5F9;
    }
    
    ListItem:hover {
        background: #334155;
    }
    
    ListItem.-selected {
        background: #3B82F6;
    }
    
    Select {
        background: #1E293B;
        border: solid #334155;
        color: #F1F5F9;
    }
    
    Select:focus {
        border: solid #3B82F6;
    }
    
    DataTable {
        background: #0F172A;
    }
    
    DataTable > .datatable--header {
        background: #1E293B;
        color: #F1F5F9;
        text-style: bold;
    }
    
    DataTable > .datatable--cursor {
        background: #3B82F6;
    }
    
    TabbedContent ContentSwitcher {
        background: #0F172A;
    }
    
    TabPane {
        background: #0F172A;
    }
    
    Tabs {
        background: #1E293B;
    }
    
    Tab {
        background: #1E293B;
        color: #94A3B8;
    }
    
    Tab.-active {
        background: #3B82F6;
        color: #FFFFFF;
    }
    
    ProgressBar Bar {
        color: #3B82F6;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("escape", "go_back", "Back", show=True),
        Binding("?", "show_help", "Help", show=True),
        Binding("l", "goto_cases_manager", "Cases", show=False),
        Binding("m", "goto_model_ops", "Model Ops", show=False),
        Binding("a", "goto_audit", "Audit", show=False),
        Binding("c", "goto_collaboration", "Collab", show=False),
        Binding("r", "goto_research", "Research", show=False),
        Binding("v", "goto_visual", "Visual Dx", show=False),
    ]
    
    SCREENS = {
        "welcome": WelcomeScreen,
        "dashboard": DashboardScreen,
        "onboarding": OnboardingScreen,
        "specialists": SpecialistGridScreen,
        "evidence": EvidenceExplorerScreen,
        "admin_audit": AdminAuditScreen,
        "model_operations": ModelOperationsScreen,
        "collaboration": CollaborationRoomScreen,
        "research": ResearchHubScreen,
        "visual": VisualDiagnosisScreen,
        "cases_manager": CasesManagerScreen,
        "help": HelpScreen,
    }
    
    def __init__(self):
        """Initialize the BioSage application."""
        super().__init__()
        self._current_case_id: str | None = None
        self._current_patient_id: str | None = None
    
    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.push_screen("welcome")
    
    def action_go_back(self) -> None:
        """Go back to the previous screen."""
        if len(self.screen_stack) > 1:
            self.pop_screen()
    
    def action_show_help(self) -> None:
        """Show help screen."""
        self.push_screen("help")
    
    def goto_dashboard(self) -> None:
        """Navigate to dashboard screen."""
        self.push_screen("dashboard")
    
    def goto_onboarding(self) -> None:
        """Navigate to onboarding/new case screen."""
        self.push_screen("onboarding")
    
    def goto_cases_manager(self) -> None:
        """Navigate to cases manager screen."""
        self.push_screen("cases_manager")
    
    def goto_case(self, case_id: str, patient_id: str = "") -> None:
        """Navigate to case view screen."""
        self._current_case_id = case_id
        self._current_patient_id = patient_id
        screen = CaseViewScreen(case_id=case_id)
        self.push_screen(screen)
    
    def goto_diagnosis(self, case_id: str, patient_id: str) -> None:
        """Navigate to diagnosis progress screen."""
        self._current_case_id = case_id
        self._current_patient_id = patient_id
        screen = DiagnosisProgressScreen(case_id=case_id, patient_id=patient_id)
        self.push_screen(screen)
    
    def goto_specialists(self) -> None:
        """Navigate to specialists overview screen."""
        self.push_screen("specialists")
    
    def goto_evidence(self) -> None:
        """Navigate to evidence library screen."""
        self.push_screen("evidence")
    
    def goto_patient_summary(self, patient_id: str) -> None:
        """Navigate to patient summary screen."""
        self._current_patient_id = patient_id
        screen = PatientSummaryScreen(patient_id=patient_id)
        self.push_screen(screen)
    
    def goto_tests_orders(self, case_id: str) -> None:
        """Navigate to tests and orders screen."""
        self._current_case_id = case_id
        screen = TestsOrdersScreen(case_id=case_id)
        self.push_screen(screen)
    
    def action_goto_model_ops(self) -> None:
        """Navigate to model operations screen."""
        self.push_screen("model_operations")
    
    def action_goto_audit(self) -> None:
        """Navigate to audit screen."""
        self.push_screen("admin_audit")
    
    def action_goto_cases_manager(self) -> None:
        """Navigate to cases manager screen."""
        self.push_screen("cases_manager")
    
    def action_goto_collaboration(self) -> None:
        """Navigate to collaboration room screen."""
        self.push_screen("collaboration")
    
    def action_goto_research(self) -> None:
        """Navigate to research hub screen."""
        self.push_screen("research")
    
    def action_goto_visual(self) -> None:
        """Navigate to visual diagnosis screen."""
        self.push_screen("visual")
    
    def goto_diagnosis_result(self, case_id: str, patient_id: str = "", diagnosis_data: dict = None) -> None:
        """Navigate to diagnosis result screen."""
        self._current_case_id = case_id
        self._current_patient_id = patient_id
        screen = DiagnosisResultScreen(case_id=case_id, patient_id=patient_id, diagnosis_data=diagnosis_data)
        self.push_screen(screen)
    
    def goto_collaboration(self, case_id: str = None) -> None:
        """Navigate to collaboration room for a case."""
        screen = CollaborationRoomScreen(case_id=case_id)
        self.push_screen(screen)
    
    def goto_research(self, case_id: str = None) -> None:
        """Navigate to research hub for a case."""
        screen = ResearchHubScreen(case_id=case_id)
        self.push_screen(screen)


def run() -> None:
    """Run the BioSage application."""
    app = BioSageApp()
    app.run()


if __name__ == "__main__":
    run()
