"""
Diagnosis Result Screen for BioSage Terminal.
Displays comprehensive diagnosis results from ARGUS debate framework.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Any
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import (
    Header, Footer, Static, Button, Collapsible,
    DataTable, TabbedContent, TabPane,
)
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from rich.text import Text
from rich.panel import Panel
from rich.table import Table

from biosage_terminal.storage import get_data_store
from biosage_terminal.theme import COLORS


class DifferentialCard(Static):
    """A card displaying a differential diagnosis."""
    
    def __init__(
        self,
        diagnosis: str,
        score: float,
        why_top: str,
        citations: list[dict],
        rank: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.diagnosis = diagnosis
        self.score = score
        self.why_top = why_top
        self.citations = citations
        self.rank = rank
        
    def compose(self) -> ComposeResult:
        # Color based on rank
        rank_colors = {
            1: COLORS["success"],
            2: COLORS["info"],
            3: COLORS["warning"],
        }
        rank_color = rank_colors.get(self.rank, COLORS["text_muted"])
        
        bar_filled = int(self.score * 20)
        bar = f"[{rank_color}]" + "#" * bar_filled + f"[/{rank_color}]" + "-" * (20 - bar_filled)
        
        citations_str = ""
        for c in self.citations[:2]:
            citations_str += f"\n  - {c.get('span', '')[:60]}... (Source: {c.get('doc_id', 'Unknown')})"
        
        content = f"""[bold]#{self.rank} {self.diagnosis}[/bold] [{rank_color}]{self.score * 100:.1f}%[/{rank_color}]
{bar}

[dim]{self.why_top}[/dim]
{citations_str if citations_str else '[dim]No citations available[/dim]'}"""
        
        yield Static(content, classes="differential-card")


class RecommendationCard(Static):
    """A card displaying a recommendation."""
    
    def __init__(
        self,
        title: str,
        rationale: str,
        priority: str = "medium",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.title = title
        self.rationale = rationale
        self.priority = priority
        
    def compose(self) -> ComposeResult:
        priority_colors = {
            "high": COLORS["error"],
            "medium": COLORS["warning"],
            "low": COLORS["text_muted"],
        }
        priority_color = priority_colors.get(self.priority.lower(), COLORS["text_muted"])
        
        content = f"""[bold]{self.title}[/bold] [{priority_color}][{self.priority.upper()}][/{priority_color}]
[dim]{self.rationale}[/dim]"""
        
        yield Static(content, classes="recommendation-card")


class TestPlanCard(Static):
    """A card displaying a test plan for a diagnosis."""
    
    def __init__(
        self,
        diagnosis: str,
        plan: str,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.diagnosis = diagnosis
        self.plan = plan
        
    def compose(self) -> ComposeResult:
        content = f"""[bold]{self.diagnosis}[/bold]
[dim]{self.plan}[/dim]"""
        
        yield Static(content, classes="test-plan-card")


class DiagnosisResultScreen(Screen):
    """Comprehensive diagnosis result display."""
    
    BINDINGS = [
        ("escape", "go_back", "Back"),
        ("d", "goto_dashboard", "Dashboard"),
        ("t", "goto_tests", "Order Tests"),
        ("p", "goto_patient", "Patient"),
        ("e", "export_report", "Export"),
        ("1", "tab_differential", "Differential"),
        ("2", "tab_recommendations", "Recommendations"),
        ("3", "tab_test_plans", "Test Plans"),
    ]
    
    CSS = """
    DiagnosisResultScreen {
        background: #0F172A;
    }
    
    .result-header {
        dock: top;
        height: 3;
        background: #3B82F6;
        color: #F1F5F9;
        padding: 0 2;
    }
    
    .summary-section {
        height: 8;
        padding: 1 2;
        background: #1E293B;
        border: solid #3B82F6;
        margin: 1;
    }
    
    .differential-card {
        padding: 1;
        margin: 1;
        background: #1E293B;
        border: solid #3B82F6;
    }
    
    .recommendation-card {
        padding: 1;
        margin: 1;
        background: #1E293B;
        border: solid #0EA5E9;
    }
    
    .test-plan-card {
        padding: 1;
        margin: 1;
        background: #1E293B;
        border: solid #F59E0B;
    }
    
    .section-title {
        text-style: bold;
        color: #8B5CF6;
        padding: 1;
    }
    
    .metric-box {
        padding: 1;
        margin: 0 1;
        background: #1E293B;
        text-align: center;
    }
    
    .metrics-row {
        layout: grid;
        grid-size: 4;
        height: 5;
        padding: 1;
    }
    
    .next-test-box {
        padding: 1;
        margin: 1;
        background: #22C55E;
        color: #F1F5F9;
        border: solid #22C55E;
    }
    
    .action-buttons {
        height: 5;
        padding: 1;
        dock: bottom;
        background: #1E293B;
        border-top: solid #334155;
        margin-top: 1;
    }
    """
    
    def __init__(
        self,
        case_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        diagnosis_data: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.case_id = case_id
        self.patient_id = patient_id
        self.store = get_data_store()
        
        # Use provided data or load from storage
        if diagnosis_data:
            self.diagnosis_data = diagnosis_data
        else:
            self.diagnosis_data = self._load_diagnosis_data()
    
    def _load_diagnosis_data(self) -> dict:
        """Load diagnosis data from storage or return demo data."""
        if self.case_id:
            case_data = self.store.cases.load_sync(self.case_id)
            if case_data and "diagnosis_result" in case_data:
                return case_data["diagnosis_result"]
        
        # Return demo data
        return {
            "fused": {
                "differential": [
                    {
                        "diagnosis": "Systemic Lupus Erythematosus (SLE)",
                        "score_global": 0.89,
                        "why_top": "Patient presents with classic SLE features including malar rash, joint pain, and positive ANA with anti-dsDNA antibodies. The multi-system involvement is characteristic.",
                        "citations": [
                            {"doc_id": "NEJM-2023-SLE-Review", "span": "Malar rash occurs in 50% of SLE patients and is highly specific"},
                            {"doc_id": "ACR-Guidelines-2019", "span": "Anti-dsDNA positivity with clinical features meets ACR criteria"}
                        ]
                    },
                    {
                        "diagnosis": "Mixed Connective Tissue Disease (MCTD)",
                        "score_global": 0.72,
                        "why_top": "Overlapping features of SLE and systemic sclerosis. Consider if anti-U1 RNP is positive.",
                        "citations": [
                            {"doc_id": "Rheumatology-2021", "span": "MCTD presents with features of multiple connective tissue diseases"}
                        ]
                    },
                    {
                        "diagnosis": "Drug-Induced Lupus",
                        "score_global": 0.34,
                        "why_top": "Less likely given no recent drug exposure history, but should review medication history.",
                        "citations": [
                            {"doc_id": "Pharmacology-Review-2020", "span": "Hydralazine and procainamide are common triggers"}
                        ]
                    }
                ],
                "next_best_test": {
                    "name": "Anti-Sm and Anti-RNP Antibodies",
                    "why": "Will help differentiate between SLE and MCTD. Anti-Sm is highly specific for SLE.",
                    "linked_hypotheses": ["SLE", "MCTD"]
                },
                "disagreement_score": 0.15,
                "test_plans": [
                    {
                        "diagnosis": "SLE",
                        "plan": "Complete autoantibody panel (Anti-Sm, Anti-RNP, Anti-Ro, Anti-La), Complement levels (C3, C4), Renal function, Urinalysis for proteinuria"
                    },
                    {
                        "diagnosis": "MCTD",
                        "plan": "Anti-U1 RNP titers, Pulmonary function tests, Echocardiogram for pulmonary hypertension screening"
                    }
                ]
            },
            "recommendations": [
                {
                    "title": "Order Comprehensive Autoantibody Panel",
                    "rationale": "Essential for confirming SLE diagnosis and ruling out overlap syndromes. Include Anti-Sm, Anti-RNP, Anti-Ro, Anti-La.",
                    "priority": "high"
                },
                {
                    "title": "Initiate Hydroxychloroquine",
                    "rationale": "First-line therapy for SLE with excellent safety profile. Helps prevent flares and organ damage.",
                    "priority": "high"
                },
                {
                    "title": "Rheumatology Referral",
                    "rationale": "Multidisciplinary care recommended for autoimmune conditions. Specialist should guide long-term management.",
                    "priority": "medium"
                },
                {
                    "title": "Sun Protection Counseling",
                    "rationale": "Photosensitivity is common in SLE. Educate on UV protection to prevent flares.",
                    "priority": "low"
                }
            ]
        }
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        fused = self.diagnosis_data.get("fused", {})
        recommendations = self.diagnosis_data.get("recommendations", [])
        
        with Container(classes="result-header"):
            yield Static(
                f"[bold]Diagnosis Result[/bold] | Case: {self.case_id or 'BSG-2024-001'} | "
                "[t] Tests [p] Patient [e] Export [ESC] Back"
            )
        
        # Summary metrics row
        differential = fused.get("differential", [])
        top_dx = differential[0] if differential else {"diagnosis": "Unknown", "score_global": 0}
        next_test = fused.get("next_best_test", {})
        disagreement = fused.get("disagreement_score", 0)
        
        with Horizontal(classes="metrics-row"):
            yield Static(
                f"[bold]Top Diagnosis[/bold]\n[{COLORS['success']}]{top_dx['diagnosis'][:25]}[/{COLORS['success']}]\n{top_dx['score_global'] * 100:.1f}%",
                classes="metric-box"
            )
            yield Static(
                f"[bold]Confidence[/bold]\n[{COLORS['accent_purple']}]{top_dx['score_global'] * 100:.1f}%[/{COLORS['accent_purple']}]",
                classes="metric-box"
            )
            yield Static(
                f"[bold]Disagreement[/bold]\n[{COLORS['warning']}]{disagreement * 100:.1f}%[/{COLORS['warning']}]",
                classes="metric-box"
            )
            yield Static(
                f"[bold]Specialists[/bold]\n[{COLORS['info']}]6 Agents[/{COLORS['info']}]",
                classes="metric-box"
            )
        
        # Next best test
        with Container(classes="next-test-box"):
            yield Static(
                f"[bold]Next Best Test:[/bold] {next_test.get('name', 'Unknown')}\n"
                f"[dim]{next_test.get('why', 'No rationale provided')}[/dim]"
            )
        
        # Tabbed content
        with TabbedContent(id="result-tabs"):
            with TabPane("Differential Diagnosis", id="tab-differential"):
                with ScrollableContainer():
                    yield Static("[bold]Differential Diagnosis[/bold]", classes="section-title")
                    
                    for i, dx in enumerate(differential, 1):
                        yield DifferentialCard(
                            diagnosis=dx.get("diagnosis", "Unknown"),
                            score=dx.get("score_global", 0),
                            why_top=dx.get("why_top", ""),
                            citations=dx.get("citations", []),
                            rank=i
                        )
            
            with TabPane("Recommendations", id="tab-recommendations"):
                with ScrollableContainer():
                    yield Static("[bold]Clinical Recommendations[/bold]", classes="section-title")
                    
                    for rec in recommendations:
                        yield RecommendationCard(
                            title=rec.get("title", ""),
                            rationale=rec.get("rationale", ""),
                            priority=rec.get("priority", "medium")
                        )
            
            with TabPane("Test Plans", id="tab-test-plans"):
                with ScrollableContainer():
                    yield Static("[bold]Diagnosis-Specific Test Plans[/bold]", classes="section-title")
                    
                    test_plans = fused.get("test_plans", [])
                    for plan in test_plans:
                        yield TestPlanCard(
                            diagnosis=plan.get("diagnosis", ""),
                            plan=plan.get("plan", "")
                        )
        
        # Action buttons
        with Horizontal(classes="action-buttons"):
            yield Button("Order Tests", id="btn-tests", variant="primary")
            yield Button("View Patient", id="btn-patient", variant="default")
            yield Button("Export Report", id="btn-export", variant="default")
            yield Button("Back to Dashboard", id="btn-dashboard", variant="default")
        
        yield Footer()
    
    def action_go_back(self) -> None:
        """Go back to previous screen."""
        self.app.pop_screen()
    
    def action_goto_dashboard(self) -> None:
        """Navigate to dashboard."""
        self.app.switch_screen("dashboard")
    
    def action_goto_tests(self) -> None:
        """Navigate to tests orders screen."""
        from biosage_terminal.screens.tests_orders import TestsOrdersScreen
        self.app.push_screen(TestsOrdersScreen(
            case_id=self.case_id,
            patient_id=self.patient_id,
            diagnosis_data=self.diagnosis_data
        ))
    
    def action_goto_patient(self) -> None:
        """Navigate to patient summary."""
        if self.patient_id:
            from biosage_terminal.screens.patient_summary import PatientSummaryScreen
            self.app.push_screen(PatientSummaryScreen(patient_id=self.patient_id))
        else:
            self.notify("No patient associated with this case", severity="warning")
    
    def action_export_report(self) -> None:
        """Export diagnosis report."""
        report_content = self._generate_report()
        report_id = f"diagnosis_report_{self.case_id or 'unknown'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        path = self.store.save_report(report_id, report_content, "md")
        
        self.store.audit.log_event(
            event_type="report",
            user="current_user",
            action="export_diagnosis_report",
            details={"case_id": self.case_id, "path": str(path)}
        )
        
        self.notify(f"Report exported to {path}", severity="information")
    
    def _generate_report(self) -> str:
        """Generate markdown report."""
        fused = self.diagnosis_data.get("fused", {})
        recommendations = self.diagnosis_data.get("recommendations", [])
        
        report = f"""# Diagnosis Report
## Case: {self.case_id or 'Unknown'}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Differential Diagnosis

"""
        
        for i, dx in enumerate(fused.get("differential", []), 1):
            report += f"""### {i}. {dx.get('diagnosis', 'Unknown')} ({dx.get('score_global', 0) * 100:.1f}%)

{dx.get('why_top', '')}

"""
        
        next_test = fused.get("next_best_test", {})
        report += f"""## Next Best Test

**{next_test.get('name', 'Unknown')}**

{next_test.get('why', '')}

---

## Recommendations

"""
        
        for rec in recommendations:
            report += f"""### {rec.get('title', '')} [{rec.get('priority', 'medium').upper()}]

{rec.get('rationale', '')}

"""
        
        return report
    
    def action_tab_differential(self) -> None:
        """Switch to differential tab."""
        tabs = self.query_one("#result-tabs", TabbedContent)
        tabs.active = "tab-differential"
    
    def action_tab_recommendations(self) -> None:
        """Switch to recommendations tab."""
        tabs = self.query_one("#result-tabs", TabbedContent)
        tabs.active = "tab-recommendations"
    
    def action_tab_test_plans(self) -> None:
        """Switch to test plans tab."""
        tabs = self.query_one("#result-tabs", TabbedContent)
        tabs.active = "tab-test-plans"
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-tests":
            self.action_goto_tests()
        elif button_id == "btn-patient":
            self.action_goto_patient()
        elif button_id == "btn-export":
            self.action_export_report()
        elif button_id == "btn-dashboard":
            self.action_goto_dashboard()
