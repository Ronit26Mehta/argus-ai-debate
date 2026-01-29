"""
Research Hub Screen for BioSage Terminal.
AI-curated research insights, clinical trial matching, and research briefs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import (
    Header, Footer, Static, Button, Input, Switch, Label,
    DataTable, TabbedContent, TabPane,
)
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from rich.text import Text
from rich.panel import Panel
from rich.table import Table

from biosage_terminal.storage import get_data_store
from biosage_terminal.theme import COLORS


class ResearchSuggestionCard(Static):
    """Card for research suggestions."""
    
    def __init__(
        self,
        title: str,
        category: str,
        confidence: float,
        description: str,
        suggested_actions: list[str],
        similar_cases: int,
        literature: int,
        priority: str = "medium",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.title = title
        self.category = category
        self.confidence = confidence
        self.description = description
        self.suggested_actions = suggested_actions
        self.similar_cases = similar_cases
        self.literature = literature
        self.priority = priority
        
    def compose(self) -> ComposeResult:
        priority_color = {
            "high": COLORS["error"],
            "medium": COLORS["warning"],
            "low": COLORS["text_muted"],
        }.get(self.priority, COLORS["text_muted"])
        
        actions_str = " | ".join(self.suggested_actions[:3])
        
        content = f"""[bold]{self.title}[/bold]
[{priority_color}][{self.priority.upper()} PRIORITY][/{priority_color}] | [{COLORS["info"]}]{self.category.replace('-', ' ').title()}[/{COLORS["info"]}]
[dim]{self.description}[/dim]

Confidence: [{COLORS["accent_purple"]}]{int(self.confidence * 100)}%[/{COLORS["accent_purple"]}] | Similar Cases: {self.similar_cases} | Literature: {self.literature}
Suggested: [dim]{actions_str}[/dim]"""
        
        yield Static(content, classes="research-card")


class ClinicalTrialCard(Static):
    """Card for clinical trials."""
    
    def __init__(
        self,
        trial_id: str,
        title: str,
        phase: str,
        status: str,
        location: str,
        match_score: float,
        eligibility: str,
        endpoint: str,
        contact: str,
        distance: str,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.trial_id = trial_id
        self.title = title
        self.phase = phase
        self.status = status
        self.location = location
        self.match_score = match_score
        self.eligibility = eligibility
        self.endpoint = endpoint
        self.contact = contact
        self.distance = distance
        
    def compose(self) -> ComposeResult:
        status_color = COLORS["success"] if self.status == "recruiting" else COLORS["text_muted"]
        
        content = f"""[bold]{self.title}[/bold]
[{status_color}]{self.status.upper()}[/{status_color}] | {self.phase} | {self.trial_id}
{self.location} ({self.distance})

Match Score: [{COLORS["accent_purple"]}]{int(self.match_score * 100)}%[/{COLORS["accent_purple"]}]
Eligibility: [dim]{self.eligibility[:80]}...[/dim]
Endpoint: [dim]{self.endpoint}[/dim]
Contact: {self.contact}"""
        
        yield Static(content, classes="trial-card")


class ResearchBriefCard(Static):
    """Card for research briefs."""
    
    def __init__(
        self,
        title: str,
        author: str,
        date_generated: str,
        pages: int,
        sections: list[str],
        key_insights: list[str],
        citations: int,
        downloads: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.title = title
        self.author = author
        self.date_generated = date_generated
        self.pages = pages
        self.sections = sections
        self.key_insights = key_insights
        self.citations = citations
        self.downloads = downloads
        
    def compose(self) -> ComposeResult:
        sections_str = " | ".join(self.sections[:4])
        insights_str = "\n".join(f"  * {insight[:70]}..." if len(insight) > 70 else f"  * {insight}" for insight in self.key_insights[:3])
        
        content = f"""[bold]{self.title}[/bold]
By {self.author} | Generated {self.date_generated} | {self.pages} pages

Sections: [dim]{sections_str}[/dim]

Key Insights:
{insights_str}

[dim]Citations: {self.citations} | Downloads: {self.downloads}[/dim]"""
        
        yield Static(content, classes="brief-card")


class ResearchHubScreen(Screen):
    """Research hub with AI-curated insights."""
    
    BINDINGS = [
        ("escape", "go_back", "Back"),
        ("d", "goto_dashboard", "Dashboard"),
        ("h", "toggle_house_mode", "House Mode"),
        ("g", "generate_brief", "Generate Brief"),
        ("1", "tab_suggestions", "Suggestions"),
        ("2", "tab_trials", "Trials"),
        ("3", "tab_briefs", "Briefs"),
    ]
    
    CSS = """
    ResearchHubScreen {
        background: #0F172A;
    }
    
    .research-header {
        dock: top;
        height: 3;
        background: #3B82F6;
        color: #F1F5F9;
        padding: 0 2;
    }
    
    .search-row {
        height: 3;
        padding: 1 2;
        background: #1E293B;
    }
    
    .research-card {
        padding: 1;
        margin: 1;
        background: #1E293B;
        border: solid #3B82F6;
    }
    
    .trial-card {
        padding: 1;
        margin: 1;
        background: #1E293B;
        border: solid #0EA5E9;
    }
    
    .brief-card {
        padding: 1;
        margin: 1;
        background: #1E293B;
        border: solid #8B5CF6;
    }
    
    .stats-grid {
        layout: grid;
        grid-size: 3;
        height: 5;
        padding: 1;
    }
    
    .stat-box {
        text-align: center;
        padding: 1;
        background: #1E293B;
        margin: 0 1;
    }
    
    .action-buttons {
        height: 3;
        padding: 0 1;
    }
    
    .house-mode-indicator {
        color: #F59E0B;
    }
    """
    
    def __init__(self, case_id: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.case_id = case_id
        self.store = get_data_store()
        self.house_mode = False
        self.search_query = ""
        
        # Research suggestions data
        self.research_suggestions = [
            {
                "id": "rare-001",
                "title": "Lupus-like Syndrome in Young Adults with Complement Deficiency",
                "category": "rare-disease",
                "confidence": 0.73,
                "description": "Investigate C1q deficiency presenting as SLE-like symptoms in patients under 30",
                "suggested_actions": ["Genetic screening", "Family history analysis", "Immunological profiling"],
                "similar_cases": 3,
                "literature": 12,
                "priority": "medium"
            },
            {
                "id": "drug-rep-002",
                "title": "Hydroxychloroquine Alternatives in Refractory Autoimmune Cases",
                "category": "drug-repurposing",
                "confidence": 0.85,
                "description": "Novel antimalarial compounds showing promise for treatment-resistant lupus",
                "suggested_actions": ["Literature review", "Mechanism analysis", "Safety profile"],
                "similar_cases": 8,
                "literature": 27,
                "priority": "high"
            },
            {
                "id": "analogy-003",
                "title": "COVID-19 Induced Autoimmunity: Lessons from Viral Triggers",
                "category": "analogical-reasoning",
                "confidence": 0.68,
                "description": "Post-viral autoimmune syndrome patterns applicable to current case",
                "suggested_actions": ["Viral serology", "Temporal analysis", "Immune profiling"],
                "similar_cases": 15,
                "literature": 89,
                "priority": "low"
            }
        ]
        
        # Clinical trials data
        self.clinical_trials = [
            {
                "id": "NCT05234567",
                "title": "Phase II Trial of CAR-T Cell Therapy for Refractory Lupus",
                "phase": "Phase II",
                "status": "recruiting",
                "location": "Multiple Centers",
                "match_score": 0.89,
                "eligibility": "Age 18-65, Active SLE, Failed 2+ standard therapies",
                "endpoint": "SLEDAI-2K reduction at 24 weeks",
                "contact": "Dr. Sarah Johnson, Johns Hopkins",
                "distance": "12 miles"
            },
            {
                "id": "NCT05876543",
                "title": "Precision Medicine Approach to Lupus Treatment Selection",
                "phase": "Phase III",
                "status": "recruiting",
                "location": "Stanford Medical Center",
                "match_score": 0.76,
                "eligibility": "Newly diagnosed SLE, ANA positive",
                "endpoint": "Time to treatment response",
                "contact": "Dr. Michael Chen, Stanford University",
                "distance": "8 miles"
            },
            {
                "id": "NCT05123890",
                "title": "Biomarker-Guided Therapy in Lupus Nephritis",
                "phase": "Phase II/III",
                "status": "active",
                "location": "Mayo Clinic",
                "match_score": 0.67,
                "eligibility": "Lupus nephritis, eGFR >30",
                "endpoint": "Renal response at 52 weeks",
                "contact": "Dr. Lisa Wang, Mayo Clinic",
                "distance": "45 miles"
            }
        ]
        
        # Research briefs data
        self.research_briefs = [
            {
                "id": "brief-001",
                "title": "Systemic Lupus Erythematosus: Current Diagnostic Challenges",
                "author": "BioSage Research Team",
                "date_generated": "2024-01-15",
                "pages": 15,
                "sections": ["Current Guidelines", "Diagnostic Gaps", "Emerging Biomarkers", "Technology Integration"],
                "key_insights": [
                    "ANA testing sensitivity varies by substrate and technique",
                    "Machine learning models show 94% accuracy in early diagnosis",
                    "Multi-omics approaches reveal novel therapeutic targets"
                ],
                "citations": 47,
                "downloads": 234
            },
            {
                "id": "brief-002",
                "title": "Drug Repurposing Opportunities in Autoimmune Disease",
                "author": "AI Research Consortium",
                "date_generated": "2024-01-10",
                "pages": 23,
                "sections": ["Computational Methods", "Clinical Evidence", "Regulatory Pathways", "Case Studies"],
                "key_insights": [
                    "783 existing drugs show potential for autoimmune indications",
                    "Network-based approaches identify unexpected therapeutic targets",
                    "Regulatory fast-track pathways available for 15 compounds"
                ],
                "citations": 63,
                "downloads": 412
            }
        ]
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        house_indicator = " [HOUSE MODE]" if self.house_mode else ""
        
        with Container(classes="research-header"):
            yield Static(
                f"[bold]Research Hub[/bold]{house_indicator} | [h] Toggle House Mode [g] Generate Brief "
                "[1] Suggestions [2] Trials [3] Briefs [ESC] Back"
            )
        
        with Horizontal(classes="search-row"):
            yield Input(placeholder="Search research...", id="search-input")
            yield Button("Search", id="btn-search", variant="primary")
            yield Static("  House Mode:", classes="house-label")
            yield Switch(value=self.house_mode, id="house-switch")
        
        with TabbedContent(id="research-tabs"):
            with TabPane("Research Suggestions", id="tab-suggestions"):
                with ScrollableContainer():
                    yield Static(
                        f"[bold]AI-Curated Research Suggestions[/bold]\n"
                        f"[dim]{'Expanded search including rare diseases and analogical reasoning' if self.house_mode else 'Standard research suggestions based on current case patterns'}[/dim]",
                        classes="section-header"
                    )
                    
                    for suggestion in self.research_suggestions:
                        yield ResearchSuggestionCard(
                            title=suggestion["title"],
                            category=suggestion["category"],
                            confidence=suggestion["confidence"],
                            description=suggestion["description"],
                            suggested_actions=suggestion["suggested_actions"],
                            similar_cases=suggestion["similar_cases"],
                            literature=suggestion["literature"],
                            priority=suggestion["priority"]
                        )
                        with Horizontal(classes="action-buttons"):
                            yield Button("Explore Research", variant="primary")
                            yield Button("View Literature", variant="default")
                            yield Button("Generate Brief", variant="default")
            
            with TabPane("Clinical Trials", id="tab-trials"):
                with ScrollableContainer():
                    yield Static(
                        "[bold]Clinical Trial Matching[/bold]\n"
                        "[dim]Relevant clinical trials based on patient characteristics and current case[/dim]",
                        classes="section-header"
                    )
                    
                    for trial in self.clinical_trials:
                        yield ClinicalTrialCard(
                            trial_id=trial["id"],
                            title=trial["title"],
                            phase=trial["phase"],
                            status=trial["status"],
                            location=trial["location"],
                            match_score=trial["match_score"],
                            eligibility=trial["eligibility"],
                            endpoint=trial["endpoint"],
                            contact=trial["contact"],
                            distance=trial["distance"]
                        )
                        with Horizontal(classes="action-buttons"):
                            yield Button("Check Eligibility", variant="primary")
                            yield Button("ClinicalTrials.gov", variant="default")
                            yield Button("Contact PI", variant="default")
            
            with TabPane("Research Briefs", id="tab-briefs"):
                with ScrollableContainer():
                    with Horizontal():
                        yield Static(
                            "[bold]Auto-Generated Research Briefs[/bold]\n"
                            "[dim]Comprehensive reports on relevant medical topics[/dim]",
                            classes="section-header"
                        )
                        yield Button("Generate New Brief", id="btn-new-brief", variant="primary")
                    
                    for brief in self.research_briefs:
                        yield ResearchBriefCard(
                            title=brief["title"],
                            author=brief["author"],
                            date_generated=brief["date_generated"],
                            pages=brief["pages"],
                            sections=brief["sections"],
                            key_insights=brief["key_insights"],
                            citations=brief["citations"],
                            downloads=brief["downloads"]
                        )
                        with Horizontal(classes="action-buttons"):
                            yield Button("Preview", variant="default")
                            yield Button("Download PDF", variant="primary")
        
        yield Footer()
    
    def action_go_back(self) -> None:
        """Go back to previous screen."""
        self.app.pop_screen()
    
    def action_goto_dashboard(self) -> None:
        """Navigate to dashboard."""
        self.app.switch_screen("dashboard")
    
    def action_toggle_house_mode(self) -> None:
        """Toggle house mode for expanded search."""
        self.house_mode = not self.house_mode
        switch = self.query_one("#house-switch", Switch)
        switch.value = self.house_mode
        
        mode_str = "enabled" if self.house_mode else "disabled"
        self.notify(f"House Mode {mode_str}", severity="information")
        
        self.store.audit.log_event(
            event_type="research",
            user="current_user",
            action="toggle_house_mode",
            details={"enabled": self.house_mode}
        )
    
    def action_generate_brief(self) -> None:
        """Generate a new research brief."""
        self.notify("Generating research brief... This may take a moment.", severity="information")
        
        self.store.audit.log_event(
            event_type="research",
            user="current_user",
            action="generate_brief",
            details={"case_id": self.case_id}
        )
    
    def action_tab_suggestions(self) -> None:
        """Switch to suggestions tab."""
        tabs = self.query_one("#research-tabs", TabbedContent)
        tabs.active = "tab-suggestions"
    
    def action_tab_trials(self) -> None:
        """Switch to trials tab."""
        tabs = self.query_one("#research-tabs", TabbedContent)
        tabs.active = "tab-trials"
    
    def action_tab_briefs(self) -> None:
        """Switch to briefs tab."""
        tabs = self.query_one("#research-tabs", TabbedContent)
        tabs.active = "tab-briefs"
    
    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch changes."""
        if event.switch.id == "house-switch":
            self.house_mode = event.value
            mode_str = "enabled" if self.house_mode else "disabled"
            self.notify(f"House Mode {mode_str}", severity="information")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-search":
            search_input = self.query_one("#search-input", Input)
            query = search_input.value.strip()
            if query:
                self.notify(f"Searching for: {query}", severity="information")
        elif button_id == "btn-new-brief":
            self.action_generate_brief()
