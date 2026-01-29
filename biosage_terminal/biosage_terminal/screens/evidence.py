"""
Evidence Explorer screen for BioSage Terminal.
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Footer, Input, Tree
from textual.binding import Binding
from rich.text import Text

from biosage_terminal.storage import get_data_store
from biosage_terminal.theme import SPECIALTY_ICONS, SPECIALTY_COLORS


class EvidenceExplorerScreen(Screen):
    """Evidence explorer with tree view and search."""
    
    BINDINGS = [
        Binding("escape", "go_back", "Back"),
        Binding("d", "goto_dashboard", "Dashboard"),
        Binding("q", "quit", "Quit"),
    ]
    
    CSS = """
    EvidenceExplorerScreen {
        background: #0F172A;
    }
    
    .evidence-header {
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
    }
    
    .search-input {
        width: 60%;
    }
    
    .main-content {
        layout: horizontal;
        height: 1fr;
    }
    
    .tree-panel {
        width: 40%;
        border-right: solid #334155;
        padding: 1;
    }
    
    .details-panel {
        width: 60%;
        padding: 1;
    }
    
    .entity-section {
        margin-bottom: 2;
    }
    
    .section-title {
        text-style: bold;
        color: #F1F5F9;
        margin-bottom: 1;
    }
    
    .detail-card {
        background: #1E293B;
        border: solid #334155;
        padding: 1;
        margin: 1 0;
    }
    
    .citation-card {
        background: #0F172A;
        border: solid #334155;
        padding: 1;
        margin: 1 0;
    }
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selected_entity = None
        self.knowledge_data = self._load_knowledge_data()
    
    def _load_knowledge_data(self) -> dict:
        """Load knowledge base data."""
        store = get_data_store()
        
        # Build knowledge structure from cases and evidence
        knowledge = {
            "diseases": [],
            "symptoms": [],
            "treatments": [],
            "tests": [],
        }
        
        all_cases = store.cases.list_all_sync()
        seen_diseases = set()
        seen_symptoms = set()
        seen_treatments = set()
        seen_tests = set()
        
        for case in all_cases:
            # Collect diseases from diagnoses
            for diag in case.get("diagnoses", []):
                disease = diag.get("diagnosis", "")
                if disease and disease not in seen_diseases:
                    seen_diseases.add(disease)
                    knowledge["diseases"].append({
                        "name": disease,
                        "confidence": diag.get("confidence", 0),
                        "specialist": diag.get("specialist", ""),
                    })
            
            # Collect symptoms
            for symptom in case.get("symptoms", []):
                if symptom and symptom not in seen_symptoms:
                    seen_symptoms.add(symptom)
                    knowledge["symptoms"].append({"name": symptom})
            
            # Collect tests
            for test in case.get("test_recommendations", []):
                test_name = test.get("test_name", "")
                if test_name and test_name not in seen_tests:
                    seen_tests.add(test_name)
                    knowledge["tests"].append({
                        "name": test_name,
                        "rationale": test.get("rationale", ""),
                    })
            
            # Collect treatments
            for rec in case.get("treatment_recommendations", []):
                treatment = rec.get("recommendation", "")
                if treatment and treatment not in seen_treatments:
                    seen_treatments.add(treatment)
                    knowledge["treatments"].append({
                        "name": treatment,
                        "rationale": rec.get("rationale", ""),
                    })
        
        return knowledge
    
    def compose(self) -> ComposeResult:
        yield Static(self._render_header(), classes="evidence-header")
        
        with Container(classes="search-bar"):
            yield Input(
                placeholder="Search entities...",
                id="search_input",
                classes="search-input",
            )
        
        with Horizontal(classes="main-content"):
            with ScrollableContainer(classes="tree-panel"):
                yield Static(self._render_tree_view(), id="tree_view")
            
            with ScrollableContainer(classes="details-panel"):
                yield Static(self._render_details(), id="details_view")
        
        yield Footer()
    
    def _render_header(self) -> Text:
        """Render the header."""
        text = Text()
        text.append("BioSage", style="bold #3B82F6")
        text.append(" | ", style="#334155")
        text.append("Evidence Explorer", style="bold #F1F5F9")
        return text
    
    def _render_tree_view(self) -> Text:
        """Render the tree view of entities."""
        text = Text()
        
        # Diseases
        text.append("[-] Diseases", style="bold #F1F5F9")
        text.append(f" ({len(self.knowledge_data['diseases'])})\n", style="#64748B")
        for disease in self.knowledge_data["diseases"][:10]:
            name = disease["name"]
            specialist = disease.get("specialist", "")
            icon = SPECIALTY_ICONS.get(specialist, "")
            color = SPECIALTY_COLORS.get(specialist, "#94A3B8")
            text.append(f"    |-- {icon} ", style=color)
            text.append(f"{name}\n", style="#F1F5F9")
        if len(self.knowledge_data["diseases"]) > 10:
            text.append(f"    `-- ... and {len(self.knowledge_data['diseases']) - 10} more\n", style="#64748B")
        
        text.append("\n", style="")
        
        # Symptoms
        text.append("[-] Symptoms", style="bold #F1F5F9")
        text.append(f" ({len(self.knowledge_data['symptoms'])})\n", style="#64748B")
        for symptom in self.knowledge_data["symptoms"][:10]:
            text.append(f"    |-- {symptom['name']}\n", style="#94A3B8")
        if len(self.knowledge_data["symptoms"]) > 10:
            text.append(f"    `-- ... and {len(self.knowledge_data['symptoms']) - 10} more\n", style="#64748B")
        
        text.append("\n", style="")
        
        # Tests
        text.append("[-] Diagnostic Tests", style="bold #F1F5F9")
        text.append(f" ({len(self.knowledge_data['tests'])})\n", style="#64748B")
        for test in self.knowledge_data["tests"][:10]:
            text.append(f"    |-- {test['name']}\n", style="#06B6D4")
        if len(self.knowledge_data["tests"]) > 10:
            text.append(f"    `-- ... and {len(self.knowledge_data['tests']) - 10} more\n", style="#64748B")
        
        text.append("\n", style="")
        
        # Treatments
        text.append("[-] Treatments", style="bold #F1F5F9")
        text.append(f" ({len(self.knowledge_data['treatments'])})\n", style="#64748B")
        for treatment in self.knowledge_data["treatments"][:10]:
            name = treatment["name"][:40] + "..." if len(treatment["name"]) > 40 else treatment["name"]
            text.append(f"    |-- {name}\n", style="#22C55E")
        if len(self.knowledge_data["treatments"]) > 10:
            text.append(f"    `-- ... and {len(self.knowledge_data['treatments']) - 10} more\n", style="#64748B")
        
        if not any(self.knowledge_data.values()):
            text.append("\nNo evidence data yet.\n", style="#64748B")
            text.append("Evidence will be collected as cases are diagnosed.\n", style="#94A3B8")
        
        return text
    
    def _render_details(self) -> Text:
        """Render the details panel."""
        text = Text()
        
        text.append("=== Knowledge Base Details ===\n\n", style="bold #F1F5F9")
        
        # Statistics
        text.append("Statistics\n", style="bold #3B82F6")
        text.append("-" * 30 + "\n", style="#334155")
        text.append(f"Diseases: {len(self.knowledge_data['diseases'])}\n", style="#F1F5F9")
        text.append(f"Symptoms: {len(self.knowledge_data['symptoms'])}\n", style="#F1F5F9")
        text.append(f"Tests: {len(self.knowledge_data['tests'])}\n", style="#F1F5F9")
        text.append(f"Treatments: {len(self.knowledge_data['treatments'])}\n", style="#F1F5F9")
        
        text.append("\n", style="")
        
        # Top diseases by specialty
        text.append("Diseases by Specialty\n", style="bold #3B82F6")
        text.append("-" * 30 + "\n", style="#334155")
        
        by_specialty = {}
        for disease in self.knowledge_data["diseases"]:
            spec = disease.get("specialist", "unknown")
            if spec not in by_specialty:
                by_specialty[spec] = []
            by_specialty[spec].append(disease)
        
        for spec, diseases in by_specialty.items():
            icon = SPECIALTY_ICONS.get(spec, "[?]")
            color = SPECIALTY_COLORS.get(spec, "#94A3B8")
            text.append(f"\n{icon} {spec.title() if spec else 'Unknown'}\n", style=f"bold {color}")
            for d in diseases[:3]:
                conf = d.get("confidence", 0)
                text.append(f"  - {d['name']} ({int(conf * 100)}%)\n", style="#94A3B8")
        
        text.append("\n", style="")
        
        # Recent evidence
        text.append("Recent Citations\n", style="bold #3B82F6")
        text.append("-" * 30 + "\n", style="#334155")
        
        store = get_data_store()
        evidence_list = store.evidence.list_all_sync()
        
        if not evidence_list:
            text.append("No citations collected yet.\n", style="#64748B")
        else:
            for ev in evidence_list[:5]:
                title = ev.get("title", "Untitled")
                score = ev.get("relevance_score", 0)
                text.append(f"\n[EV] {title}\n", style="bold #06B6D4")
                text.append(f"     Relevance: {int(score * 100)}%\n", style="#94A3B8")
        
        return text
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes."""
        query = event.value.lower()
        # Filter and update tree view based on search
        # For simplicity, just refresh the view
        # In a full implementation, would filter the tree
        pass
    
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
