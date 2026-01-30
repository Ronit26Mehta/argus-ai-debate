"""
Help Screen - Documentation and keyboard shortcuts for BioSage Terminal.

Features:
- Navigation sidebar with topic buttons
- Scrollable documentation content
- Keyboard shortcuts reference
- ARGUS framework explanation
- LLM provider configuration guide
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Button, Footer
from textual.binding import Binding
from rich.text import Text


class HelpScreen(Screen):
    """Help and documentation screen for BioSage Terminal."""
    
    BINDINGS = [
        Binding("escape", "go_back", "Back"),
        Binding("q", "quit", "Quit"),
        Binding("d", "goto_dashboard", "Dashboard"),
        Binding("1", "show_intro", "Intro"),
        Binding("2", "show_keyboard", "Keys"),
        Binding("3", "show_diagnosis", "Diagnosis"),
        Binding("4", "show_specialists", "Specialists"),
        Binding("5", "show_llm", "LLM Setup"),
    ]
    
    CSS = """
    HelpScreen {
        background: #0F172A;
    }
    
    .help-header {
        dock: top;
        height: 5;
        background: #1E40AF;
        padding: 1 2;
        border-bottom: tall #3B82F6;
        content-align: center middle;
    }
    
    .help-main {
        height: 1fr;
        layout: horizontal;
    }
    
    .help-sidebar {
        width: 28;
        height: 100%;
        background: #1E293B;
        border-right: solid #334155;
        padding: 1;
    }
    
    .sidebar-title {
        text-style: bold;
        color: #3B82F6;
        margin-bottom: 1;
        text-align: center;
    }
    
    .nav-btn {
        width: 100%;
        height: 3;
        margin-bottom: 1;
        background: #0F172A;
        border: solid #334155;
    }
    
    .nav-btn:hover {
        background: #334155;
        border: solid #3B82F6;
    }
    
    .nav-btn:focus {
        background: #1E40AF;
        border: solid #60A5FA;
    }
    
    .nav-btn.-active {
        background: #1E40AF;
        border: solid #3B82F6;
    }
    
    .help-content-area {
        width: 1fr;
        height: 100%;
        padding: 1 2;
    }
    
    .content-scroll {
        height: 100%;
        border: solid #334155;
        background: #1E293B;
    }
    
    .content-inner {
        padding: 2;
    }
    
    .help-footer-hint {
        dock: bottom;
        height: 3;
        background: #1E293B;
        border-top: solid #334155;
        padding: 1 2;
        color: #94A3B8;
    }
    """
    
    HELP_INTRO = """[bold #3B82F6]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]
[bold #60A5FA]🏥 WELCOME TO BIOSAGE TERMINAL[/]
[bold #3B82F6]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]

[bold #F1F5F9]What is BioSage?[/]
BioSage Terminal is an AI-Powered Medical Diagnostic Assistant that uses the
ARGUS debate-based reasoning framework for evidence-based diagnostics.

[bold #F1F5F9]Key Features:[/]
  [#22C55E]●[/] [bold]Multi-Agent AI Debate[/] - 6 specialist agents + moderator + jury
  [#22C55E]●[/] [bold]Evidence-Based Reasoning[/] - Bayesian posterior computation
  [#22C55E]●[/] [bold]Patient Management[/] - Complete patient and case tracking
  [#22C55E]●[/] [bold]Local Storage[/] - All data stored securely on your machine
  [#22C55E]●[/] [bold]Full Audit Trail[/] - Track all diagnostic decisions

[bold #3B82F6]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]
[bold #60A5FA]QUICK START GUIDE[/]
[bold #3B82F6]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]

[bold #EAB308]Step 1:[/] Create a New Patient
   Press [bold #60A5FA][o][/] or click "New Patient" to start patient onboarding

[bold #EAB308]Step 2:[/] Enter Patient Information
   Fill in demographics, chief complaint, symptoms, and vitals

[bold #EAB308]Step 3:[/] Run AI Diagnosis
   Press [bold #60A5FA][r][/] to start the ARGUS debate framework

[bold #EAB308]Step 4:[/] Review Results
   View differential diagnoses with confidence scores and evidence

[bold #EAB308]Step 5:[/] Order Tests (Optional)
   Request additional tests based on AI recommendations

[bold #3B82F6]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]
[bold #60A5FA]MAIN SCREENS[/]
[bold #3B82F6]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]

  [bold #3B82F6][d][/] [bold]Dashboard[/]      - Overview of cases, stats, and quick actions
  [bold #3B82F6][o][/] [bold]New Patient[/]    - Patient onboarding and case creation
  [bold #3B82F6][l][/] [bold]Cases Manager[/]  - View and manage all diagnostic cases
  [bold #3B82F6][s][/] [bold]Specialists[/]    - View AI specialist agents
  [bold #3B82F6][e][/] [bold]Evidence[/]       - Browse evidence library
  [bold #3B82F6][m][/] [bold]Model Ops[/]      - LLM provider configuration
  [bold #3B82F6][a][/] [bold]Audit[/]          - View audit trail and logs
  [bold #3B82F6][c][/] [bold]Collaboration[/]  - Team collaboration room
  [bold #3B82F6][r][/] [bold]Research Hub[/]   - Medical research integration
  [bold #3B82F6][v][/] [bold]Visual Dx[/]      - Visual diagnosis tools

[#94A3B8]Tip: Press [bold]?[/] from any screen to open this help page[/]"""

    HELP_KEYBOARD = """[bold #3B82F6]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]
[bold #60A5FA]⌨️  KEYBOARD SHORTCUTS[/]
[bold #3B82F6]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]

[bold #F1F5F9]GLOBAL NAVIGATION[/]
[#334155]─────────────────────────────────────────────────────────────────────────────[/]
  [bold #60A5FA]q[/]           Quit application
  [bold #60A5FA]Escape[/]      Go back / Cancel
  [bold #60A5FA]?[/]           Open help screen
  [bold #60A5FA]d[/]           Go to Dashboard
  [bold #60A5FA]o[/]           New Patient / Onboarding
  [bold #60A5FA]l[/]           Cases Manager

[bold #F1F5F9]SCREEN NAVIGATION[/]
[#334155]─────────────────────────────────────────────────────────────────────────────[/]
  [bold #60A5FA]s[/]           Specialists Grid
  [bold #60A5FA]e[/]           Evidence Explorer
  [bold #60A5FA]m[/]           Model Operations
  [bold #60A5FA]a[/]           Admin Audit
  [bold #60A5FA]c[/]           Collaboration Room
  [bold #60A5FA]r[/]           Research Hub
  [bold #60A5FA]v[/]           Visual Diagnosis

[bold #F1F5F9]CASES MANAGER[/]
[#334155]─────────────────────────────────────────────────────────────────────────────[/]
  [bold #60A5FA]n[/]           Create new case
  [bold #60A5FA]Enter[/]       Open selected case
  [bold #60A5FA]r[/]           Run diagnosis on selected case
  [bold #60A5FA]t[/]           Order tests for selected case
  [bold #60A5FA]1[/]           Filter: All cases
  [bold #60A5FA]2[/]           Filter: Open cases
  [bold #60A5FA]3[/]           Filter: Diagnosed cases
  [bold #60A5FA]f[/]           Focus search/filter

[bold #F1F5F9]GENERAL CONTROLS[/]
[#334155]─────────────────────────────────────────────────────────────────────────────[/]
  [bold #60A5FA]Tab[/]         Move to next element
  [bold #60A5FA]Shift+Tab[/]   Move to previous element
  [bold #60A5FA]Enter[/]       Activate button / Confirm
  [bold #60A5FA]Space[/]       Toggle switches / Select
  [bold #60A5FA]↑ ↓[/]         Navigate lists and tables
  [bold #60A5FA]← →[/]         Navigate tabs / Expand trees

[#94A3B8]Tip: Most screens show available shortcuts in the footer bar[/]"""

    HELP_DIAGNOSIS = """[bold #3B82F6]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]
[bold #60A5FA]🔬 THE ARGUS DIAGNOSIS FRAMEWORK[/]
[bold #3B82F6]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]

[bold #F1F5F9]What is ARGUS?[/]
ARGUS (Agentic Research & Governance Unified System) is a multi-agent debate
framework that evaluates diagnostic hypotheses through structured argumentation.

[bold #3B82F6]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]
[bold #60A5FA]HOW THE DEBATE WORKS[/]
[bold #3B82F6]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]

[bold #A855F7]1. MODERATOR AGENT[/]
   [#94A3B8]Orchestrates the debate, creates agenda, and manages rounds[/]
   • Analyzes patient symptoms and history
   • Creates debate proposition (diagnostic hypothesis)
   • Controls flow between specialist agents

[bold #3B82F6]2. SPECIALIST AGENTS (6 Domains)[/]
   [#94A3B8]Each specialist gathers domain-specific evidence[/]
   • [#EF4444]Infectious Disease[/] - Bacterial, viral, parasitic
   • [#F97316]Cardiology[/] - Heart and vascular conditions
   • [#A855F7]Neurology[/] - Brain and nervous system
   • [#EC4899]Oncology[/] - Cancer and tumors
   • [#EAB308]Autoimmune[/] - Immune system disorders
   • [#22C55E]Toxicology[/] - Poisoning and drug effects

[bold #EF4444]3. REFUTER AGENT[/]
   [#94A3B8]Challenges weak evidence and generates rebuttals[/]
   • Identifies logical fallacies
   • Proposes alternative explanations
   • Strengthens diagnostic reasoning

[bold #22C55E]4. JURY AGENT[/]
   [#94A3B8]Renders final verdict via Bayesian aggregation[/]
   • Weighs all evidence and rebuttals
   • Computes posterior probabilities
   • Generates verdict: SUPPORTED / REJECTED / UNDECIDED

[bold #3B82F6]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]
[bold #60A5FA]THE C-DAG (Conceptual Debate Graph)[/]
[bold #3B82F6]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]

The C-DAG visualizes the debate structure:
  [#EAB308]◆[/] [bold]Proposition[/]  - The diagnosis being evaluated
  [#22C55E]●[/] [bold]Supporting[/]   - Evidence favoring the diagnosis
  [#EF4444]●[/] [bold]Attacking[/]    - Evidence against the diagnosis
  [#A855F7]◇[/] [bold]Rebuttals[/]    - Challenges to evidence

[bold #F1F5F9]Confidence Scores:[/]
  • Computed using Bayesian posterior probability
  • Calibrated against prior medical knowledge
  • Higher confidence = stronger evidence support

[#94A3B8]All debates are recorded in the audit trail for transparency[/]"""

    HELP_SPECIALISTS = """[bold #3B82F6]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]
[bold #60A5FA]👨‍⚕️ AI SPECIALIST AGENTS[/]
[bold #3B82F6]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]

BioSage uses 6 specialized AI agents, each focused on a medical domain:

[bold #EF4444]━━━ INFECTIOUS DISEASE ━━━[/]
[#94A3B8]Expertise: Bacterial, viral, fungal, and parasitic infections[/]
• Analyzes infection patterns and symptoms
• Considers epidemiological factors
• Evaluates antimicrobial resistance

[bold #F97316]━━━ CARDIOLOGY ━━━[/]
[#94A3B8]Expertise: Heart and cardiovascular conditions[/]
• Evaluates cardiac symptoms (chest pain, palpitations)
• Analyzes ECG patterns and biomarkers
• Assesses cardiovascular risk factors

[bold #A855F7]━━━ NEUROLOGY ━━━[/]
[#94A3B8]Expertise: Brain, spinal cord, and nervous system[/]
• Evaluates neurological symptoms
• Considers stroke, seizure, headache patterns
• Analyzes cognitive and motor function

[bold #EC4899]━━━ ONCOLOGY ━━━[/]
[#94A3B8]Expertise: Cancer detection and tumor analysis[/]
• Identifies potential malignancy indicators
• Evaluates tumor markers and imaging findings
• Considers paraneoplastic syndromes

[bold #EAB308]━━━ AUTOIMMUNE ━━━[/]
[#94A3B8]Expertise: Immune system disorders[/]
• Analyzes autoantibody patterns
• Evaluates systemic inflammatory conditions
• Considers rheumatologic manifestations

[bold #22C55E]━━━ TOXICOLOGY ━━━[/]
[#94A3B8]Expertise: Poisoning, overdose, and drug effects[/]
• Identifies toxidromes and exposure patterns
• Evaluates drug interactions
• Considers environmental and occupational exposures

[bold #3B82F6]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]
[bold #60A5FA]HOW SPECIALISTS COLLABORATE[/]
[bold #3B82F6]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]

Each specialist independently evaluates the patient case and provides:
  • Domain-specific evidence (supporting or attacking)
  • Confidence weight based on symptom relevance
  • Key observations from their specialty perspective

The Jury Agent then aggregates all specialist inputs using Bayesian
inference to produce a balanced, multi-perspective diagnosis.

[#94A3B8]Press [bold]s[/] from any screen to view the Specialists Grid[/]"""

    HELP_LLM = """[bold #3B82F6]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]
[bold #60A5FA]🤖 LLM PROVIDER CONFIGURATION[/]
[bold #3B82F6]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]

BioSage supports multiple LLM providers. It auto-detects available providers
in this priority order:

[bold #F1F5F9]PROVIDER PRIORITY[/]
[#334155]─────────────────────────────────────────────────────────────────────────────[/]
  [bold #22C55E]1.[/] [bold]Google Gemini[/]      (Recommended)
  [bold #60A5FA]2.[/] [bold]OpenAI[/]            GPT-4, GPT-4o
  [bold #A855F7]3.[/] [bold]Anthropic[/]         Claude 3.5
  [bold #F97316]4.[/] [bold]Groq[/]              Llama 3.1
  [bold #EAB308]5.[/] [bold]Mistral[/]           Mistral Large
  [bold #EC4899]6.[/] [bold]Cohere[/]            Command R+
  [bold #94A3B8]7.[/] [bold]Ollama[/]            Local models

[bold #3B82F6]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]
[bold #60A5FA]ENVIRONMENT VARIABLES[/]
[bold #3B82F6]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]

Set your API keys as environment variables:

[bold #F1F5F9]API Keys:[/]
  [#60A5FA]GEMINI_API_KEY[/]       Google Gemini API key
  [#60A5FA]OPENAI_API_KEY[/]       OpenAI API key
  [#60A5FA]ANTHROPIC_API_KEY[/]    Anthropic API key
  [#60A5FA]GROQ_API_KEY[/]         Groq API key
  [#60A5FA]MISTRAL_API_KEY[/]      Mistral API key
  [#60A5FA]COHERE_API_KEY[/]       Cohere API key

[bold #F1F5F9]Model Names (Optional):[/]
  [#60A5FA]GEMINI_MODEL[/]         Default: gemini-1.5-pro
  [#60A5FA]OPENAI_MODEL[/]         Default: gpt-4o
  [#60A5FA]ANTHROPIC_MODEL[/]      Default: claude-3-5-sonnet-20241022
  [#60A5FA]GROQ_MODEL[/]           Default: llama-3.1-70b-versatile
  [#60A5FA]MISTRAL_MODEL[/]        Default: mistral-large-latest
  [#60A5FA]COHERE_MODEL[/]         Default: command-r-plus
  [#60A5FA]OLLAMA_MODEL[/]         Default: llama3.1

[bold #F1F5F9]Data Storage:[/]
  [#60A5FA]BIOSAGE_DATA_DIR[/]     Default: ~/.biosage

[bold #3B82F6]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]
[bold #60A5FA]CHECKING CONFIGURATION[/]
[bold #3B82F6]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]

Run this command to check your API configuration:

  [bold #22C55E]biosage --check-api[/]

This will show which providers are available and which is being used.

[bold #F1F5F9]Using a .env File:[/]
You can also create a [bold].env[/] file in your working directory:

  [#60A5FA]GEMINI_API_KEY=your_key_here
  GEMINI_MODEL=gemini-1.5-pro[/]

[#94A3B8]Press [bold]m[/] to open Model Operations for provider status[/]"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_section = "intro"
    
    def compose(self) -> ComposeResult:
        yield Static(self._render_header(), classes="help-header")
        
        with Horizontal(classes="help-main"):
            # Sidebar navigation
            with Vertical(classes="help-sidebar"):
                yield Static("[bold #3B82F6]📚 Help Topics[/]", classes="sidebar-title")
                yield Button("Introduction [1]", id="btn_intro", classes="nav-btn -active")
                yield Button("Keyboard [2]", id="btn_keyboard", classes="nav-btn")
                yield Button("Diagnosis [3]", id="btn_diagnosis", classes="nav-btn")
                yield Button("Specialists [4]", id="btn_specialists", classes="nav-btn")
                yield Button("LLM Setup [5]", id="btn_llm", classes="nav-btn")
            
            # Main content area
            with Container(classes="help-content-area"):
                with ScrollableContainer(classes="content-scroll"):
                    yield Static(self.HELP_INTRO, id="help_content", classes="content-inner")
        
        yield Static(
            "[#60A5FA]Navigation:[/] [1-5] Topics  [Esc] Back  [q] Quit  [d] Dashboard",
            classes="help-footer-hint"
        )
        yield Footer()
    
    def _render_header(self) -> Text:
        """Render the help header."""
        text = Text()
        text.append("\n", style="")
        text.append("📖 BioSage Help", style="bold #FFFFFF")
        text.append("  │  ", style="bold #60A5FA")
        text.append("Documentation & Keyboard Shortcuts", style="#93C5FD")
        text.append("\n", style="")
        return text
    
    def _update_content(self, section: str) -> None:
        """Update the help content based on selected section."""
        content_map = {
            "intro": self.HELP_INTRO,
            "keyboard": self.HELP_KEYBOARD,
            "diagnosis": self.HELP_DIAGNOSIS,
            "specialists": self.HELP_SPECIALISTS,
            "llm": self.HELP_LLM,
        }
        
        content = content_map.get(section, self.HELP_INTRO)
        self.query_one("#help_content", Static).update(content)
        self.current_section = section
        
        # Update button states
        button_map = {
            "intro": "btn_intro",
            "keyboard": "btn_keyboard",
            "diagnosis": "btn_diagnosis",
            "specialists": "btn_specialists",
            "llm": "btn_llm",
        }
        
        for sec, btn_id in button_map.items():
            btn = self.query_one(f"#{btn_id}", Button)
            if sec == section:
                btn.add_class("-active")
            else:
                btn.remove_class("-active")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle sidebar button presses."""
        button_id = event.button.id
        section_map = {
            "btn_intro": "intro",
            "btn_keyboard": "keyboard",
            "btn_diagnosis": "diagnosis",
            "btn_specialists": "specialists",
            "btn_llm": "llm",
        }
        
        if button_id in section_map:
            self._update_content(section_map[button_id])
    
    def action_show_intro(self) -> None:
        """Show introduction section."""
        self._update_content("intro")
    
    def action_show_keyboard(self) -> None:
        """Show keyboard shortcuts section."""
        self._update_content("keyboard")
    
    def action_show_diagnosis(self) -> None:
        """Show diagnosis framework section."""
        self._update_content("diagnosis")
    
    def action_show_specialists(self) -> None:
        """Show specialists section."""
        self._update_content("specialists")
    
    def action_show_llm(self) -> None:
        """Show LLM setup section."""
        self._update_content("llm")
    
    def action_go_back(self) -> None:
        """Go back to previous screen."""
        self.app.pop_screen()
    
    def action_goto_dashboard(self) -> None:
        """Navigate to dashboard."""
        self.app.goto_dashboard()
