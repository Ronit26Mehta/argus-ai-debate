"""
Help Screen - Documentation and keyboard shortcuts.

Features:
- Navigation sidebar
- Markdown documentation
- Keyboard shortcuts reference
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Button


class HelpScreen(Screen):
    """Help and documentation screen."""
    
    DEFAULT_CSS = """
    HelpScreen {
        background: #0a0a0a;
    }
    
    HelpScreen #help-scroll {
        width: 100%;
        height: 100%;
    }
    
    HelpScreen #help-container {
        width: 100%;
        height: auto;
        padding: 1;
    }
    
    HelpScreen #header-bar {
        width: 100%;
        height: 3;
        background: #0d0d0d;
        border: solid #665200;
        padding: 0 1;
        margin-bottom: 1;
    }
    
    HelpScreen #header-title {
        width: 100%;
        height: 1;
        color: #ffcc33;
        text-style: bold;
        margin-top: 1;
    }
    
    HelpScreen #content-area {
        width: 100%;
        height: auto;
        min-height: 40;
        layout: horizontal;
    }
    
    HelpScreen #nav-sidebar {
        width: 25%;
        height: auto;
        background: #0d0d0d;
        border: solid #665200;
        padding: 1;
        margin-right: 1;
    }
    
    HelpScreen #help-content {
        width: 75%;
        height: auto;
        background: #0d0d0d;
        border: solid #665200;
        padding: 1;
    }
    
    HelpScreen .section-title {
        width: 100%;
        height: 1;
        color: #ffcc33;
        text-style: bold;
        margin-bottom: 1;
    }
    
    HelpScreen .nav-btn {
        width: 100%;
        height: 3;
        margin-bottom: 1;
        background: #0a0a0a;
        border: solid #4d3800;
    }
    
    HelpScreen .nav-btn:hover {
        background: #1a1400;
    }
    
    HelpScreen .nav-btn:focus {
        border: solid #ffb000;
    }
    
    HelpScreen #doc-scroll {
        width: 100%;
        height: auto;
        min-height: 20;
    }
    
    HelpScreen #doc-content {
        width: 100%;
        height: auto;
        padding: 1;
    }
    
    HelpScreen #nav-hint {
        width: 100%;
        height: 1;
        color: #806000;
        margin-top: 1;
    }
    """
    
    HELP_INTRO = """[bold #ffcc33][*] WELCOME TO ARGUS TERMINAL[/]

[bold #ff8c00]What is ARGUS?[/]
ARGUS (Agentic Research & Governance Unified System) is a multi-agent AI debate 
framework that evaluates claims through structured argumentation.

[bold #ff8c00]Quick Start:[/]
1. Press [bold]2[/] to go to the Debate screen
2. Enter a proposition (claim to evaluate)
3. Click START DEBATE
4. Watch the agents analyze and debate
5. View the verdict and C-DAG graph

[bold #ff8c00]Screens:[/]
- [bold]1 - Main[/]: Dashboard with quick stats and actions
- [bold]2 - Debate[/]: Run AI debates on any claim
- [bold]3 - Providers[/]: Configure LLM providers (27+)
- [bold]4 - Tools[/]: Browse and use integrated tools (19)
- [bold]5 - Knowledge[/]: Connectors and document retrieval
- [bold]6 - Benchmark[/]: Evaluate models on standard datasets
- [bold]7 - Settings[/]: Theme and configuration options
- [bold]8 - Help[/]: This documentation screen

[dim]-----------------------------------------------------------------[/dim]
[#33ff33]Tip: Use Tab to focus elements, Enter to activate buttons[/]"""

    HELP_KEYBOARD = """[bold #ffcc33][*] KEYBOARD SHORTCUTS[/]

[bold #ff8c00]Navigation[/]
-----------------------------------------
[bold]1[/]         Go to Main Dashboard
[bold]2[/]         Go to Debate System
[bold]3[/]         Go to Providers
[bold]4[/]         Go to Tools
[bold]5[/]         Go to Knowledge
[bold]6[/]         Go to Benchmarks
[bold]7[/]         Go to Settings
[bold]8[/]         Go to Help (this screen)
[bold]q[/]         Quit application
[bold]Escape[/]    Return to Main

[bold #ff8c00]General[/]
-----------------------------------------
[bold]Tab[/]       Move focus to next element
[bold]Shift+Tab[/] Move focus to previous element
[bold]Enter[/]     Activate focused button/input
[bold]Space[/]     Toggle switches/checkboxes
[bold]Arrow UD[/]  Navigate lists and trees
[bold]Arrow LR[/]  Expand/collapse tree nodes

[bold #ff8c00]Input Fields[/]
-----------------------------------------
Click or Tab into input, type to edit
Press Enter to confirm most inputs"""

    HELP_DEBATE = """[bold #ffcc33][*] DEBATE SYSTEM[/]

[bold #ff8c00]How Debates Work[/]
-----------------------------------------
ARGUS uses multiple AI agents to evaluate claims:

[bold #b833ff]1. Moderator[/]
   Orchestrates the debate, manages rounds

[bold #b833ff]2. Specialist[/]
   Gathers evidence supporting the claim

[bold #b833ff]3. Refuter[/]
   Generates challenges and counter-evidence

[bold #b833ff]4. Jury[/]
   Evaluates all evidence and renders verdict

[bold #ff8c00]The C-DAG[/]
-----------------------------------------
The Conceptual Debate Graph (C-DAG) visualizes:
- [#ffcc33]Proposition[/] - The claim being evaluated
- [#33ff33]Supporting evidence[/] - Data favoring the claim
- [#ff3333]Attacking evidence[/] - Data against the claim
- [#b833ff]Rebuttals[/] - Challenges to evidence

[bold #ff8c00]Posterior Probability[/]
-----------------------------------------
The final probability reflects the jury's
assessment after considering all evidence:
- <30%: Claim is likely FALSE
- 30-70%: Uncertain, needs more evidence
- >70%: Claim is likely TRUE"""

    HELP_PROVIDERS = """[bold #ffcc33][*] LLM PROVIDERS[/]

[bold #ff8c00]Supported Providers (27+)[/]
-----------------------------------------
- [bold]OpenAI[/]: GPT-4, GPT-4-turbo, GPT-3.5
- [bold]Anthropic[/]: Claude 3 Opus/Sonnet/Haiku
- [bold]Google[/]: Gemini 1.5 Pro/Flash
- [bold]Cohere[/]: Command R+, Command
- [bold]Mistral[/]: Large, Medium, Small
- [bold]Groq[/]: LLaMA3, Mixtral (fast)
- [bold]DeepSeek[/]: Chat, Coder
- [bold]Together AI[/]: Various open models
- [bold]Fireworks[/]: LLaMA, Mixtral
- [bold]Ollama[/]: Local models (free!)

[bold #ff8c00]Configuration[/]
-----------------------------------------
1. Select provider from tree
2. Enter your API key
3. Choose model
4. Click Save

[bold #ff8c00]API Keys[/]
-----------------------------------------
Set via environment variables:
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GOOGLE_API_KEY
- COHERE_API_KEY
- MISTRAL_API_KEY"""

    HELP_TOOLS = """[bold #ffcc33][*] INTEGRATED TOOLS (19)[/]

[bold #ff8c00]Search Tools[/]
-----------------------------------------
- [bold]duckduckgo[/] - Web search
- [bold]wikipedia[/] - Encyclopedia
- [bold]arxiv[/] - Academic papers
- [bold]tavily[/] - AI search engine

[bold #ff8c00]Web Tools[/]
-----------------------------------------
- [bold]web_scraper[/] - Extract web content
- [bold]url_reader[/] - Read page text
- [bold]robots_checker[/] - Check crawl permissions

[bold #ff8c00]Code Tools[/]
-----------------------------------------
- [bold]python_repl[/] - Execute Python
- [bold]shell[/] - Run shell commands
- [bold]code_interpreter[/] - Analyze code

[bold #ff8c00]File Tools[/]
-----------------------------------------
- [bold]read_file[/] - Read file contents
- [bold]write_file[/] - Write to files
- [bold]list_directory[/] - Browse folders

[bold #ff8c00]Data Tools[/]
-----------------------------------------
- [bold]sql_query[/] - Query databases
- [bold]pandas[/] - Data analysis
- [bold]json_parser[/] - Parse JSON

[bold #ff8c00]Finance Tools[/]
-----------------------------------------
- [bold]stock_price[/] - Stock quotes
- [bold]sec_filings[/] - SEC documents
- [bold]company_info[/] - Company data"""
    
    def __init__(self) -> None:
        super().__init__()
        self._current_help = self.HELP_INTRO
    
    def compose(self) -> ComposeResult:
        """Compose the help screen layout."""
        with ScrollableContainer(id="help-scroll"):
            with Container(id="help-container"):
                # Header
                with Horizontal(id="header-bar"):
                    yield Static(
                        "[*] HELP  |  Documentation and keyboard shortcuts",
                        id="header-title"
                    )
                
                # Content
                with Horizontal(id="content-area"):
                    # Navigation sidebar
                    with Vertical(id="nav-sidebar"):
                        yield Static("[-] TOPICS", classes="section-title")
                        yield Button("[?] Introduction", id="help-intro", classes="nav-btn")
                        yield Button("[K] Keyboard", id="help-keyboard", classes="nav-btn")
                        yield Button("[X] Debate System", id="help-debate", classes="nav-btn")
                        yield Button("[C] Providers", id="help-providers", classes="nav-btn")
                        yield Button("[T] Tools", id="help-tools", classes="nav-btn")
                    
                    # Documentation content
                    with Vertical(id="help-content"):
                        yield Static("[-] DOCUMENTATION", classes="section-title")
                        with ScrollableContainer(id="doc-scroll"):
                            yield Static(self.HELP_INTRO, id="doc-content")
                
                yield Static(
                    "[dim]Press 1=Main 2=Debate 3=Providers 4=Tools 5=Knowledge 6=Benchmark 7=Settings 8=Help q=Quit[/dim]",
                    id="nav-hint"
                )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle navigation button presses."""
        content_map = {
            "help-intro": self.HELP_INTRO,
            "help-keyboard": self.HELP_KEYBOARD,
            "help-debate": self.HELP_DEBATE,
            "help-providers": self.HELP_PROVIDERS,
            "help-tools": self.HELP_TOOLS,
        }
        
        if event.button.id in content_map:
            doc = self.query_one("#doc-content", Static)
            doc.update(content_map[event.button.id])
