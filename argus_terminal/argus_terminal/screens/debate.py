"""
Debate Screen - Full debate system interface.

Features:
- Proposition input
- Agent orchestra display
- Live debate progress
- C-DAG visualization
- Verdict display
"""

from textual import work
from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Button, Input, Select
from textual.worker import Worker, WorkerState

from argus_terminal.widgets.agent_status import AgentStatusWidget
from argus_terminal.widgets.cdag_view import CDAGViewWidget
from argus_terminal.widgets.posterior_gauge import PosteriorGaugeWidget
from argus_terminal.widgets.log_viewer import LogViewerWidget, LogLevel
from argus_terminal.widgets.progress_panel import ProgressPanel
from argus_terminal.utils.argus_bridge import run_debate


class DebateScreen(Screen):
    """Debate system screen with full orchestration display."""
    
    DEFAULT_CSS = """
    DebateScreen {
        background: #0a0a0a;
    }
    
    DebateScreen #debate-scroll {
        width: 100%;
        height: 100%;
    }
    
    DebateScreen #debate-container {
        width: 100%;
        height: auto;
        padding: 1;
    }
    
    DebateScreen #header-bar {
        width: 100%;
        height: 3;
        background: #0d0d0d;
        border: solid #665200;
        padding: 0 1;
        margin-bottom: 1;
    }
    
    DebateScreen #header-title {
        width: 100%;
        height: 1;
        color: #ffcc33;
        text-style: bold;
        margin-top: 1;
    }
    
    DebateScreen #content-area {
        width: 100%;
        height: auto;
        min-height: 40;
        layout: horizontal;
    }
    
    DebateScreen #left-panel {
        width: 40%;
        height: auto;
        padding-right: 1;
    }
    
    DebateScreen #right-panel {
        width: 60%;
        height: auto;
    }
    
    DebateScreen .panel {
        width: 100%;
        height: auto;
        background: #0d0d0d;
        border: solid #665200;
        padding: 1;
        margin-bottom: 1;
    }
    
    DebateScreen .panel-title {
        width: 100%;
        height: 1;
        color: #ffcc33;
        text-style: bold;
        margin-bottom: 1;
    }
    
    DebateScreen .input-row {
        width: 100%;
        height: 3;
        layout: horizontal;
        margin-bottom: 1;
    }
    
    DebateScreen .input-label {
        width: 15;
        height: 1;
        color: #ff8c00;
        margin-top: 1;
    }
    
    DebateScreen .input-field {
        width: 1fr;
        height: 3;
    }
    
    DebateScreen Input {
        background: #1a1400;
        border: solid #4d3800;
        color: #ffcc33;
    }
    
    DebateScreen Input:focus {
        border: solid #ffb000;
    }
    
    DebateScreen #start-btn {
        width: 100%;
        height: 3;
        background: #665200;
        border: solid #997a00;
        color: #ffcc33;
        text-style: bold;
        margin-top: 1;
    }
    
    DebateScreen #start-btn:hover {
        background: #806000;
    }
    
    DebateScreen #start-btn:focus {
        border: solid #ffb000;
    }
    
    DebateScreen #agents-panel {
        height: auto;
        min-height: 12;
    }
    
    DebateScreen #progress-panel {
        height: auto;
    }
    
    DebateScreen #graph-panel {
        height: 30;
    }
    
    DebateScreen #log-panel {
        height: 15;
    }
    
    DebateScreen #verdict-panel {
        height: auto;
    }
    
    DebateScreen #nav-hint {
        width: 100%;
        height: 1;
        color: #806000;
        margin-top: 1;
    }
    """
    
    def __init__(self) -> None:
        super().__init__()
        self._is_running = False
    
    def compose(self) -> ComposeResult:
        """Compose the debate screen layout."""
        with ScrollableContainer(id="debate-scroll"):
            with Container(id="debate-container"):
                # Header
                with Horizontal(id="header-bar"):
                    yield Static(
                        "[*] DEBATE SYSTEM  |  Multi-agent AI debate orchestration",
                        id="header-title"
                    )
                
                # Content
                with Horizontal(id="content-area"):
                    # Left panel - Configuration
                    with Vertical(id="left-panel"):
                        # Proposition input
                        with Vertical(classes="panel"):
                            yield Static("[-] PROPOSITION", classes="panel-title")
                            yield Input(
                                placeholder="Enter a claim to debate...",
                                id="proposition-input"
                            )
                            
                            with Horizontal(classes="input-row"):
                                yield Static("Prior:", classes="input-label")
                                yield Input(value="0.5", id="prior-input", classes="input-field")
                            
                            with Horizontal(classes="input-row"):
                                yield Static("Domain:", classes="input-label")
                                yield Input(value="General", id="domain-input", classes="input-field")
                            
                            with Horizontal(classes="input-row"):
                                yield Static("Max Rounds:", classes="input-label")
                                yield Input(value="5", id="rounds-input", classes="input-field")
                            
                            yield Button("> START DEBATE", id="start-btn", variant="primary")
                        
                        # Agent Orchestra
                        with Vertical(classes="panel", id="agents-panel"):
                            yield Static("[-] AGENT ORCHESTRA", classes="panel-title")
                            yield AgentStatusWidget(id="agent-status")
                        
                        # Progress
                        with Vertical(classes="panel", id="progress-panel"):
                            yield Static("[-] PROGRESS", classes="panel-title")
                            yield ProgressPanel(id="debate-progress")
                    
                    # Right panel - Results
                    with Vertical(id="right-panel"):
                        # C-DAG Graph
                        with Vertical(classes="panel", id="graph-panel"):
                            yield Static("[-] CONCEPTUAL DEBATE GRAPH (C-DAG)", classes="panel-title")
                            yield CDAGViewWidget(id="cdag-view")
                        
                        # Verdict
                        with Vertical(classes="panel", id="verdict-panel"):
                            yield Static("[-] VERDICT", classes="panel-title")
                            yield PosteriorGaugeWidget(id="verdict-gauge")
                
                yield Static(
                    "[dim]Press 1=Main 2=Debate 3=Providers 4=Tools 5=Knowledge 6=Benchmark 7=Settings 8=Help q=Quit[/dim]",
                    id="nav-hint"
                )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "start-btn":
            self.start_debate_process()
            
    def start_debate_process(self) -> None:
        """Initiate the debate process."""
        if self._is_running:
            return
            
        proposition = self.query_one("#proposition-input", Input).value
        # Defensive check for textual versions that might return None
        if not proposition:
            return
            
        self._is_running = True
        
        # Reset widgets
        self.query_one("#agent-status", AgentStatusWidget).reset()
        self.query_one("#cdag-view", CDAGViewWidget).clear()
        self.query_one("#debate-progress", ProgressPanel).start(total_steps=5, status="Initializing debate...")
        
        # Start background worker
        self.run_debate_worker(proposition)
        
    @work(exclusive=True)
    async def run_debate_worker(self, proposition: str) -> None:
        """Run debate in background."""
        try:
            # Update status to working
            agent_status = self.query_one("#agent-status", AgentStatusWidget)
            agent_status.update_agent("mod", state="working", action="Analyzing proposition")
            
            progress = self.query_one("#debate-progress", ProgressPanel)
            progress.update(step=1, status="Debating...")
            
            # Run debate (this call mimics the synchronous argus call)
            # Add defensive checks for input values
            prior_input = self.query_one("#prior-input", Input).value
            rounds_input = self.query_one("#rounds-input", Input).value
            
            try:
                prior_val = float(prior_input) if prior_input else 0.5
                rounds_val = int(rounds_input) if rounds_input else 5
            except ValueError:
                prior_val = 0.5
                rounds_val = 5
            
            result = run_debate(
                proposition=proposition,
                prior=prior_val,
                max_rounds=rounds_val
            )
            
            # Handle completion
            if result.get("error"):
                progress.update(step=5, status=f"Error: {result['error']}")
                agent_status.update_agent("mod", state="error")
            else:
                # Update UI with results
                progress.complete(status="Debate Completed")
                
                # Update verdict
                verdict_data = result.get("verdict", {})
                if verdict_data:
                    self.query_one("#verdict-gauge", PosteriorGaugeWidget).update_posterior(
                        verdict_data.get("posterior", 0.5)
                    )
                
                # Update graph if available (demo graph for now if none returned)
                # In real app, we'd parse the graph object
                self.query_one("#cdag-view", CDAGViewWidget).set_demo_data()
                
                # Set agents to complete
                agent_status.update_agent("mod", state="complete")
                agent_status.update_agent("spec", state="complete")
                agent_status.update_agent("ref", state="complete")
                agent_status.update_agent("jury", state="complete")
                
        except Exception as e:
            self.query_one("#debate-progress", ProgressPanel).update(step=0, status=f"System Error: {str(e)}")
        finally:
            self._is_running = False

