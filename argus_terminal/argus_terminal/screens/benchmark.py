"""
Benchmark Screen - Evaluation and benchmarking.

Features:
- Dataset browser
- Benchmark configuration
- Results display
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Button, Input, ProgressBar, DataTable

from argus_terminal.utils.argus_bridge import list_datasets


class BenchmarkScreen(Screen):
    """Benchmarking and evaluation screen."""
    
    DEFAULT_CSS = """
    BenchmarkScreen {
        background: #0a0a0a;
    }
    
    BenchmarkScreen #benchmark-scroll {
        width: 100%;
        height: 100%;
    }
    
    BenchmarkScreen #benchmark-container {
        width: 100%;
        height: auto;
        padding: 1;
    }
    
    BenchmarkScreen #header-bar {
        width: 100%;
        height: 3;
        background: #0d0d0d;
        border: solid #665200;
        padding: 0 1;
        margin-bottom: 1;
    }
    
    BenchmarkScreen #header-title {
        width: 100%;
        height: 1;
        color: #ffcc33;
        text-style: bold;
        margin-top: 1;
    }
    
    BenchmarkScreen #content-area {
        width: 100%;
        height: auto;
        min-height: 40;
        layout: horizontal;
    }
    
    BenchmarkScreen #left-panel {
        width: 35%;
        height: auto;
        padding-right: 1;
    }
    
    BenchmarkScreen #right-panel {
        width: 65%;
        height: auto;
    }
    
    BenchmarkScreen .panel {
        width: 100%;
        height: auto;
        background: #0d0d0d;
        border: solid #665200;
        padding: 1;
        margin-bottom: 1;
    }
    
    BenchmarkScreen .section-title {
        width: 100%;
        height: 1;
        color: #ffcc33;
        text-style: bold;
        margin-bottom: 1;
    }
    
    BenchmarkScreen #datasets-list {
        height: auto;
        max-height: 30;
        overflow-y: auto;
    }
    
    BenchmarkScreen .dataset-card {
        width: 100%;
        height: 4;
        background: #0a0a0a;
        border: solid #4d3800;
        padding: 0 1;
        margin-bottom: 1;
    }
    
    BenchmarkScreen .dataset-card:focus {
        border: solid #ffb000;
    }
    
    BenchmarkScreen .dataset-name {
        color: #ffcc33;
        text-style: bold;
    }
    
    BenchmarkScreen .dataset-desc {
        color: #ff8c00;
    }
    
    BenchmarkScreen .config-row {
        width: 100%;
        height: 3;
        layout: horizontal;
        margin-bottom: 1;
    }
    
    BenchmarkScreen .config-label {
        width: 12;
        height: 1;
        color: #ff8c00;
        margin-top: 1;
    }
    
    BenchmarkScreen .config-input {
        width: 1fr;
        height: 3;
    }
    
    BenchmarkScreen Input {
        background: #1a1400;
        border: solid #4d3800;
        color: #ffcc33;
    }
    
    BenchmarkScreen #run-btn {
        width: 100%;
        height: 3;
        margin-top: 1;
    }
    
    BenchmarkScreen #progress-panel {
        height: auto;
    }
    
    BenchmarkScreen ProgressBar {
        width: 100%;
        height: 1;
        margin: 1 0;
    }
    
    BenchmarkScreen #results-panel {
        height: auto;
    }
    
    BenchmarkScreen #metrics-row {
        width: 100%;
        height: 4;
        layout: horizontal;
        margin-bottom: 1;
    }
    
    BenchmarkScreen .metric-card {
        width: 1fr;
        height: 100%;
        background: #0a0a0a;
        border: solid #4d3800;
        padding: 0 1;
        margin-right: 1;
    }
    
    BenchmarkScreen .metric-label {
        color: #ff8c00;
    }
    
    BenchmarkScreen .metric-value {
        color: #33ff33;
        text-style: bold;
    }
    
    BenchmarkScreen DataTable {
        height: 15;
        background: #0a0a0a;
    }
    
    BenchmarkScreen #nav-hint {
        width: 100%;
        height: 1;
        color: #806000;
        margin-top: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        """Compose the benchmark screen layout."""
        datasets = list_datasets()
        
        with ScrollableContainer(id="benchmark-scroll"):
            with Container(id="benchmark-container"):
                # Header
                with Horizontal(id="header-bar"):
                    yield Static(
                        "[*] BENCHMARKS  |  Evaluation and performance testing",
                        id="header-title"
                    )
                
                # Content
                with Horizontal(id="content-area"):
                    # Left panel
                    with Vertical(id="left-panel"):
                        # Datasets
                        with Vertical(classes="panel"):
                            yield Static(f"[-] DATASETS ({len(datasets)})", classes="section-title")
                            
                            # Use a container for scrolling datasets specifically if needed, 
                            # but the whole page is scrollable now.
                            # We'll just stack buttons.
                            with Vertical(id="datasets-list"):
                                for ds in datasets:
                                    # Simple description based on name, as list_datasets currently returns strings
                                    desc = "Evaluation dataset"
                                    if "truthful" in ds.lower(): desc = "Truthfulness evaluation"
                                    elif "mmlu" in ds.lower(): desc = "Multitask understanding"
                                    elif "gsm8k" in ds.lower(): desc = "Math reasoning"
                                    
                                    yield Button(f"[DATA] {ds}\n{desc}", id=f"ds-{ds.lower()}", classes="dataset-card")
                        
                        # Configuration
                        with Vertical(classes="panel"):
                            yield Static("[-] CONFIGURATION", classes="section-title")
                            
                            with Horizontal(classes="config-row"):
                                yield Static("Samples:", classes="config-label")
                                yield Input(value="100", id="config-samples", classes="config-input")
                            
                            with Horizontal(classes="config-row"):
                                yield Static("Rounds:", classes="config-label")
                                yield Input(value="3", id="config-rounds", classes="config-input")
                            
                            with Horizontal(classes="config-row"):
                                yield Static("Model:", classes="config-label")
                                yield Input(value="gpt-4", id="config-model", classes="config-input")
                            
                            yield Button("> RUN BENCHMARK", id="run-btn", variant="primary")
                    
                    # Right panel
                    with Vertical(id="right-panel"):
                        # Progress
                        with Vertical(classes="panel", id="progress-panel"):
                            yield Static("[-] PROGRESS", classes="section-title")
                            yield ProgressBar(total=100, id="bench-progress")
                            yield Static("Ready to run benchmark...", id="progress-status")
                        
                        # Results
                        with Vertical(classes="panel", id="results-panel"):
                            yield Static("[-] RESULTS", classes="section-title")
                            
                            with Horizontal(id="metrics-row"):
                                with Vertical(classes="metric-card"):
                                    yield Static("Accuracy", classes="metric-label")
                                    yield Static("--", classes="metric-value", id="metric-accuracy")
                                with Vertical(classes="metric-card"):
                                    yield Static("Precision", classes="metric-label")
                                    yield Static("--", classes="metric-value", id="metric-precision")
                                with Vertical(classes="metric-card"):
                                    yield Static("Recall", classes="metric-label")
                                    yield Static("--", classes="metric-value", id="metric-recall")
                                with Vertical(classes="metric-card"):
                                    yield Static("F1 Score", classes="metric-label")
                                    yield Static("--", classes="metric-value", id="metric-f1")
                            
                            # Results table
                            table = DataTable(id="results-table")
                            table.add_columns("Sample", "Expected", "Predicted", "Correct")
                            table.add_row("1", "True", "True", "OK")
                            table.add_row("2", "False", "False", "OK")
                            table.add_row("3", "True", "False", "X")
                            table.add_row("4", "True", "True", "OK")
                            table.add_row("5", "False", "False", "OK")
                            yield table
                
                yield Static(
                    "[dim]Press 1=Main 2=Debate 3=Providers 4=Tools 5=Knowledge 6=Benchmark 7=Settings 8=Help q=Quit[/dim]",
                    id="nav-hint"
                )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "run-btn":
            self._run_demo_benchmark()
        elif event.button.id and event.button.id.startswith("ds-"):
            name = event.button.id.replace("ds-", "").upper()
            self.notify(f"Selected dataset: {name}", severity="information")
    
    def _run_demo_benchmark(self) -> None:
        """Run a demo benchmark."""
        progress = self.query_one("#bench-progress", ProgressBar)
        status = self.query_one("#progress-status", Static)
        
        progress.progress = 100
        status.update("[#33ff33]Benchmark complete![/]")
        
        # Update metrics
        self.query_one("#metric-accuracy", Static).update("[bold #33ff33]87.5%[/]")
        self.query_one("#metric-precision", Static).update("[bold #33ff33]89.2%[/]")
        self.query_one("#metric-recall", Static).update("[bold #33ff33]85.1%[/]")
        self.query_one("#metric-f1", Static).update("[bold #33ff33]87.1%[/]")
        
        self.notify("Benchmark completed successfully!", severity="information")
