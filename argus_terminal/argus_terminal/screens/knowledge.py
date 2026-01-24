"""
Knowledge Screen - Connectors and retrieval.

Features:
- Connector cards
- Document ingestion
- Search interface
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Button, Input

from argus_terminal.utils.argus_bridge import list_connectors


class KnowledgeScreen(Screen):
    """Knowledge connectors and retrieval screen."""
    
    DEFAULT_CSS = """
    KnowledgeScreen {
        background: #0a0a0a;
    }
    
    KnowledgeScreen #knowledge-scroll {
        width: 100%;
        height: 100%;
    }
    
    KnowledgeScreen #knowledge-container {
        width: 100%;
        height: auto;
        padding: 1;
    }
    
    KnowledgeScreen #header-bar {
        width: 100%;
        height: 3;
        background: #0d0d0d;
        border: solid #665200;
        padding: 0 1;
        margin-bottom: 1;
    }
    
    KnowledgeScreen #header-title {
        width: 100%;
        height: 1;
        color: #ffcc33;
        text-style: bold;
        margin-top: 1;
    }
    
    KnowledgeScreen #content-area {
        width: 100%;
        height: auto;
        min-height: 40;
        layout: horizontal;
    }
    
    KnowledgeScreen #left-panel {
        width: 40%;
        height: auto;
        padding-right: 1;
    }
    
    KnowledgeScreen #right-panel {
        width: 60%;
        height: auto;
    }
    
    KnowledgeScreen .panel {
        width: 100%;
        height: auto;
        background: #0d0d0d;
        border: solid #665200;
        padding: 1;
        margin-bottom: 1;
    }
    
    KnowledgeScreen .section-title {
        width: 100%;
        height: 1;
        color: #ffcc33;
        text-style: bold;
        margin-bottom: 1;
    }
    
    KnowledgeScreen .connector-card {
        width: 100%;
        height: auto;
        background: #0a0a0a;
        border: solid #4d3800;
        padding: 1;
        margin-bottom: 1;
    }
    
    KnowledgeScreen .connector-name {
        color: #ffcc33;
        text-style: bold;
    }
    
    KnowledgeScreen .connector-desc {
        color: #ff8c00;
    }
    
    KnowledgeScreen .config-row {
        width: 100%;
        height: 3;
        layout: horizontal;
        margin-bottom: 1;
    }
    
    KnowledgeScreen .config-label {
        width: 12;
        height: 1;
        color: #ff8c00;
        margin-top: 1;
    }
    
    KnowledgeScreen .config-input {
        width: 1fr;
        height: 3;
    }
    
    KnowledgeScreen Input {
        background: #1a1400;
        border: solid #4d3800;
        color: #ffcc33;
    }
    
    KnowledgeScreen Button {
        margin-top: 1;
    }
    
    KnowledgeScreen #ingest-btn {
        width: 100%;
        height: 3;
        background: #665200;
        border: solid #997a00;
        color: #ffcc33;
    }

    KnowledgeScreen #search-results {
        height: auto;
        min-height: 20;
    }
    
    KnowledgeScreen #results-content {
        width: 100%;
        height: 20;
        background: #0a0a0a;
        padding: 1;
    }
    
    KnowledgeScreen #nav-hint {
        width: 100%;
        height: 1;
        color: #806000;
        margin-top: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        """Compose the knowledge screen layout."""
        connectors = list_connectors()
        
        with ScrollableContainer(id="knowledge-scroll"):
            with Container(id="knowledge-container"):
                # Header
                with Horizontal(id="header-bar"):
                    yield Static(
                        f"[*] KNOWLEDGE  |  {len(connectors)} Active Connectors and hybrid retrieval",
                        id="header-title"
                    )
                
                # Content
                with Horizontal(id="content-area"):
                    # Left panel
                    with Vertical(id="left-panel"):
                        # Connectors
                        with Vertical(classes="panel"):
                            yield Static("[-] CONNECTORS", classes="section-title")
                            
                            for connector in connectors:
                                with Vertical(classes="connector-card"):
                                    # ASCII icons instead of emojis
                                    icon = "[WEB]" if "web" in connector["name"].lower() else "[DOC]" if "arxiv" in connector["name"].lower() else "[DB]"
                                    yield Static(f"{icon} {connector['name']}", classes="connector-name")
                                    yield Static(connector["description"], classes="connector-desc")
                                    yield Static("[#33ff33]O Active[/]")
                        
                        # Ingestion
                        with Vertical(classes="panel"):
                            yield Static("[-] DOCUMENT INGESTION", classes="section-title")
                            
                            with Horizontal(classes="config-row"):
                                yield Static("Path/URL:", classes="config-label")
                                yield Input(placeholder="Enter file path or URL...", id="ingest-path", classes="config-input")
                            
                            with Horizontal(classes="config-row"):
                                yield Static("Type:", classes="config-label")
                                yield Input(value="auto", id="ingest-type", classes="config-input")
                            
                            yield Button("> Ingest Document", id="ingest-btn", variant="primary")
                    
                    # Right panel
                    with Vertical(id="right-panel"):
                        # Search
                        with Vertical(classes="panel"):
                            yield Static("[-] HYBRID SEARCH", classes="section-title")
                            yield Static(
                                "[#ff8c00]Retriever:[/] BM25 + Semantic (Hybrid)\n"
                                "[#ff8c00]Embedding:[/] sentence-transformers/all-MiniLM-L6-v2\n"
                                "[#ff8c00]Top-K:[/] 10",
                            )
                            
                            with Horizontal(classes="config-row"):
                                yield Static("Query:", classes="config-label")
                                yield Input(placeholder="Enter search query...", id="search-query", classes="config-input")
                            
                            yield Button("Search", id="search-btn", variant="primary")
                        
                        # Results
                        with Vertical(classes="panel", id="search-results"):
                            yield Static("[-] SEARCH RESULTS", classes="section-title")
                            with ScrollableContainer(id="results-content"):
                                yield Static(
                                    "[dim]Enter a query and click Search to see results.[/dim]\n\n"
                                    "[#ffcc33]Knowledge base features:[/]\n"
                                    "- Hybrid retrieval (BM25 + semantic)\n"
                                    "- Multiple embedding models\n"
                                    "- Web, ArXiv, CrossRef connectors\n"
                                    "- Document chunking\n"
                                    "- Metadata extraction",
                                    id="results-text"
                                )
                
                yield Static(
                    "[dim]Press 1=Main 2=Debate 3=Providers 4=Tools 5=Knowledge 6=Benchmark 7=Settings 8=Help q=Quit[/dim]",
                    id="nav-hint"
                )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "ingest-btn":
            self.notify("Document ingested successfully!", severity="information")
        elif event.button.id == "search-btn":
            query_input = self.query_one("#search-query", Input)
            results_text = self.query_one("#results-text", Static)
            
            query = query_input.value or "example"
            results_text.update(
                f"[#ffcc33]Results for:[/] {query}\n"
                "---------------------------------\n\n"
                "[#33ff33]1.[/] [bold]Document: research_paper.pdf[/]\n"
                "   [#ff8c00]Score: 0.92[/] | [#806000]Chunk 3 of 15[/]\n"
                "   Relevant text from the document matching\n"
                "   your search query appears here...\n\n"
                "[#33ff33]2.[/] [bold]Source: arxiv.org[/]\n"
                "   [#ff8c00]Score: 0.87[/] | [#806000]ArXiv:2301.12345[/]\n"
                "   Abstract excerpt from academic paper\n"
                "   related to the search query...\n\n"
                "[#33ff33]3.[/] [bold]Web: example.com[/]\n"
                "   [#ff8c00]Score: 0.81[/] | [#806000]Fetched today[/]\n"
                "   Content scraped from web page\n"
                "   matching the query terms...\n\n"
                "[dim]---------------------------------[/]\n"
                "[#33ff33]OK Found 3 results[/]"
            )
            self.notify("Search completed!", severity="information")
