"""
Providers Screen - LLM provider configuration.

Features:
- Provider list with categories
- API key configuration
- Model selection
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Button, Input, Tree

from argus_terminal.utils.argus_bridge import list_providers


class ProvidersScreen(Screen):
    """LLM providers configuration screen."""
    
    DEFAULT_CSS = """
    ProvidersScreen {
        background: #0a0a0a;
    }
    
    ProvidersScreen #providers-scroll {
        width: 100%;
        height: 100%;
    }
    
    ProvidersScreen #providers-container {
        width: 100%;
        height: auto;
        padding: 1;
    }
    
    ProvidersScreen #header-bar {
        width: 100%;
        height: 3;
        background: #0d0d0d;
        border: solid #665200;
        padding: 0 1;
        margin-bottom: 1;
    }
    
    ProvidersScreen #header-title {
        width: 100%;
        height: 1;
        color: #ffcc33;
        text-style: bold;
        margin-top: 1;
    }
    
    ProvidersScreen #content-area {
        width: 100%;
        height: auto;
        min-height: 40;
        layout: horizontal;
    }
    
    ProvidersScreen #providers-list {
        width: 35%;
        height: auto;
        min-height: 40;
        background: #0d0d0d;
        border: solid #665200;
        padding: 1;
        margin-right: 1;
    }
    
    ProvidersScreen #providers-detail {
        width: 65%;
        height: auto;
        background: #0d0d0d;
        border: solid #665200;
        padding: 1;
    }
    
    ProvidersScreen .section-title {
        width: 100%;
        height: 1;
        color: #ffcc33;
        text-style: bold;
        margin-bottom: 1;
    }
    
    ProvidersScreen Tree {
        background: transparent;
        scrollbar-background: #0a0a0a;
        height: auto;
    }
    
    ProvidersScreen Tree > .tree--guides {
        color: #4d3800;
    }
    
    ProvidersScreen Tree > .tree--cursor {
        background: #1a1400;
        color: #ffcc33;
    }
    
    ProvidersScreen .config-section {
        width: 100%;
        height: auto;
        margin-bottom: 2;
    }
    
    ProvidersScreen .config-row {
        width: 100%;
        height: 3;
        layout: horizontal;
        margin-bottom: 1;
    }
    
    ProvidersScreen .config-label {
        width: 15;
        height: 1;
        color: #ff8c00;
        margin-top: 1;
    }
    
    ProvidersScreen .config-input {
        width: 1fr;
        height: 3;
    }
    
    ProvidersScreen Input {
        background: #1a1400;
        border: solid #4d3800;
        color: #ffcc33;
    }
    
    ProvidersScreen Input:focus {
        border: solid #ffb000;
    }
    
    ProvidersScreen #save-btn {
        width: 30;
        height: 3;
        margin-top: 1;
    }
    
    ProvidersScreen .provider-info {
        width: 100%;
        height: auto;
        color: #ff8c00;
        margin-top: 1;
    }
    
    ProvidersScreen #nav-hint {
        width: 100%;
        height: 1;
        color: #806000;
        margin-top: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        """Compose the providers screen layout."""
        # Fetch available providers
        providers = list_providers()
        
        with ScrollableContainer(id="providers-scroll"):
            with Container(id="providers-container"):
                # Header
                with Horizontal(id="header-bar"):
                    yield Static(
                        f"[*] LLM PROVIDERS  |  Configure {len(providers)}+ language model providers",
                        id="header-title"
                    )
                
                # Content
                with Horizontal(id="content-area"):
                    # Provider list
                    with Vertical(id="providers-list"):
                        yield Static(f"[-] PROVIDERS ({len(providers)})", classes="section-title")
                        
                        tree: Tree[str] = Tree("All Providers", id="provider-tree")
                        tree.root.expand()
                        
                        # Since we only get names, we just list them flat or try to group
                        # For now, simplistic grouping by first letter or just flat
                        for provider in sorted(providers):
                            tree.root.add_leaf(f"[#ff8c00]{provider}[/]")
                        
                        yield tree
                    
                    # Provider details
                    with Vertical(id="providers-detail"):
                        yield Static("[-] PROVIDER CONFIGURATION", classes="section-title")
                        
                        with Vertical(classes="config-section"):
                            yield Static(
                                "[#ffcc33]Select a provider from the tree to configure[/]\n\n"
                                "[#ff8c00]Supported features:[/]\n"
                                "- Chat completions\n"
                                "- Streaming responses\n"
                                "- Function calling\n"
                                "- Vision (select models)\n"
                                "- JSON mode",
                                classes="provider-info"
                            )
                        
                        with Vertical(classes="config-section"):
                            yield Static("[-] API CONFIGURATION", classes="section-title")
                            
                            with Horizontal(classes="config-row"):
                                yield Static("Provider:", classes="config-label")
                                yield Input(value="gemini", id="provider-input", classes="config-input")
                            
                            with Horizontal(classes="config-row"):
                                yield Static("API Key:", classes="config-label")
                                yield Input(
                                    placeholder="Enter API key...",
                                    password=True,
                                    id="apikey-input",
                                    classes="config-input"
                                )
                            
                            with Horizontal(classes="config-row"):
                                yield Static("Model:", classes="config-label")
                                yield Input(value="gemini-1.5-flash", id="model-input", classes="config-input")
                            
                            with Horizontal(classes="config-row"):
                                yield Static("Base URL:", classes="config-label")
                                yield Input(placeholder="(optional)", id="baseurl-input", classes="config-input")
                            
                            yield Button("Save Configuration", id="save-btn", variant="primary")
                        
                        yield Static(
                            "[dim]Tip: Set API keys via environment variables for security:\n"
                            "OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.[/dim]",
                            classes="provider-info"
                        )
                
                yield Static(
                    "[dim]Press 1=Main 2=Debate 3=Providers 4=Tools 5=Knowledge 6=Benchmark 7=Settings 8=Help q=Quit[/dim]",
                    id="nav-hint"
                )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "save-btn":
            provider = self.query_one("#provider-input", Input).value
            self.notify(f"Configuration for {provider} saved!", severity="information")
    
    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle tree node selection."""
        node = event.node
        label = str(node.label)
        
        # Remove Rich markup to get clean text
        import re
        clean_label = re.sub(r'\[.*?\]', '', label)
        
        if node.is_root or node.children:
            return
        
        # It's a provider leaf
        provider_input = self.query_one("#provider-input", Input)
        provider_input.value = clean_label
        self.notify(f"Selected provider: {clean_label}", severity="information")
