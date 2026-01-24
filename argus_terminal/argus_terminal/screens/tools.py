"""
Tools Screen - Tool browser and executor.

Features:
- Tool categories tree
- Tool details panel
- Parameter input
- Execution output
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Button, Input, Tree

from argus_terminal.utils.argus_bridge import list_tools


class ToolsScreen(Screen):
    """Tools browser and executor screen."""
    
    DEFAULT_CSS = """
    ToolsScreen {
        background: #0a0a0a;
    }
    
    ToolsScreen #tools-scroll {
        width: 100%;
        height: 100%;
    }
    
    ToolsScreen #tools-container {
        width: 100%;
        height: auto;
        padding: 1;
    }
    
    ToolsScreen #header-bar {
        width: 100%;
        height: 3;
        background: #0d0d0d;
        border: solid #665200;
        padding: 0 1;
        margin-bottom: 1;
    }
    
    ToolsScreen #header-title {
        width: 100%;
        height: 1;
        color: #ffcc33;
        text-style: bold;
        margin-top: 1;
    }
    
    ToolsScreen #content-area {
        width: 100%;
        height: auto;
        min-height: 40;
        layout: horizontal;
    }
    
    ToolsScreen #tools-list {
        width: 30%;
        height: auto;
        min-height: 40;
        background: #0d0d0d;
        border: solid #665200;
        padding: 1;
        margin-right: 1;
    }
    
    ToolsScreen #tools-detail {
        width: 70%;
        height: auto;
    }
    
    ToolsScreen .section-title {
        width: 100%;
        height: 1;
        color: #ffcc33;
        text-style: bold;
        margin-bottom: 1;
    }
    
    ToolsScreen Tree {
        background: transparent;
        height: auto;
    }
    
    ToolsScreen Tree > .tree--cursor {
        background: #1a1400;
        color: #ffcc33;
    }
    
    ToolsScreen .panel {
        width: 100%;
        height: auto;
        background: #0d0d0d;
        border: solid #665200;
        padding: 1;
        margin-bottom: 1;
    }
    
    ToolsScreen .config-row {
        width: 100%;
        height: 3;
        layout: horizontal;
        margin-bottom: 1;
    }
    
    ToolsScreen .config-label {
        width: 12;
        height: 1;
        color: #ff8c00;
        margin-top: 1;
    }
    
    ToolsScreen .config-input {
        width: 1fr;
        height: 3;
    }
    
    ToolsScreen Input {
        background: #1a1400;
        border: solid #4d3800;
        color: #ffcc33;
    }
    
    ToolsScreen #execute-btn {
        width: 20;
        height: 3;
    }
    
    ToolsScreen #output-panel {
        height: auto;
        min-height: 15;
    }
    
    ToolsScreen #output-content {
        width: 100%;
        height: 15;
        background: #0a0a0a;
        color: #33ff33;
        padding: 1;
    }
    
    ToolsScreen #nav-hint {
        width: 100%;
        height: 1;
        color: #806000;
        margin-top: 1;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.available_tools = []
    
    def compose(self) -> ComposeResult:
        """Compose the tools screen layout."""
        self.available_tools = list_tools()
        
        # Group tools by category
        categories = {}
        for tool in self.available_tools:
            # Defensive check: ensure dictionary structure
            if isinstance(tool, str):
                tool = {"name": tool, "category": "Uncategorized", "description": "Unknown"}
            elif not isinstance(tool, dict):
                continue
                
            cat = tool.get("category", "Uncategorized")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(tool)
            
        with ScrollableContainer(id="tools-scroll"):
            with Container(id="tools-container"):
                # Header
                with Horizontal(id="header-bar"):
                    yield Static(
                        f"[*] TOOLS  |  Browse and execute {len(self.available_tools)} integrated tools",
                        id="header-title"
                    )
                
                # Content
                with Horizontal(id="content-area"):
                    # Tools list
                    with Vertical(id="tools-list"):
                        yield Static("[-] AVAILABLE TOOLS", classes="section-title")
                        
                        tree: Tree[str] = Tree("Tools", id="tools-tree")
                        tree.root.expand()
                        
                        for category, tools in categories.items():
                            cat_node = tree.root.add(f"[#ffcc33]{category}[/]", expand=True)
                            for tool in tools:
                                name = tool.get("name", "Unknown")
                                cat_node.add_leaf(f"[#ff8c00]{name}[/]")
                        
                        yield tree
                    
                    # Tool details
                    with Vertical(id="tools-detail"):
                        # Tool info
                        with Vertical(classes="panel"):
                            yield Static("[-] TOOL: -", classes="section-title", id="tool-name")
                            yield Static(
                                "Select a tool from the list to view details.",
                                id="tool-desc"
                            )
                        
                        # Parameters
                        with Vertical(classes="panel"):
                            yield Static("[-] PARAMETERS", classes="section-title")
                            
                            with Horizontal(classes="config-row"):
                                yield Static("Input:", classes="config-label")
                                yield Input(
                                    placeholder="Enter arguments...",
                                    id="param-query",
                                    classes="config-input"
                                )
                            
                            yield Button("> Execute", id="execute-btn", variant="primary")
                        
                        # Output
                        with Vertical(classes="panel", id="output-panel"):
                            yield Static("[-] OUTPUT", classes="section-title")
                            with ScrollableContainer(id="output-content"):
                                yield Static(
                                    "[dim]Execute a tool to see output here.[/dim]",
                                    id="output-text"
                                )
                
                yield Static(
                    "[dim]Press 1=Main 2=Debate 3=Providers 4=Tools 5=Knowledge 6=Benchmark 7=Settings 8=Help q=Quit[/dim]",
                    id="nav-hint"
                )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "execute-btn":
            query_input = self.query_one("#param-query", Input)
            output_text = self.query_one("#output-text", Static)
            tool_name = self.query_one("#tool-name", Static)
            
            # Get clean tool name
            import re
            tool_label = str(tool_name.renderable)
            tool_clean = re.sub(r'\[-\] TOOL: ', '', tool_label)
            # Also strip ASCII icon if present (e.g. "[WEB] ")
            tool_clean = tool_clean.split(' ')[-1] if ' ' in tool_clean else tool_clean
            
            if tool_clean == "-" or not tool_clean:
                self.notify("Please select a tool first.", severity="error")
                return

            query = query_input.value or ""
            
            output_text.update(f"[dim]Executing {tool_clean}...[/]")
            
            # Execute via bridge
            from argus_terminal.utils.argus_bridge import execute_tool
            result = execute_tool(tool_clean, query)
            
            output_text.update(
                f"[#33ff33]$ {tool_clean} \"{query}\"[/]\n"
                "---------------------------------\n"
                f"{result}"
            )
            self.notify("Tool executed!", severity="information")
    
    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle tree node selection."""
        node = event.node
        label = str(node.label)
        
        # Remove Rich markup to get clean text
        import re
        clean_label = re.sub(r'\[.*?\]', '', label)
        
        if node.is_root or node.children:
            return
        
        # Find tool details
        selected_tool = next((t for t in self.available_tools 
                              if isinstance(t, dict) and t.get("name") == clean_label), None)
        
        if selected_tool:
            tool_name = self.query_one("#tool-name", Static)
            tool_desc = self.query_one("#tool-desc", Static)
            
            tool_name.update(f"[-] TOOL: {clean_label}")
            
            desc = selected_tool.get("description", "No description")
            cat = selected_tool.get("category", "General")
            
            tool_desc.update(
                f"[#ff8c00]Description:[/] {desc}\n"
                f"[#ff8c00]Category:[/] {cat}\n"
            )
            
            self.notify(f"Selected tool: {clean_label}", severity="information")
