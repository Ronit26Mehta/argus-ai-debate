"""
Sidebar Widget - Navigation menu for Argus Terminal.

Features:
- Module navigation
- Visual indicators for active screen
- Icons for each module
"""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static, Button
from textual.reactive import reactive
from textual.message import Message


class SidebarItem(Button):
    """A single sidebar navigation item."""
    
    DEFAULT_CSS = """
    SidebarItem {
        width: 100%;
        height: 3;
        min-width: 20;
        background: transparent;
        border: none;
        padding: 0 1;
        margin: 0;
        content-align: left middle;
    }
    
    SidebarItem:hover {
        background: #1a1400;
    }
    
    SidebarItem:focus {
        background: #1a1400;
        text-style: bold;
    }
    
    SidebarItem.--active {
        background: #ffb000;
        color: #0a0a0a;
        text-style: bold;
    }
    """
    
    class Selected(Message):
        """Message emitted when sidebar item is selected."""
        
        def __init__(self, item_id: str, label: str) -> None:
            super().__init__()
            self.item_id = item_id
            self.label = label
    
    def __init__(
        self,
        label: str,
        icon: str = "▸",
        item_id: str = "",
        hotkey: str = "",
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        display_text = f"{icon} {label}"
        if hotkey:
            display_text = f"{display_text} [{hotkey}]"
        super().__init__(display_text, name=name, id=id, classes=classes)
        self.item_id = item_id or label.lower().replace(" ", "_")
        self.label_text = label
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        event.stop()
        self.post_message(self.Selected(self.item_id, self.label_text))


class SidebarWidget(Static):
    """Sidebar navigation widget."""
    
    DEFAULT_CSS = """
    SidebarWidget {
        dock: left;
        width: 26;
        height: 100%;
        background: #0d0d0d;
        border-right: double #665200;
        padding: 0;
    }
    
    SidebarWidget #sidebar-header {
        width: 100%;
        height: 3;
        background: #0d0d0d;
        border-bottom: solid #4d3800;
        padding: 1 1;
        content-align: center middle;
    }
    
    SidebarWidget #sidebar-title {
        color: #ffcc33;
        text-style: bold;
    }
    
    SidebarWidget #sidebar-nav {
        width: 100%;
        height: auto;
        padding: 1 0;
    }
    
    SidebarWidget #sidebar-footer {
        dock: bottom;
        width: 100%;
        height: 3;
        border-top: solid #4d3800;
        padding: 1;
        content-align: center middle;
        color: #806000;
    }
    """
    
    # Reactive active item
    active_item: reactive[str] = reactive("main")
    
    # Navigation items with icons
    MENU_ITEMS = [
        ("main", "Dashboard", "◈", "F1"),
        ("debate", "Debate", "⚔", "F2"),
        ("providers", "Providers", "☁", "F3"),
        ("tools", "Tools", "⚙", "F4"),
        ("knowledge", "Knowledge", "📚", "F5"),
        ("benchmark", "Benchmark", "📊", "F6"),
        ("settings", "Settings", "⚡", "F7"),
        ("help", "Help", "❓", "F8"),
    ]
    
    def __init__(
        self,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
    
    def compose(self) -> ComposeResult:
        """Compose the sidebar layout."""
        with Vertical(id="sidebar-header"):
            yield Static("◈ NAVIGATION", id="sidebar-title")
        
        with Vertical(id="sidebar-nav"):
            for item_id, label, icon, hotkey in self.MENU_ITEMS:
                item = SidebarItem(
                    label=label,
                    icon=icon,
                    item_id=item_id,
                    hotkey=hotkey,
                    id=f"nav-{item_id}",
                )
                if item_id == self.active_item:
                    item.add_class("--active")
                yield item
        
        yield Static("v2.0.0", id="sidebar-footer")
    
    def watch_active_item(self, value: str) -> None:
        """Update active item styling when reactive changes."""
        for item_id, _, _, _ in self.MENU_ITEMS:
            try:
                item = self.query_one(f"#nav-{item_id}", SidebarItem)
                if item_id == value:
                    item.add_class("--active")
                else:
                    item.remove_class("--active")
            except Exception:
                pass
    
    def set_active(self, item_id: str) -> None:
        """Set the active navigation item."""
        self.active_item = item_id
    
    def on_sidebar_item_selected(self, event: SidebarItem.Selected) -> None:
        """Handle sidebar item selection."""
        self.set_active(event.item_id)
