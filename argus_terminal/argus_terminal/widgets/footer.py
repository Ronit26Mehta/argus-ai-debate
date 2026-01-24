"""
Footer Widget - Command bar and keyboard shortcut hints.

Features:
- Command prompt input
- Keyboard shortcut hints
- Status messages
"""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static
from textual.reactive import reactive


class FooterWidget(Static):
    """Footer widget with command hints and shortcuts."""
    
    DEFAULT_CSS = """
    FooterWidget {
        dock: bottom;
        height: 3;
        width: 100%;
        background: #0d0d0d;
        border-top: double #665200;
        padding: 0 2;
    }
    
    FooterWidget #footer-container {
        width: 100%;
        height: 100%;
        layout: horizontal;
    }
    
    FooterWidget #footer-shortcuts {
        width: 1fr;
        height: 1;
        margin-top: 1;
        color: #806000;
    }
    
    FooterWidget .hotkey {
        color: #b833ff;
        text-style: bold;
    }
    
    FooterWidget #footer-message {
        width: auto;
        height: 1;
        margin-top: 1;
        margin-right: 2;
        color: #ffb000;
    }
    """
    
    # Reactive properties
    message: reactive[str] = reactive("")
    
    # Default shortcuts to display
    DEFAULT_SHORTCUTS = [
        ("F1", "Help"),
        ("F2", "Debate"),
        ("F3", "Providers"),
        ("F4", "Tools"),
        ("F5", "Knowledge"),
        ("F6", "Benchmark"),
        ("F7", "Settings"),
        ("Ctrl+Q", "Quit"),
    ]
    
    def __init__(
        self,
        shortcuts: list[tuple[str, str]] | None = None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.shortcuts = shortcuts or self.DEFAULT_SHORTCUTS
    
    def compose(self) -> ComposeResult:
        """Compose the footer layout."""
        with Horizontal(id="footer-container"):
            yield Static(self._format_shortcuts(), id="footer-shortcuts")
            yield Static(self.message, id="footer-message")
    
    def _format_shortcuts(self) -> str:
        """Format shortcuts for display with markup."""
        parts = []
        for key, action in self.shortcuts[:6]:  # Limit to 6 shortcuts
            parts.append(f"[bold #b833ff]{key}[/] {action}")
        return " │ ".join(parts)
    
    def watch_message(self, value: str) -> None:
        """Update message display when reactive changes."""
        try:
            message_widget = self.query_one("#footer-message", Static)
            message_widget.update(value)
        except Exception:
            pass
    
    def show_message(self, text: str, duration: float = 3.0) -> None:
        """
        Show a temporary message.
        
        Args:
            text: Message to display
            duration: How long to show (seconds)
        """
        self.message = text
        if duration > 0:
            self.set_timer(duration, self._clear_message)
    
    def _clear_message(self) -> None:
        """Clear the message."""
        self.message = ""
    
    def update_shortcuts(self, shortcuts: list[tuple[str, str]]) -> None:
        """Update the displayed shortcuts."""
        self.shortcuts = shortcuts
        try:
            shortcuts_widget = self.query_one("#footer-shortcuts", Static)
            shortcuts_widget.update(self._format_shortcuts())
        except Exception:
            pass
