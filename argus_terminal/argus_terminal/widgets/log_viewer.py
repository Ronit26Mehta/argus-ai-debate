"""
Log Viewer Widget - Real-time log output display.

Features:
- Scrollable log view
- Color-coded log levels
- Auto-scroll to bottom
"""

from datetime import datetime
from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.widgets import Static
from textual.reactive import reactive
from dataclasses import dataclass
from enum import Enum


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


@dataclass
class LogEntry:
    """Log entry data."""
    timestamp: datetime
    level: LogLevel
    message: str
    source: str = ""


class LogViewerWidget(Vertical):
    """Scrollable log viewer with colored output."""
    
    DEFAULT_CSS = """
    LogViewerWidget {
        width: 100%;
        height: 100%;
        background: #0a0a0a;
        border: solid #665200;
    }
    
    LogViewerWidget #log-title {
        width: 100%;
        height: 1;
        color: #ffcc33;
        text-style: bold;
        background: #0d0d0d;
        padding: 0 1;
        border-bottom: solid #4d3800;
    }
    
    LogViewerWidget #log-container {
        width: 100%;
        height: 1fr;
        padding: 0 1;
    }
    
    LogViewerWidget #log-content {
        width: 100%;
        height: auto;
    }
    """
    
    # Log level colors
    LEVEL_COLORS = {
        LogLevel.DEBUG: "#806000",
        LogLevel.INFO: "#ff8c00",
        LogLevel.WARNING: "#ffcc00",
        LogLevel.ERROR: "#ff3333",
        LogLevel.SUCCESS: "#33ff33",
    }
    
    # Log level icons
    LEVEL_ICONS = {
        LogLevel.DEBUG: "○",
        LogLevel.INFO: "●",
        LogLevel.WARNING: "▲",
        LogLevel.ERROR: "✖",
        LogLevel.SUCCESS: "✔",
    }
    
    def __init__(
        self,
        title: str = "◈ ACTIVITY LOG",
        max_entries: int = 100,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.title = title
        self.max_entries = max_entries
        self._entries: list[LogEntry] = []
    
    def compose(self) -> ComposeResult:
        """Compose the log viewer layout."""
        yield Static(self.title, id="log-title")
        with ScrollableContainer(id="log-container"):
            yield Static(id="log-content")
    
    def on_mount(self) -> None:
        """Initialize the log."""
        self._render_log()
    
    def _render_log(self) -> None:
        """Render the log entries."""
        content = self.query_one("#log-content", Static)
        
        if not self._entries:
            content.update("[dim]No log entries yet.[/dim]")
            return
        
        lines = []
        for entry in self._entries:
            lines.append(self._format_entry(entry))
        
        content.update("\n".join(lines))
        
        # Auto-scroll to bottom
        try:
            container = self.query_one("#log-container", ScrollableContainer)
            container.scroll_end(animate=False)
        except Exception:
            pass
    
    def _format_entry(self, entry: LogEntry) -> str:
        """Format a log entry for display."""
        color = self.LEVEL_COLORS[entry.level]
        icon = self.LEVEL_ICONS[entry.level]
        
        timestamp = entry.timestamp.strftime("%H:%M:%S")
        source = f"[{entry.source}] " if entry.source else ""
        
        return (
            f"[#b833ff]{timestamp}[/] "
            f"[{color}]{icon}[/] "
            f"[#806000]{source}[/]"
            f"[{color}]{entry.message}[/]"
        )
    
    def log(
        self,
        message: str,
        level: LogLevel = LogLevel.INFO,
        source: str = "",
    ) -> None:
        """Add a log entry."""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            source=source,
        )
        
        self._entries.append(entry)
        
        # Trim if exceeding max entries
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]
        
        self._render_log()
    
    def info(self, message: str, source: str = "") -> None:
        """Log an info message."""
        self.log(message, LogLevel.INFO, source)
    
    def debug(self, message: str, source: str = "") -> None:
        """Log a debug message."""
        self.log(message, LogLevel.DEBUG, source)
    
    def warning(self, message: str, source: str = "") -> None:
        """Log a warning message."""
        self.log(message, LogLevel.WARNING, source)
    
    def error(self, message: str, source: str = "") -> None:
        """Log an error message."""
        self.log(message, LogLevel.ERROR, source)
    
    def success(self, message: str, source: str = "") -> None:
        """Log a success message."""
        self.log(message, LogLevel.SUCCESS, source)
    
    def clear(self) -> None:
        """Clear all log entries."""
        self._entries = []
        self._render_log()
