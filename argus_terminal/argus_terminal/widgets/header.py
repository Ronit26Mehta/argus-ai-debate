"""
Custom Header Widget - Bloomberg-style top bar with logo and status.

Features:
- ASCII ARGUS logo
- System status indicators
- Real-time clock
- Active session info
"""

from datetime import datetime
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static
from textual.reactive import reactive


# ASCII Art Logo
ARGUS_LOGO = """╔═╗╦═╗╔═╗╦ ╦╔═╗
╠═╣╠╦╝║ ╦║ ║╚═╗
╩ ╩╩╚═╚═╝╚═╝╚═╝"""

ARGUS_LOGO_COMPACT = "◈ ARGUS"


class HeaderWidget(Static):
    """Bloomberg-style header widget with logo, status, and clock."""
    
    DEFAULT_CSS = """
    HeaderWidget {
        dock: top;
        height: 3;
        width: 100%;
        background: #0d0d0d;
        border-bottom: double #665200;
        padding: 0 2;
    }
    
    HeaderWidget #header-container {
        width: 100%;
        height: 100%;
        layout: horizontal;
    }
    
    HeaderWidget #header-logo {
        width: auto;
        height: 1;
        margin-top: 1;
        color: #ffcc33;
    }
    
    HeaderWidget #header-version {
        width: auto;
        height: 1;
        margin-top: 1;
        margin-left: 2;
        color: #ff8c00;
    }
    
    HeaderWidget #header-status {
        width: 1fr;
        height: 1;
        margin-top: 1;
        content-align: center middle;
        color: #ffb000;
    }
    
    HeaderWidget #header-session {
        width: auto;
        height: 1;
        margin-top: 1;
        margin-right: 2;
        color: #9400d3;
    }
    
    HeaderWidget #header-time {
        width: 22;
        height: 1;
        margin-top: 1;
        content-align: right middle;
        color: #ff8c00;
    }
    """
    
    # Reactive properties
    status_text: reactive[str] = reactive("● READY")
    session_name: reactive[str] = reactive("")
    
    def __init__(
        self,
        version: str = "3.1.0",
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.version = version
        
    def compose(self) -> ComposeResult:
        """Compose the header layout."""
        with Horizontal(id="header-container"):
            yield Static(ARGUS_LOGO_COMPACT, id="header-logo")
            yield Static(f"v{self.version}", id="header-version")
            yield Static(self.status_text, id="header-status")
            yield Static(self.session_name, id="header-session")
            yield Static(self._get_time_string(), id="header-time")
    
    def _get_time_string(self) -> str:
        """Get formatted time string."""
        now = datetime.now()
        return now.strftime("▐ %Y-%m-%d │ %H:%M:%S")
    
    def on_mount(self) -> None:
        """Start the clock update timer."""
        self.set_interval(1.0, self._update_time)
    
    def _update_time(self) -> None:
        """Update the time display."""
        time_widget = self.query_one("#header-time", Static)
        time_widget.update(self._get_time_string())
    
    def watch_status_text(self, value: str) -> None:
        """Update status display when reactive changes."""
        try:
            status_widget = self.query_one("#header-status", Static)
            status_widget.update(value)
        except Exception:
            pass
    
    def watch_session_name(self, value: str) -> None:
        """Update session display when reactive changes."""
        try:
            session_widget = self.query_one("#header-session", Static)
            session_widget.update(f"◇ {value}" if value else "")
        except Exception:
            pass
    
    def set_status(self, status: str, status_type: str = "normal") -> None:
        """
        Set the status text with indicator.
        
        Args:
            status: Status message
            status_type: One of 'normal', 'active', 'idle', 'error'
        """
        indicators = {
            "normal": "●",
            "active": "◉",
            "idle": "○",
            "error": "✖",
            "working": "◐",
        }
        indicator = indicators.get(status_type, "●")
        self.status_text = f"{indicator} {status}"
