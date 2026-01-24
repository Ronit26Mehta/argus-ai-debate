"""
Progress Panel Widget - Progress tracking with status.

Features:
- Progress bar
- Step display
- Time tracking
"""

from datetime import datetime
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Static, ProgressBar
from textual.reactive import reactive


class ProgressPanel(Vertical):
    """Progress tracking panel with bar and status."""
    
    DEFAULT_CSS = """
    ProgressPanel {
        width: 100%;
        height: 6;
        background: #0d0d0d;
        border: solid #665200;
        padding: 1;
    }
    
    ProgressPanel #progress-header {
        width: 100%;
        height: 1;
        layout: horizontal;
        margin-bottom: 1;
    }
    
    ProgressPanel #progress-title {
        width: 1fr;
        color: #ffcc33;
        text-style: bold;
    }
    
    ProgressPanel #progress-percent {
        width: auto;
        color: #ff8c00;
    }
    
    ProgressPanel #progress-bar {
        width: 100%;
        height: 1;
        margin-bottom: 1;
    }
    
    ProgressPanel #progress-status {
        width: 100%;
        height: 1;
        layout: horizontal;
    }
    
    ProgressPanel #progress-step {
        width: 1fr;
        color: #ffb000;
    }
    
    ProgressPanel #progress-time {
        width: auto;
        color: #806000;
    }
    """
    
    # Reactive properties
    progress: reactive[float] = reactive(0.0)
    status: reactive[str] = reactive("Waiting...")
    current_step: reactive[int] = reactive(0)
    total_steps: reactive[int] = reactive(1)
    
    def __init__(
        self,
        title: str = "Progress",
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.title = title
        self._start_time: datetime | None = None
    
    def compose(self) -> ComposeResult:
        """Compose the progress panel layout."""
        with Horizontal(id="progress-header"):
            yield Static(f"◈ {self.title}", id="progress-title")
            yield Static("0%", id="progress-percent")
        
        yield ProgressBar(total=100, id="progress-bar")
        
        with Horizontal(id="progress-status"):
            yield Static(self.status, id="progress-step")
            yield Static("", id="progress-time")
    
    def watch_progress(self, value: float) -> None:
        """Update progress display."""
        value = max(0.0, min(1.0, value))
        
        try:
            # Update percentage
            percent_widget = self.query_one("#progress-percent", Static)
            percent_widget.update(f"{value:.0%}")
            
            # Update progress bar
            bar = self.query_one("#progress-bar", ProgressBar)
            bar.progress = value * 100
        except Exception:
            pass
    
    def watch_status(self, value: str) -> None:
        """Update status display."""
        try:
            step_widget = self.query_one("#progress-step", Static)
            step_widget.update(f"Step {self.current_step}/{self.total_steps}: {value}")
        except Exception:
            pass
    
    def _update_time(self) -> None:
        """Update elapsed time display."""
        if self._start_time:
            elapsed = datetime.now() - self._start_time
            minutes = int(elapsed.total_seconds() // 60)
            seconds = int(elapsed.total_seconds() % 60)
            
            try:
                time_widget = self.query_one("#progress-time", Static)
                time_widget.update(f"⏱ {minutes:02d}:{seconds:02d}")
            except Exception:
                pass
    
    def start(self, total_steps: int = 1) -> None:
        """Start progress tracking."""
        self._start_time = datetime.now()
        self.total_steps = total_steps
        self.current_step = 0
        self.progress = 0.0
        self.status = "Starting..."
        
        # Start timer update
        self.set_interval(1.0, self._update_time, name="progress_timer")
    
    def update(
        self,
        step: int | None = None,
        status: str | None = None,
        progress: float | None = None,
    ) -> None:
        """Update progress state."""
        if step is not None:
            self.current_step = step
        if status is not None:
            self.status = status
        if progress is not None:
            self.progress = progress
        elif step is not None and self.total_steps > 0:
            self.progress = step / self.total_steps
    
    def complete(self, status: str = "Complete!") -> None:
        """Mark as complete."""
        self.progress = 1.0
        self.current_step = self.total_steps
        self.status = status
        
        # Stop timer
        try:
            self.remove_timer("progress_timer")
        except Exception:
            pass
    
    def reset(self) -> None:
        """Reset progress."""
        self._start_time = None
        self.progress = 0.0
        self.current_step = 0
        self.status = "Waiting..."
        
        try:
            self.remove_timer("progress_timer")
            time_widget = self.query_one("#progress-time", Static)
            time_widget.update("")
        except Exception:
            pass
