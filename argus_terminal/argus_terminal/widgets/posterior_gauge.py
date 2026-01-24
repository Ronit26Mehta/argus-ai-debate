"""
Posterior Gauge Widget - Visual probability display.

Features:
- Horizontal bar gauge
- Color gradient based on value
- Numeric display
"""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static
from textual.reactive import reactive


class PosteriorGaugeWidget(Vertical):
    """Visual gauge for displaying posterior probability."""
    
    DEFAULT_CSS = """
    PosteriorGaugeWidget {
        width: 100%;
        height: 5;
        background: #0d0d0d;
        border: solid #665200;
        padding: 1;
    }
    
    PosteriorGaugeWidget #gauge-header {
        width: 100%;
        height: 1;
        layout: horizontal;
        margin-bottom: 1;
    }
    
    PosteriorGaugeWidget #gauge-label {
        width: 1fr;
        color: #ffcc33;
        text-style: bold;
    }
    
    PosteriorGaugeWidget #gauge-value {
        width: auto;
        text-style: bold;
    }
    
    PosteriorGaugeWidget #gauge-bar-container {
        width: 100%;
        height: 1;
    }
    
    PosteriorGaugeWidget #gauge-bar {
        width: 100%;
        height: 1;
        background: #1a1400;
    }
    """
    
    # Reactive value
    value: reactive[float] = reactive(0.5)
    
    def __init__(
        self,
        value: float = 0.5,
        label: str = "Posterior Probability",
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.value = value
        self.label = label
    
    def compose(self) -> ComposeResult:
        """Compose the gauge layout."""
        with Horizontal(id="gauge-header"):
            yield Static(f"◈ {self.label}", id="gauge-label")
            yield Static(self._format_value(), id="gauge-value")
        
        yield Static(self._render_bar(), id="gauge-bar")
    
    def _format_value(self) -> str:
        """Format the value with color."""
        color = self._get_color()
        return f"[bold {color}]{self.value:.1%}[/]"
    
    def _get_color(self) -> str:
        """Get color based on value."""
        if self.value < 0.3:
            return "#ff3333"  # Red - low
        elif self.value < 0.7:
            return "#ffcc00"  # Yellow - medium
        else:
            return "#33ff33"  # Green - high
    
    def _render_bar(self) -> str:
        """Render the gauge bar."""
        width = 50  # Bar width in characters
        filled = int(self.value * width)
        empty = width - filled
        
        color = self._get_color()
        
        # Build bar with gradient effect
        bar_chars = "█" * filled
        empty_chars = "░" * empty
        
        return f"[{color}]{bar_chars}[/][#333333]{empty_chars}[/]"
    
    def watch_value(self, value: float) -> None:
        """Update display when value changes."""
        # Clamp value between 0 and 1
        value = max(0.0, min(1.0, value))
        
        try:
            value_widget = self.query_one("#gauge-value", Static)
            value_widget.update(self._format_value())
            
            bar_widget = self.query_one("#gauge-bar", Static)
            bar_widget.update(self._render_bar())
        except Exception:
            pass
    
    def set_value(self, value: float) -> None:
        """Set the gauge value."""
        self.value = max(0.0, min(1.0, value))


class CompactPosteriorGauge(Static):
    """Compact single-line gauge for inline display."""
    
    DEFAULT_CSS = """
    CompactPosteriorGauge {
        width: auto;
        height: 1;
    }
    """
    
    value: reactive[float] = reactive(0.5)
    
    def __init__(
        self,
        value: float = 0.5,
        width: int = 20,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.value = value
        self.bar_width = width
    
    def render(self) -> str:
        """Render the compact gauge."""
        filled = int(self.value * self.bar_width)
        empty = self.bar_width - filled
        
        if self.value < 0.3:
            color = "#ff3333"
        elif self.value < 0.7:
            color = "#ffcc00"
        else:
            color = "#33ff33"
        
        bar = f"[{color}]{'█' * filled}[/][#333333]{'░' * empty}[/]"
        return f"[{bar}] [{color}]{self.value:.0%}[/]"
