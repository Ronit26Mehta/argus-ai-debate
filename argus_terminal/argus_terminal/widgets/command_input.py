"""
Command Input Widget - Bloomberg-style command line.

Features:
- Command history
- Auto-complete suggestions
- Quick command shortcuts
"""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static, Input
from textual.message import Message
from textual.reactive import reactive


class CommandInput(Static):
    """Bloomberg-style command input widget."""
    
    DEFAULT_CSS = """
    CommandInput {
        height: 3;
        width: 100%;
        background: #0d0d0d;
        padding: 0 1;
    }
    
    CommandInput #command-container {
        width: 100%;
        height: 100%;
        layout: horizontal;
    }
    
    CommandInput #command-prompt {
        width: auto;
        height: 1;
        margin-top: 1;
        margin-right: 1;
        color: #b833ff;
        text-style: bold;
    }
    
    CommandInput #command-field {
        width: 1fr;
        height: 1;
        margin-top: 1;
        background: #0a0a0a;
        border: none;
        color: #ffb000;
        padding: 0;
    }
    
    CommandInput #command-field:focus {
        border: none;
    }
    """
    
    class CommandSubmitted(Message):
        """Message emitted when a command is submitted."""
        
        def __init__(self, command: str) -> None:
            super().__init__()
            self.command = command
    
    # Reactive properties
    prompt: reactive[str] = reactive("ARGUS>")
    
    # Command history
    _history: list[str] = []
    _history_index: int = -1
    
    # Quick commands mapping
    QUICK_COMMANDS = {
        "d": "debate",
        "p": "providers",
        "t": "tools",
        "k": "knowledge",
        "b": "benchmark",
        "s": "settings",
        "h": "help",
        "q": "quit",
    }
    
    def __init__(
        self,
        prompt: str = "ARGUS>",
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.prompt = prompt
    
    def compose(self) -> ComposeResult:
        """Compose the command input layout."""
        with Horizontal(id="command-container"):
            yield Static(self.prompt, id="command-prompt")
            yield Input(
                placeholder="Enter command or press F1 for help...",
                id="command-field",
            )
    
    def watch_prompt(self, value: str) -> None:
        """Update prompt display when reactive changes."""
        try:
            prompt_widget = self.query_one("#command-prompt", Static)
            prompt_widget.update(value)
        except Exception:
            pass
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle command submission."""
        command = event.value.strip()
        if command:
            # Add to history
            if not self._history or self._history[-1] != command:
                self._history.append(command)
            self._history_index = -1
            
            # Expand quick commands
            expanded_command = self.QUICK_COMMANDS.get(command.lower(), command)
            
            # Clear input
            event.input.value = ""
            
            # Post message
            self.post_message(self.CommandSubmitted(expanded_command))
    
    def history_up(self) -> None:
        """Navigate to previous command in history."""
        if self._history:
            if self._history_index < len(self._history) - 1:
                self._history_index += 1
            try:
                input_widget = self.query_one("#command-field", Input)
                input_widget.value = self._history[-(self._history_index + 1)]
            except Exception:
                pass
    
    def history_down(self) -> None:
        """Navigate to next command in history."""
        if self._history_index > 0:
            self._history_index -= 1
            try:
                input_widget = self.query_one("#command-field", Input)
                input_widget.value = self._history[-(self._history_index + 1)]
            except Exception:
                pass
        elif self._history_index == 0:
            self._history_index = -1
            try:
                input_widget = self.query_one("#command-field", Input)
                input_widget.value = ""
            except Exception:
                pass
    
    def focus_input(self) -> None:
        """Focus the command input field."""
        try:
            input_widget = self.query_one("#command-field", Input)
            input_widget.focus()
        except Exception:
            pass
    
    def clear(self) -> None:
        """Clear the command input."""
        try:
            input_widget = self.query_one("#command-field", Input)
            input_widget.value = ""
        except Exception:
            pass
