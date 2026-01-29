"""
Collaboration Room Screen for BioSage Terminal.
Multi-disciplinary case discussion and decision making.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import (
    Header, Footer, Static, Button, Input, Label,
    DataTable, TabbedContent, TabPane, TextArea,
)
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from rich.text import Text
from rich.panel import Panel
from rich.table import Table

from biosage_terminal.storage import get_data_store
from biosage_terminal.theme import COLORS


class ParticipantCard(Static):
    """A card showing a collaboration participant."""
    
    def __init__(
        self,
        name: str,
        role: str,
        status: str = "online",
        last_seen: str = "now",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.participant_name = name
        self.role = role
        self.status = status
        self.last_seen = last_seen
        
    def compose(self) -> ComposeResult:
        status_color = {
            "online": COLORS["success"],
            "away": COLORS["warning"],
            "offline": COLORS["text_muted"],
        }.get(self.status, COLORS["text_muted"])
        
        initials = "".join(n[0] for n in self.participant_name.split()[:2])
        
        content = f"""[bold]{self.participant_name}[/bold]
[dim]{self.role}[/dim]
[{status_color}]* {self.status.title()}[/{status_color}] [dim]{self.last_seen}[/dim]"""
        
        yield Static(content, classes="participant-info")


class ChatMessage(Static):
    """A chat message widget."""
    
    def __init__(
        self,
        sender: str,
        message: str,
        timestamp: str,
        is_ai: bool = False,
        reactions: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sender = sender
        self.message = message
        self.timestamp = timestamp
        self.is_ai = is_ai
        self.reactions = reactions or {"thumbs_up": 0, "thumbs_down": 0}
        
    def compose(self) -> ComposeResult:
        sender_color = COLORS["accent_purple"] if self.is_ai else COLORS["info"]
        badge = " [AI Update]" if self.is_ai else ""
        
        content = f"""[{sender_color}][bold]{self.sender}[/bold][/{sender_color}]{badge} [dim]{self.timestamp}[/dim]
{self.message}
[dim]+{self.reactions.get('thumbs_up', 0)} -{self.reactions.get('thumbs_down', 0)}[/dim]"""
        
        yield Static(content, classes="chat-message")


class DiscussionCard(Static):
    """A discussion thread card."""
    
    def __init__(
        self,
        title: str,
        participants: int,
        messages: int,
        last_activity: str,
        status: str = "active",
        priority: str = "medium",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.title = title
        self.participants_count = participants
        self.messages_count = messages
        self.last_activity = last_activity
        self.status = status
        self.priority = priority
        
    def compose(self) -> ComposeResult:
        priority_color = {
            "high": COLORS["error"],
            "medium": COLORS["warning"],
            "low": COLORS["text_muted"],
        }.get(self.priority, COLORS["text_muted"])
        
        status_indicator = "[*]" if self.status == "active" else "[ ]"
        
        content = f"""{status_indicator} [{priority_color}][{self.priority.upper()}][/{priority_color}]
[bold]{self.title[:40]}{'...' if len(self.title) > 40 else ''}[/bold]
[dim]{self.participants_count} participants | {self.messages_count} messages[/dim]
[dim]Last: {self.last_activity}[/dim]"""
        
        yield Static(content, classes="discussion-card")


class PinnedItem(Static):
    """A pinned item widget."""
    
    def __init__(
        self,
        title: str,
        content: str,
        item_type: str,
        pinned_by: str,
        timestamp: str,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.title = title
        self.item_content = content
        self.item_type = item_type
        self.pinned_by = pinned_by
        self.timestamp = timestamp
        
    def compose(self) -> ComposeResult:
        type_color = {
            "case": COLORS["info"],
            "decision": COLORS["success"],
            "note": COLORS["warning"],
        }.get(self.item_type, COLORS["text_muted"])
        
        display = f"""[{type_color}][{self.item_type.upper()}][/{type_color}] [bold]{self.title}[/bold]
[dim]{self.item_content[:60]}{'...' if len(self.item_content) > 60 else ''}[/dim]
[dim]Pinned by {self.pinned_by} | {self.timestamp}[/dim]"""
        
        yield Static(display, classes="pinned-item")


class CollaborationRoomScreen(Screen):
    """Multi-disciplinary collaboration room."""
    
    BINDINGS = [
        ("escape", "go_back", "Back"),
        ("d", "goto_dashboard", "Dashboard"),
        ("s", "send_message", "Send"),
        ("v", "cast_vote", "Vote"),
    ]
    
    CSS = """
    CollaborationRoomScreen {
        background: #0F172A;
    }
    
    .collab-header {
        dock: top;
        height: 3;
        background: #3B82F6;
        color: #F1F5F9;
        padding: 0 2;
    }
    
    .main-grid {
        layout: grid;
        grid-size: 4;
        grid-columns: 1fr 2fr 1fr;
        padding: 1;
        height: 100%;
    }
    
    .sidebar-left {
        column-span: 1;
        height: 100%;
        padding: 0 1;
    }
    
    .main-chat {
        column-span: 2;
        height: 100%;
        padding: 0 1;
    }
    
    .sidebar-right {
        column-span: 1;
        height: 100%;
        padding: 0 1;
    }
    
    .section-box {
        border: solid #3B82F6;
        padding: 1;
        margin-bottom: 1;
    }
    
    .section-title {
        text-style: bold;
        color: #8B5CF6;
        margin-bottom: 1;
    }
    
    .participant-info {
        padding: 0 0 1 0;
        border: none none solid none #334155;
    }
    
    .chat-message {
        padding: 1;
        margin-bottom: 1;
        background: #1E293B;
    }
    
    .discussion-card {
        padding: 1;
        margin-bottom: 1;
        background: #1E293B;
    }
    
    .pinned-item {
        padding: 1;
        margin-bottom: 1;
        background: #1E293B;
    }
    
    .message-input {
        height: 3;
        margin-top: 1;
    }
    
    .vote-option {
        height: 2;
        margin-bottom: 1;
    }
    
    .vote-bar {
        background: #1E293B;
        height: 1;
    }
    
    .controls-row {
        height: 3;
        dock: bottom;
    }
    """
    
    def __init__(self, case_id: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.case_id = case_id
        self.store = get_data_store()
        self.message_input = ""
        
        # Sample data
        self.participants = [
            {"name": "Dr. Sarah Chen", "role": "Rheumatologist", "status": "online", "last_seen": "now"},
            {"name": "Dr. Michael Kim", "role": "Infectious Disease", "status": "online", "last_seen": "2 min ago"},
            {"name": "Dr. Lisa Wang", "role": "Cardiology", "status": "away", "last_seen": "15 min ago"},
            {"name": "Dr. James Brown", "role": "Neurology", "status": "offline", "last_seen": "1 hour ago"},
        ]
        
        self.discussions = [
            {"title": "SLE vs MCTD Differential", "participants": 3, "messages": 12, "last_activity": "5 min ago", "status": "active", "priority": "high"},
            {"title": "Weekly Case Review", "participants": 8, "messages": 47, "last_activity": "1 hour ago", "status": "scheduled", "priority": "medium"},
            {"title": "AI Model Performance", "participants": 5, "messages": 23, "last_activity": "3 hours ago", "status": "concluded", "priority": "low"},
        ]
        
        self.chat_messages = [
            {"sender": "Dr. Sarah Chen", "message": "Looking at the ANA pattern, the speckled distribution with anti-dsDNA positivity strongly suggests SLE", "timestamp": "14:32", "is_ai": False, "reactions": {"thumbs_up": 2, "thumbs_down": 0}},
            {"sender": "Dr. Michael Kim", "message": "I agree, but we should consider the joint distribution. The lack of erosive changes makes me think about other CTDs", "timestamp": "14:35", "is_ai": False, "reactions": {"thumbs_up": 1, "thumbs_down": 0}},
            {"sender": "AI Agent", "message": "New evidence from cardiology specialist: Echo shows mild pericardial effusion, confidence for SLE increased to 89%", "timestamp": "14:38", "is_ai": True, "reactions": {"thumbs_up": 3, "thumbs_down": 0}},
        ]
        
        self.pinned_items = [
            {"title": "Patient Summary", "content": "34F with fever, joint pain, malar rash. ANA 1:640, anti-dsDNA +", "type": "case", "pinned_by": "Dr. Sarah Chen", "timestamp": "2 hours ago"},
            {"title": "Consensus: Order Anti-Sm and Anti-RNP", "content": "Voted to proceed with additional autoantibody testing", "type": "decision", "pinned_by": "System", "timestamp": "1 hour ago"},
        ]
        
        self.vote_options = [
            {"label": "Order Anti-Sm & Anti-RNP", "votes": 3, "percentage": 75},
            {"label": "Start Empiric Treatment", "votes": 1, "percentage": 25},
        ]
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        online_count = sum(1 for p in self.participants if p["status"] == "online")
        
        with Container(classes="collab-header"):
            yield Static(
                f"[bold]Collaboration Room[/bold] | Case: {self.case_id or 'BSG-2024-001'} | "
                f"[green]{online_count} Online[/green] | [d] Dashboard [s] Send [v] Vote [ESC] Back"
            )
        
        with Container(classes="main-grid"):
            # Left sidebar - Participants and Discussions
            with ScrollableContainer(classes="sidebar-left"):
                with Container(classes="section-box"):
                    yield Static(f"[bold]Participants ({len(self.participants)})[/bold]", classes="section-title")
                    for p in self.participants:
                        yield ParticipantCard(
                            name=p["name"],
                            role=p["role"],
                            status=p["status"],
                            last_seen=p["last_seen"]
                        )
                
                with Container(classes="section-box"):
                    yield Static("[bold]Discussions[/bold]", classes="section-title")
                    for d in self.discussions:
                        yield DiscussionCard(
                            title=d["title"],
                            participants=d["participants"],
                            messages=d["messages"],
                            last_activity=d["last_activity"],
                            status=d["status"],
                            priority=d["priority"]
                        )
            
            # Main chat area
            with Container(classes="main-chat"):
                with Container(classes="section-box"):
                    yield Static("[bold]SLE vs MCTD Differential Discussion[/bold]", classes="section-title")
                    
                    with ScrollableContainer(id="chat-scroll"):
                        for msg in self.chat_messages:
                            yield ChatMessage(
                                sender=msg["sender"],
                                message=msg["message"],
                                timestamp=msg["timestamp"],
                                is_ai=msg["is_ai"],
                                reactions=msg["reactions"]
                            )
                    
                    yield Input(
                        placeholder="Type your message...",
                        id="message-input",
                        classes="message-input"
                    )
                    
                    with Horizontal(classes="controls-row"):
                        yield Button("Send", id="btn-send", variant="primary")
                        yield Button("Share", id="btn-share", variant="default")
                        yield Button("Summary", id="btn-summary", variant="default")
            
            # Right sidebar - Pinned items and voting
            with ScrollableContainer(classes="sidebar-right"):
                with Container(classes="section-box"):
                    yield Static("[bold]Pinned Items[/bold]", classes="section-title")
                    for item in self.pinned_items:
                        yield PinnedItem(
                            title=item["title"],
                            content=item["content"],
                            item_type=item["type"],
                            pinned_by=item["pinned_by"],
                            timestamp=item["timestamp"]
                        )
                
                with Container(classes="section-box"):
                    yield Static("[bold]Current Vote[/bold]", classes="section-title")
                    yield Static("[dim]Next steps for case[/dim]")
                    
                    for opt in self.vote_options:
                        bar_filled = int(opt["percentage"] / 5)
                        bar = "[green]" + "#" * bar_filled + "[/green]" + "-" * (20 - bar_filled)
                        yield Static(f"{opt['label']}\n{bar} {opt['votes']} votes ({opt['percentage']}%)", classes="vote-option")
                    
                    yield Button("Cast Your Vote", id="btn-vote", variant="primary")
                
                with Container(classes="section-box"):
                    yield Static("[bold]Quick Actions[/bold]", classes="section-title")
                    yield Button("Export Discussion", id="btn-export", variant="default")
                    yield Button("Invite Specialist", id="btn-invite", variant="default")
                    yield Button("Schedule Follow-up", id="btn-schedule", variant="default")
        
        yield Footer()
    
    def action_go_back(self) -> None:
        """Go back to previous screen."""
        self.app.pop_screen()
    
    def action_goto_dashboard(self) -> None:
        """Navigate to dashboard."""
        self.app.switch_screen("dashboard")
    
    def action_send_message(self) -> None:
        """Send a message."""
        input_widget = self.query_one("#message-input", Input)
        message = input_widget.value.strip()
        if message:
            # Log the message
            self.store.audit.log_event(
                event_type="collaboration",
                user="current_user",
                action="send_message",
                details={"case_id": self.case_id, "message": message}
            )
            input_widget.value = ""
            self.notify("Message sent", severity="information")
    
    def action_cast_vote(self) -> None:
        """Cast a vote."""
        self.notify("Vote cast successfully", severity="information")
        self.store.audit.log_event(
            event_type="collaboration",
            user="current_user",
            action="cast_vote",
            details={"case_id": self.case_id}
        )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-send":
            self.action_send_message()
        elif button_id == "btn-vote":
            self.action_cast_vote()
        elif button_id == "btn-share":
            self.notify("Discussion shared", severity="information")
        elif button_id == "btn-summary":
            self.notify("Generating summary...", severity="information")
        elif button_id == "btn-export":
            self.notify("Discussion exported to reports folder", severity="information")
        elif button_id == "btn-invite":
            self.notify("Specialist invitation sent", severity="information")
        elif button_id == "btn-schedule":
            self.notify("Follow-up scheduled", severity="information")
