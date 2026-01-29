"""
Visual Diagnosis Screen for BioSage Terminal.
AI-powered image analysis for skin lesions and medical imaging.
Uses Gemini Vision API for analysis.
"""

from __future__ import annotations

import os
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import (
    Header, Footer, Static, Button, Input, Label,
    ProgressBar, TabbedContent, TabPane,
)
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from rich.text import Text
from rich.panel import Panel
from rich.progress_bar import ProgressBar as RichProgressBar

from biosage_terminal.storage import get_data_store
from biosage_terminal.theme import COLORS


class AnalysisResult(Static):
    """Display an analysis result for a disease."""
    
    def __init__(
        self,
        disease: str,
        score: float,
        highlights: str = "",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.disease = disease
        self.score = score
        self.highlights = highlights
        
    def compose(self) -> ComposeResult:
        # Determine color based on score
        if self.score > 60:
            score_color = COLORS["error"]
        elif self.score > 30:
            score_color = COLORS["warning"]
        else:
            score_color = COLORS["success"]
        
        bar_filled = int(self.score / 5)
        bar_empty = 20 - bar_filled
        bar = f"[{score_color}]" + "#" * bar_filled + f"[/{score_color}]" + "-" * bar_empty
        
        content = f"""[bold]{self.disease}[/bold] [{score_color}]{self.score:.1f}%[/{score_color}]
{bar}
[dim]{self.highlights}[/dim]"""
        
        yield Static(content, classes="analysis-result")


class ChatBubble(Static):
    """A chat bubble for the visual diagnosis conversation."""
    
    def __init__(
        self,
        sender: str,
        content: str,
        is_user: bool = False,
        is_image: bool = False,
        image_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sender = sender
        self.bubble_content = content
        self.is_user = is_user
        self.is_image = is_image
        self.image_path = image_path
        
    def compose(self) -> ComposeResult:
        sender_color = COLORS["info"] if self.is_user else COLORS["accent_purple"]
        bg_class = "user-bubble" if self.is_user else "ai-bubble"
        
        if self.is_image and self.image_path:
            content = f"""[{sender_color}][bold]{self.sender}[/bold][/{sender_color}]
[dim]Image: {self.image_path}[/dim]
[Image uploaded for analysis]"""
        else:
            content = f"""[{sender_color}][bold]{self.sender}[/bold][/{sender_color}]
{self.bubble_content}"""
        
        yield Static(content, classes=f"chat-bubble {bg_class}")


class VisualDiagnosisScreen(Screen):
    """Visual diagnosis using AI image analysis."""
    
    BINDINGS = [
        ("escape", "go_back", "Back"),
        ("d", "goto_dashboard", "Dashboard"),
        ("u", "upload_image", "Upload Image"),
        ("a", "analyze", "Analyze"),
        ("c", "clear_chat", "Clear"),
    ]
    
    CSS = """
    VisualDiagnosisScreen {
        background: #0F172A;
    }
    
    .visual-header {
        dock: top;
        height: 3;
        background: #3B82F6;
        color: #F1F5F9;
        padding: 0 2;
    }
    
    .main-container {
        layout: grid;
        grid-size: 2;
        grid-columns: 2fr 1fr;
        padding: 1;
        height: 100%;
    }
    
    .chat-container {
        height: 100%;
        padding: 0 1;
    }
    
    .results-container {
        height: 100%;
        padding: 0 1;
    }
    
    .chat-scroll {
        height: 80%;
        border: solid #3B82F6;
        padding: 1;
    }
    
    .input-area {
        height: 8;
        padding: 1;
        background: #1E293B;
    }
    
    .chat-bubble {
        padding: 1;
        margin: 1 0;
    }
    
    .user-bubble {
        background: #0EA5E9;
        text-align: right;
    }
    
    .ai-bubble {
        background: #1E293B;
    }
    
    .analysis-result {
        padding: 1;
        margin: 1 0;
        background: #1E293B;
        border: solid #3B82F6;
    }
    
    .results-section {
        border: solid #3B82F6;
        padding: 1;
        margin-bottom: 1;
    }
    
    .section-title {
        text-style: bold;
        color: #8B5CF6;
        margin-bottom: 1;
    }
    
    .file-info {
        height: 3;
        padding: 1;
        background: #1E293B;
        margin-bottom: 1;
    }
    
    .status-bar {
        height: 2;
        padding: 0 1;
        background: #1E293B;
    }
    
    .loading-indicator {
        color: #8B5CF6;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.store = get_data_store()
        self.selected_file: Optional[str] = None
        self.is_analyzing = False
        
        # Chat messages
        self.messages: list[dict] = []
        
        # Analysis results
        self.current_analysis: list[dict] = []
        self.summary: str = ""
        
        # Sample analysis for demo
        self.demo_analysis = [
            {"disease": "Systemic Lupus Erythematosus", "score": 78.5, "highlights": "Malar rash pattern consistent with butterfly distribution. Photosensitivity noted."},
            {"disease": "Dermatomyositis", "score": 45.2, "highlights": "Heliotrope discoloration around eyes. Gottron's papules not clearly visible."},
            {"disease": "Psoriasis", "score": 22.8, "highlights": "Some scaling present but pattern not typical. No nail involvement noted."},
            {"disease": "Contact Dermatitis", "score": 15.3, "highlights": "Distribution not consistent with contact pattern. No clear irritant exposure."},
        ]
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container(classes="visual-header"):
            yield Static(
                "[bold]Visual Diagnosis Assistant[/bold] | Powered by Gemini Vision | "
                "[u] Upload [a] Analyze [c] Clear [ESC] Back"
            )
        
        with Container(classes="main-container"):
            # Left side - Chat interface
            with Container(classes="chat-container"):
                yield Static("[bold]Analysis Chat[/bold]", classes="section-title")
                
                with ScrollableContainer(id="chat-scroll", classes="chat-scroll"):
                    if not self.messages:
                        yield Static(
                            "[dim]Upload a medical image to begin analysis.\n\n"
                            "Supported formats: JPG, PNG, BMP\n"
                            "Recommended: Clear, well-lit images of skin lesions[/dim]",
                            id="welcome-msg"
                        )
                
                with Container(classes="input-area"):
                    with Container(classes="file-info"):
                        yield Static(
                            f"Selected: {self.selected_file or 'No file selected'}",
                            id="file-display"
                        )
                    
                    with Horizontal():
                        yield Button("Choose Image", id="btn-upload", variant="default")
                        yield Button("Analyze", id="btn-analyze", variant="primary")
                        yield Button("Clear", id="btn-clear", variant="warning")
                    
                    yield Static("", id="status-msg", classes="status-bar")
            
            # Right side - Analysis results
            with ScrollableContainer(classes="results-container"):
                yield Static("[bold]Analysis Results[/bold]", classes="section-title")
                
                with Container(classes="results-section", id="results-section"):
                    yield Static(
                        "[dim]Analysis results will appear here after processing an image.[/dim]",
                        id="results-placeholder"
                    )
                    
                    # Summary section
                    yield Static("", id="summary-display")
                    
                    # Detailed analysis
                    with Container(id="analysis-container"):
                        pass
                    
                    # D3-style chart placeholder (text-based)
                    with Container(id="chart-container"):
                        pass
        
        yield Footer()
    
    def action_go_back(self) -> None:
        """Go back to previous screen."""
        self.app.pop_screen()
    
    def action_goto_dashboard(self) -> None:
        """Navigate to dashboard."""
        self.app.switch_screen("dashboard")
    
    def action_upload_image(self) -> None:
        """Simulate image upload."""
        # In a real TUI, we'd use a file picker
        # For now, we simulate with a demo file
        self.selected_file = "skin_lesion_sample.jpg"
        
        file_display = self.query_one("#file-display", Static)
        file_display.update(f"Selected: {self.selected_file}")
        
        # Add user message
        self._add_message("User", f"Uploaded image: {self.selected_file}", is_user=True, is_image=True)
        
        self.notify("Image selected. Press [a] to analyze.", severity="information")
    
    def action_analyze(self) -> None:
        """Analyze the selected image."""
        if not self.selected_file:
            self.notify("Please upload an image first", severity="warning")
            return
        
        if self.is_analyzing:
            self.notify("Analysis already in progress", severity="warning")
            return
        
        self.is_analyzing = True
        status = self.query_one("#status-msg", Static)
        status.update("[bold cyan]Analyzing image, please wait...[/bold cyan]")
        
        # Simulate analysis with demo data
        self._run_demo_analysis()
    
    def _run_demo_analysis(self) -> None:
        """Run demo analysis (simulates Gemini Vision API call)."""
        # Add AI response
        summary = (
            "Analysis of the skin lesion image reveals a pattern highly suggestive of "
            "autoimmune etiology. The malar distribution and photosensitive appearance "
            "are most consistent with Systemic Lupus Erythematosus. Additional serological "
            "testing (ANA, anti-dsDNA) is recommended for confirmation."
        )
        
        self._add_message("AI Assistant", summary, is_user=False)
        
        # Update results display
        self._display_results(summary, self.demo_analysis)
        
        # Log audit event
        self.store.audit.log_event(
            event_type="visual_diagnosis",
            user="current_user",
            action="analyze_image",
            details={"file": self.selected_file, "top_diagnosis": self.demo_analysis[0]["disease"]}
        )
        
        self.is_analyzing = False
        status = self.query_one("#status-msg", Static)
        status.update("[green]Analysis complete[/green]")
    
    def _add_message(
        self,
        sender: str,
        content: str,
        is_user: bool = False,
        is_image: bool = False
    ) -> None:
        """Add a message to the chat."""
        self.messages.append({
            "sender": sender,
            "content": content,
            "is_user": is_user,
            "is_image": is_image,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        
        # Update chat display
        chat_scroll = self.query_one("#chat-scroll", ScrollableContainer)
        
        # Remove welcome message if present
        try:
            welcome = self.query_one("#welcome-msg", Static)
            welcome.remove()
        except Exception:
            pass
        
        # Add new message bubble
        bubble = ChatBubble(
            sender=sender,
            content=content,
            is_user=is_user,
            is_image=is_image,
            image_path=self.selected_file if is_image else None
        )
        chat_scroll.mount(bubble)
    
    def _display_results(self, summary: str, analysis: list[dict]) -> None:
        """Display analysis results."""
        # Update summary
        summary_display = self.query_one("#summary-display", Static)
        summary_display.update(f"[bold]Summary[/bold]\n{summary}")
        
        # Remove placeholder
        try:
            placeholder = self.query_one("#results-placeholder", Static)
            placeholder.update("")
        except Exception:
            pass
        
        # Add analysis results
        analysis_container = self.query_one("#analysis-container", Container)
        for item in analysis:
            result = AnalysisResult(
                disease=item["disease"],
                score=item["score"],
                highlights=item["highlights"]
            )
            analysis_container.mount(result)
        
        # Add text-based chart
        chart_container = self.query_one("#chart-container", Container)
        chart_content = "[bold]Confidence Distribution[/bold]\n"
        for item in analysis:
            bar_len = int(item["score"] / 5)
            chart_content += f"{item['disease'][:20]:20s} {'#' * bar_len}\n"
        chart_display = Static(chart_content, classes="chart-display")
        chart_container.mount(chart_display)
    
    def action_clear_chat(self) -> None:
        """Clear the chat and results."""
        self.messages = []
        self.selected_file = None
        self.current_analysis = []
        self.summary = ""
        
        # Reset displays
        file_display = self.query_one("#file-display", Static)
        file_display.update("Selected: No file selected")
        
        status = self.query_one("#status-msg", Static)
        status.update("")
        
        summary_display = self.query_one("#summary-display", Static)
        summary_display.update("")
        
        # Clear chat scroll
        chat_scroll = self.query_one("#chat-scroll", ScrollableContainer)
        for child in list(chat_scroll.children):
            child.remove()
        
        chat_scroll.mount(Static(
            "[dim]Upload a medical image to begin analysis.\n\n"
            "Supported formats: JPG, PNG, BMP\n"
            "Recommended: Clear, well-lit images of skin lesions[/dim]",
            id="welcome-msg"
        ))
        
        # Clear results
        analysis_container = self.query_one("#analysis-container", Container)
        for child in list(analysis_container.children):
            child.remove()
        
        chart_container = self.query_one("#chart-container", Container)
        for child in list(chart_container.children):
            child.remove()
        
        try:
            placeholder = self.query_one("#results-placeholder", Static)
            placeholder.update("[dim]Analysis results will appear here after processing an image.[/dim]")
        except Exception:
            pass
        
        self.notify("Chat and results cleared", severity="information")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-upload":
            self.action_upload_image()
        elif button_id == "btn-analyze":
            self.action_analyze()
        elif button_id == "btn-clear":
            self.action_clear_chat()
