"""
YouTube Transcript Tool for ARGUS.

Extract transcripts from YouTube videos.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class YouTubeTool(BaseTool):
    """
    YouTube transcript extraction tool.
    
    Example:
        >>> tool = YouTubeTool()
        >>> result = tool(video_url="https://youtube.com/watch?v=abc123")
    """
    
    name = "youtube_transcript"
    description = "Extract transcripts/subtitles from YouTube videos."
    category = ToolCategory.DATA
    
    def __init__(self, config: Optional[ToolConfig] = None):
        super().__init__(config)
    
    def _extract_video_id(self, url_or_id: str) -> Optional[str]:
        """Extract video ID from URL or return as-is."""
        patterns = [
            r"(?:v=|/)([0-9A-Za-z_-]{11}).*",
            r"^([0-9A-Za-z_-]{11})$",
        ]
        for pattern in patterns:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)
        return None
    
    def execute(
        self,
        video_url: str = "",
        language: str = "en",
        **kwargs: Any,
    ) -> ToolResult:
        """
        Get YouTube transcript.
        
        Args:
            video_url: YouTube URL or video ID
            language: Language code for transcript
            
        Returns:
            ToolResult with transcript
        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            return ToolResult.from_error(
                "youtube-transcript-api not installed. Run: pip install youtube-transcript-api"
            )
        
        if not video_url:
            return ToolResult.from_error("video_url is required")
        
        video_id = self._extract_video_id(video_url)
        if not video_id:
            return ToolResult.from_error(f"Could not extract video ID from: {video_url}")
        
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to get requested language, fall back to auto-generated
            try:
                transcript = transcript_list.find_transcript([language])
            except Exception:
                transcript = transcript_list.find_generated_transcript([language, "en"])
            
            entries = transcript.fetch()
            
            # Format transcript
            full_text = " ".join([entry["text"] for entry in entries])
            
            formatted_entries = [
                {
                    "text": entry["text"],
                    "start": entry["start"],
                    "duration": entry["duration"],
                }
                for entry in entries[:500]  # Limit entries
            ]
            
            return ToolResult.from_data({
                "video_id": video_id,
                "language": transcript.language,
                "full_text": full_text[:10000],
                "entries": formatted_entries,
                "entry_count": len(entries),
            })
            
        except Exception as e:
            logger.error(f"YouTube transcript failed: {e}")
            return ToolResult.from_error(f"Failed to get transcript: {e}")
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "video_url": {"type": "string", "description": "YouTube URL or video ID"},
                    "language": {"type": "string", "default": "en"},
                },
                "required": ["video_url"],
            },
        }
