"""
ElevenLabs Tool for ARGUS.

AI voice synthesis and audio generation.
"""

from __future__ import annotations

import os
import logging
import base64
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class ElevenLabsTool(BaseTool):
    """
    ElevenLabs - AI voice synthesis platform.
    
    Features:
    - Text-to-speech synthesis
    - Voice cloning
    - Voice management
    - Model management
    - Audio history management
    - Sound effects generation
    
    Example:
        >>> tool = ElevenLabsTool()
        >>> result = tool(action="text_to_speech", text="Hello world", voice_id="21m00Tcm4TlvDq8ikWAM")
        >>> result = tool(action="list_voices")
    """
    
    name = "elevenlabs"
    description = "AI voice synthesis and audio generation"
    category = ToolCategory.MEDIA
    version = "1.0.0"
    
    API_BASE = "https://api.elevenlabs.io/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        
        self._session = None
        
        if not self.api_key:
            logger.warning("No ElevenLabs API key provided")
        
        logger.debug("ElevenLabs tool initialized")
    
    def _get_session(self):
        """Get HTTP session with authentication."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "xi-api-key": self.api_key,
                "Content-Type": "application/json",
            })
        return self._session
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
        accept_audio: bool = False,
    ) -> dict | bytes:
        """Make API request to ElevenLabs."""
        session = self._get_session()
        url = f"{self.API_BASE}{endpoint}"
        
        headers = {}
        if accept_audio:
            headers["Accept"] = "audio/mpeg"
        
        response = session.request(
            method=method,
            url=url,
            json=data,
            params=params,
            headers=headers if headers else None,
        )
        
        if response.status_code >= 400:
            try:
                error_data = response.json()
                detail = error_data.get("detail", {})
                if isinstance(detail, dict):
                    message = detail.get("message", f"HTTP {response.status_code}")
                else:
                    message = str(detail)
                raise Exception(message)
            except ValueError:
                raise Exception(f"HTTP {response.status_code}")
        
        if accept_audio:
            return response.content
        
        return response.json() if response.text else {}
    
    def execute(
        self,
        action: str = "list_voices",
        voice_id: Optional[str] = None,
        text: Optional[str] = None,
        model_id: Optional[str] = None,
        name: Optional[str] = None,
        history_item_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute ElevenLabs operations."""
        actions = {
            # Voice operations
            "list_voices": self._list_voices,
            "get_voice": self._get_voice,
            "add_voice": self._add_voice,
            "edit_voice": self._edit_voice,
            "delete_voice": self._delete_voice,
            "get_voice_settings": self._get_voice_settings,
            
            # Text-to-speech
            "text_to_speech": self._text_to_speech,
            "text_to_speech_stream": self._text_to_speech_stream,
            
            # Model operations
            "list_models": self._list_models,
            
            # History operations
            "list_history": self._list_history,
            "get_history_item": self._get_history_item,
            "delete_history_item": self._delete_history_item,
            "download_history_audio": self._download_history_audio,
            
            # Sound effects
            "generate_sound_effect": self._generate_sound_effect,
            
            # User info
            "get_user": self._get_user,
            "get_subscription": self._get_subscription,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            if not self.api_key:
                return ToolResult.from_error("ElevenLabs API key not configured")
            
            return actions[action](
                voice_id=voice_id,
                text=text,
                model_id=model_id,
                name=name,
                history_item_id=history_item_id,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"ElevenLabs error: {e}")
            return ToolResult.from_error(f"ElevenLabs error: {e}")
    
    # Voice operations
    def _list_voices(self, **kwargs) -> ToolResult:
        """List available voices."""
        response = self._request("GET", "/voices")
        
        voices = []
        for v in response.get("voices", []):
            voices.append({
                "voice_id": v.get("voice_id"),
                "name": v.get("name"),
                "category": v.get("category"),
                "labels": v.get("labels", {}),
                "preview_url": v.get("preview_url"),
            })
        
        return ToolResult.from_data({
            "voices": voices,
            "count": len(voices),
        })
    
    def _get_voice(
        self,
        voice_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get voice details."""
        if not voice_id:
            return ToolResult.from_error("voice_id is required")
        
        params = {}
        with_settings = kwargs.get("with_settings", True)
        if with_settings:
            params["with_settings"] = "true"
        
        response = self._request(
            "GET",
            f"/voices/{voice_id}",
            params=params if params else None,
        )
        
        return ToolResult.from_data({
            "voice": {
                "voice_id": response.get("voice_id"),
                "name": response.get("name"),
                "category": response.get("category"),
                "labels": response.get("labels", {}),
                "description": response.get("description"),
                "settings": response.get("settings"),
                "samples": response.get("samples", []),
            }
        })
    
    def _add_voice(
        self,
        name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Add a new voice (voice cloning)."""
        if not name:
            return ToolResult.from_error("name is required")
        
        files = kwargs.get("files", [])
        if not files:
            return ToolResult.from_error("files (audio samples) are required")
        
        description = kwargs.get("description", "")
        labels = kwargs.get("labels", {})
        
        import requests
        
        session = self._get_session()
        url = f"{self.API_BASE}/voices/add"
        
        # Prepare multipart form data
        form_data = {
            "name": name,
            "description": description,
        }
        
        if labels:
            import json
            form_data["labels"] = json.dumps(labels)
        
        files_list = []
        for f in files:
            if isinstance(f, str):
                # Assume base64 encoded audio
                audio_data = base64.b64decode(f)
                files_list.append(("files", ("audio.mp3", audio_data, "audio/mpeg")))
        
        # Remove Content-Type header for multipart
        headers = {"xi-api-key": self.api_key}
        
        response = requests.post(
            url,
            data=form_data,
            files=files_list if files_list else None,
            headers=headers,
        )
        
        if response.status_code >= 400:
            raise Exception(f"Failed to add voice: {response.text}")
        
        result = response.json()
        
        return ToolResult.from_data({
            "voice_id": result.get("voice_id"),
            "created": True,
        })
    
    def _edit_voice(
        self,
        voice_id: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Edit a voice."""
        if not voice_id:
            return ToolResult.from_error("voice_id is required")
        if not name:
            return ToolResult.from_error("name is required")
        
        import requests
        
        url = f"{self.API_BASE}/voices/{voice_id}/edit"
        
        form_data = {"name": name}
        
        description = kwargs.get("description")
        if description:
            form_data["description"] = description
        
        labels = kwargs.get("labels")
        if labels:
            import json
            form_data["labels"] = json.dumps(labels)
        
        headers = {"xi-api-key": self.api_key}
        
        response = requests.post(url, data=form_data, headers=headers)
        
        if response.status_code >= 400:
            raise Exception(f"Failed to edit voice: {response.text}")
        
        return ToolResult.from_data({
            "voice_id": voice_id,
            "updated": True,
        })
    
    def _delete_voice(
        self,
        voice_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete a voice."""
        if not voice_id:
            return ToolResult.from_error("voice_id is required")
        
        self._request("DELETE", f"/voices/{voice_id}")
        
        return ToolResult.from_data({
            "voice_id": voice_id,
            "deleted": True,
        })
    
    def _get_voice_settings(
        self,
        voice_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get voice settings."""
        if not voice_id:
            return ToolResult.from_error("voice_id is required")
        
        response = self._request("GET", f"/voices/{voice_id}/settings")
        
        return ToolResult.from_data({
            "settings": {
                "stability": response.get("stability"),
                "similarity_boost": response.get("similarity_boost"),
                "style": response.get("style"),
                "use_speaker_boost": response.get("use_speaker_boost"),
            }
        })
    
    # Text-to-speech
    def _text_to_speech(
        self,
        voice_id: Optional[str] = None,
        text: Optional[str] = None,
        model_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Generate speech from text."""
        if not voice_id:
            return ToolResult.from_error("voice_id is required")
        if not text:
            return ToolResult.from_error("text is required")
        
        data = {
            "text": text,
            "model_id": model_id or "eleven_monolingual_v1",
        }
        
        # Voice settings
        stability = kwargs.get("stability")
        similarity_boost = kwargs.get("similarity_boost")
        style = kwargs.get("style")
        
        if any([stability, similarity_boost, style]):
            data["voice_settings"] = {}
            if stability is not None:
                data["voice_settings"]["stability"] = stability
            if similarity_boost is not None:
                data["voice_settings"]["similarity_boost"] = similarity_boost
            if style is not None:
                data["voice_settings"]["style"] = style
        
        audio_bytes = self._request(
            "POST",
            f"/text-to-speech/{voice_id}",
            data=data,
            accept_audio=True,
        )
        
        # Return base64 encoded audio
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        return ToolResult.from_data({
            "audio_base64": audio_b64,
            "format": "mp3",
            "size_bytes": len(audio_bytes),
        })
    
    def _text_to_speech_stream(
        self,
        voice_id: Optional[str] = None,
        text: Optional[str] = None,
        model_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Generate speech with streaming (returns audio chunks)."""
        # For simplicity, this uses the same endpoint but indicates streaming
        return self._text_to_speech(
            voice_id=voice_id,
            text=text,
            model_id=model_id,
            **kwargs,
        )
    
    # Model operations
    def _list_models(self, **kwargs) -> ToolResult:
        """List available models."""
        response = self._request("GET", "/models")
        
        models = []
        for m in response if isinstance(response, list) else []:
            models.append({
                "model_id": m.get("model_id"),
                "name": m.get("name"),
                "description": m.get("description"),
                "can_be_finetuned": m.get("can_be_finetuned"),
                "can_do_text_to_speech": m.get("can_do_text_to_speech"),
                "can_do_voice_conversion": m.get("can_do_voice_conversion"),
                "languages": [lang.get("language_id") for lang in m.get("languages", [])],
            })
        
        return ToolResult.from_data({
            "models": models,
            "count": len(models),
        })
    
    # History operations
    def _list_history(
        self,
        **kwargs,
    ) -> ToolResult:
        """List generation history."""
        params = {}
        
        page_size = kwargs.get("page_size", 100)
        params["page_size"] = min(page_size, 100)
        
        start_after = kwargs.get("start_after")
        if start_after:
            params["start_after_history_item_id"] = start_after
        
        voice_id = kwargs.get("voice_id")
        if voice_id:
            params["voice_id"] = voice_id
        
        response = self._request("GET", "/history", params=params)
        
        history = []
        for h in response.get("history", []):
            history.append({
                "history_item_id": h.get("history_item_id"),
                "voice_id": h.get("voice_id"),
                "voice_name": h.get("voice_name"),
                "text": h.get("text"),
                "date_unix": h.get("date_unix"),
                "character_count_change_from": h.get("character_count_change_from"),
                "character_count_change_to": h.get("character_count_change_to"),
            })
        
        return ToolResult.from_data({
            "history": history,
            "count": len(history),
            "has_more": response.get("has_more", False),
        })
    
    def _get_history_item(
        self,
        history_item_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get history item details."""
        if not history_item_id:
            return ToolResult.from_error("history_item_id is required")
        
        response = self._request("GET", f"/history/{history_item_id}")
        
        return ToolResult.from_data({
            "history_item": {
                "history_item_id": response.get("history_item_id"),
                "voice_id": response.get("voice_id"),
                "voice_name": response.get("voice_name"),
                "text": response.get("text"),
                "date_unix": response.get("date_unix"),
                "settings": response.get("settings"),
            }
        })
    
    def _delete_history_item(
        self,
        history_item_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete a history item."""
        if not history_item_id:
            return ToolResult.from_error("history_item_id is required")
        
        self._request("DELETE", f"/history/{history_item_id}")
        
        return ToolResult.from_data({
            "history_item_id": history_item_id,
            "deleted": True,
        })
    
    def _download_history_audio(
        self,
        history_item_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Download audio from history."""
        if not history_item_id:
            return ToolResult.from_error("history_item_id is required")
        
        audio_bytes = self._request(
            "GET",
            f"/history/{history_item_id}/audio",
            accept_audio=True,
        )
        
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        return ToolResult.from_data({
            "audio_base64": audio_b64,
            "format": "mp3",
            "size_bytes": len(audio_bytes),
        })
    
    # Sound effects
    def _generate_sound_effect(
        self,
        text: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Generate sound effects from text description."""
        if not text:
            return ToolResult.from_error("text (description) is required")
        
        duration_seconds = kwargs.get("duration_seconds")
        prompt_influence = kwargs.get("prompt_influence", 0.3)
        
        data = {
            "text": text,
            "prompt_influence": prompt_influence,
        }
        
        if duration_seconds:
            data["duration_seconds"] = duration_seconds
        
        audio_bytes = self._request(
            "POST",
            "/sound-generation",
            data=data,
            accept_audio=True,
        )
        
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        return ToolResult.from_data({
            "audio_base64": audio_b64,
            "format": "mp3",
            "size_bytes": len(audio_bytes),
        })
    
    # User info
    def _get_user(self, **kwargs) -> ToolResult:
        """Get current user info."""
        response = self._request("GET", "/user")
        
        return ToolResult.from_data({
            "user": {
                "subscription": response.get("subscription", {}),
                "is_new_user": response.get("is_new_user"),
                "xi_api_key": response.get("xi_api_key"),
                "can_use_delayed_payment_methods": response.get("can_use_delayed_payment_methods"),
            }
        })
    
    def _get_subscription(self, **kwargs) -> ToolResult:
        """Get subscription info."""
        response = self._request("GET", "/user/subscription")
        
        return ToolResult.from_data({
            "subscription": {
                "tier": response.get("tier"),
                "character_count": response.get("character_count"),
                "character_limit": response.get("character_limit"),
                "can_extend_character_limit": response.get("can_extend_character_limit"),
                "allowed_to_extend_character_limit": response.get("allowed_to_extend_character_limit"),
                "next_character_count_reset_unix": response.get("next_character_count_reset_unix"),
                "voice_limit": response.get("voice_limit"),
                "max_voice_add_edits": response.get("max_voice_add_edits"),
            }
        })
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "list_voices", "get_voice", "add_voice", "edit_voice",
                            "delete_voice", "get_voice_settings",
                            "text_to_speech", "text_to_speech_stream",
                            "list_models",
                            "list_history", "get_history_item", "delete_history_item", "download_history_audio",
                            "generate_sound_effect",
                            "get_user", "get_subscription",
                        ],
                    },
                    "voice_id": {"type": "string"},
                    "text": {"type": "string"},
                    "model_id": {"type": "string"},
                    "name": {"type": "string"},
                    "history_item_id": {"type": "string"},
                    "stability": {"type": "number", "minimum": 0, "maximum": 1},
                    "similarity_boost": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["action"],
            },
        }
