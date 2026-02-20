"""
Cartesia Tool for ARGUS.

Real-time voice AI and speech synthesis.
"""

from __future__ import annotations

import os
import logging
import base64
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class CartesiaTool(BaseTool):
    """
    Cartesia - Real-time voice AI platform.
    
    Features:
    - Text-to-speech synthesis
    - Voice cloning
    - Voice management
    - Real-time streaming
    - Multi-language support
    
    Example:
        >>> tool = CartesiaTool()
        >>> result = tool(action="text_to_speech", text="Hello world", voice_id="my-voice")
        >>> result = tool(action="list_voices")
    """
    
    name = "cartesia"
    description = "Real-time voice AI and speech synthesis"
    category = ToolCategory.MEDIA
    version = "1.0.0"
    
    API_BASE = "https://api.cartesia.ai"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.api_key = api_key or os.getenv("CARTESIA_API_KEY")
        
        self._session = None
        
        if not self.api_key:
            logger.warning("No Cartesia API key provided")
        
        logger.debug("Cartesia tool initialized")
    
    def _get_session(self):
        """Get HTTP session with authentication."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "X-API-Key": self.api_key,
                "Cartesia-Version": "2024-06-10",
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
        """Make API request to Cartesia."""
        session = self._get_session()
        url = f"{self.API_BASE}{endpoint}"
        
        headers = {}
        if accept_audio:
            headers["Accept"] = "audio/wav"
        
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
                message = error_data.get("message", error_data.get("error", f"HTTP {response.status_code}"))
                raise Exception(str(message))
            except ValueError:
                raise Exception(f"HTTP {response.status_code}")
        
        if accept_audio:
            return response.content
        
        if response.status_code == 204:
            return {}
        return response.json() if response.text else {}
    
    def execute(
        self,
        action: str = "list_voices",
        voice_id: Optional[str] = None,
        text: Optional[str] = None,
        model_id: Optional[str] = None,
        name: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute Cartesia operations."""
        actions = {
            # Voice operations
            "list_voices": self._list_voices,
            "get_voice": self._get_voice,
            "create_voice": self._create_voice,
            "clone_voice": self._clone_voice,
            "update_voice": self._update_voice,
            "delete_voice": self._delete_voice,
            
            # Text-to-speech
            "text_to_speech": self._text_to_speech,
            "text_to_speech_bytes": self._text_to_speech_bytes,
            
            # Embedding operations
            "create_embedding": self._create_embedding,
            
            # API info
            "list_languages": self._list_languages,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            if not self.api_key:
                return ToolResult.from_error("Cartesia API key not configured")
            
            return actions[action](
                voice_id=voice_id,
                text=text,
                model_id=model_id,
                name=name,
                language=language,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Cartesia error: {e}")
            return ToolResult.from_error(f"Cartesia error: {e}")
    
    # Voice operations
    def _list_voices(self, **kwargs) -> ToolResult:
        """List available voices."""
        response = self._request("GET", "/voices")
        
        voices = []
        for v in response if isinstance(response, list) else []:
            voices.append({
                "id": v.get("id"),
                "name": v.get("name"),
                "description": v.get("description"),
                "language": v.get("language"),
                "is_public": v.get("is_public"),
                "created_at": v.get("created_at"),
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
        
        response = self._request("GET", f"/voices/{voice_id}")
        
        return ToolResult.from_data({
            "voice": {
                "id": response.get("id"),
                "name": response.get("name"),
                "description": response.get("description"),
                "language": response.get("language"),
                "embedding": response.get("embedding"),
                "is_public": response.get("is_public"),
                "created_at": response.get("created_at"),
            }
        })
    
    def _create_voice(
        self,
        name: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a voice from embedding."""
        if not name:
            return ToolResult.from_error("name is required")
        
        embedding = kwargs.get("embedding")
        if not embedding:
            return ToolResult.from_error("embedding is required")
        
        data = {
            "name": name,
            "embedding": embedding,
        }
        
        if language:
            data["language"] = language
        
        description = kwargs.get("description")
        if description:
            data["description"] = description
        
        response = self._request("POST", "/voices", data=data)
        
        return ToolResult.from_data({
            "voice_id": response.get("id"),
            "name": response.get("name"),
            "created": True,
        })
    
    def _clone_voice(
        self,
        name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Clone a voice from audio samples."""
        if not name:
            return ToolResult.from_error("name is required")
        
        audio_base64 = kwargs.get("audio")
        if not audio_base64:
            return ToolResult.from_error("audio (base64) is required")
        
        # First create embedding from audio
        embedding_data = {
            "audio": audio_base64,
        }
        
        enhance = kwargs.get("enhance", True)
        if enhance:
            embedding_data["enhance"] = True
        
        embedding_response = self._request(
            "POST",
            "/voices/clone",
            data=embedding_data,
        )
        
        embedding = embedding_response.get("embedding")
        
        # Create voice with embedding
        voice_data = {
            "name": name,
            "embedding": embedding,
        }
        
        language = kwargs.get("language")
        if language:
            voice_data["language"] = language
        
        description = kwargs.get("description")
        if description:
            voice_data["description"] = description
        
        response = self._request("POST", "/voices", data=voice_data)
        
        return ToolResult.from_data({
            "voice_id": response.get("id"),
            "name": response.get("name"),
            "cloned": True,
        })
    
    def _update_voice(
        self,
        voice_id: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Update a voice."""
        if not voice_id:
            return ToolResult.from_error("voice_id is required")
        
        data = {}
        
        if name:
            data["name"] = name
        
        description = kwargs.get("description")
        if description is not None:
            data["description"] = description
        
        if not data:
            return ToolResult.from_error("No fields to update")
        
        response = self._request("PATCH", f"/voices/{voice_id}", data=data)
        
        return ToolResult.from_data({
            "voice_id": response.get("id"),
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
    
    # Text-to-speech
    def _text_to_speech(
        self,
        voice_id: Optional[str] = None,
        text: Optional[str] = None,
        model_id: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Generate speech from text."""
        if not voice_id:
            return ToolResult.from_error("voice_id is required")
        if not text:
            return ToolResult.from_error("text is required")
        
        # Get voice embedding first
        voice_response = self._request("GET", f"/voices/{voice_id}")
        embedding = voice_response.get("embedding")
        
        if not embedding:
            return ToolResult.from_error("Could not retrieve voice embedding")
        
        data = {
            "model_id": model_id or "sonic-english",
            "transcript": text,
            "voice": {
                "mode": "embedding",
                "embedding": embedding,
            },
            "output_format": {
                "container": "raw",
                "encoding": "pcm_f32le",
                "sample_rate": 44100,
            },
        }
        
        if language:
            data["language"] = language
        
        speed = kwargs.get("speed")
        if speed is not None:
            data["speed"] = speed
        
        emotion = kwargs.get("emotion")
        if emotion:
            data["emotion"] = emotion
        
        response = self._request("POST", "/tts/bytes", data=data, accept_audio=True)
        
        audio_b64 = base64.b64encode(response).decode("utf-8")
        
        return ToolResult.from_data({
            "audio_base64": audio_b64,
            "format": "pcm",
            "sample_rate": 44100,
            "size_bytes": len(response),
        })
    
    def _text_to_speech_bytes(
        self,
        voice_id: Optional[str] = None,
        text: Optional[str] = None,
        model_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Generate speech and return raw bytes info."""
        return self._text_to_speech(
            voice_id=voice_id,
            text=text,
            model_id=model_id,
            **kwargs,
        )
    
    # Embedding operations
    def _create_embedding(
        self,
        **kwargs,
    ) -> ToolResult:
        """Create voice embedding from audio."""
        audio_base64 = kwargs.get("audio")
        if not audio_base64:
            return ToolResult.from_error("audio (base64) is required")
        
        data = {
            "audio": audio_base64,
        }
        
        enhance = kwargs.get("enhance", True)
        if enhance:
            data["enhance"] = True
        
        response = self._request("POST", "/voices/clone", data=data)
        
        return ToolResult.from_data({
            "embedding": response.get("embedding"),
            "created": True,
        })
    
    # API info
    def _list_languages(self, **kwargs) -> ToolResult:
        """List supported languages."""
        # Cartesia supported languages
        languages = [
            {"code": "en", "name": "English"},
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "it", "name": "Italian"},
            {"code": "pt", "name": "Portuguese"},
            {"code": "ja", "name": "Japanese"},
            {"code": "zh", "name": "Chinese"},
            {"code": "ko", "name": "Korean"},
            {"code": "hi", "name": "Hindi"},
            {"code": "pl", "name": "Polish"},
            {"code": "ru", "name": "Russian"},
            {"code": "tr", "name": "Turkish"},
            {"code": "nl", "name": "Dutch"},
            {"code": "sv", "name": "Swedish"},
        ]
        
        return ToolResult.from_data({
            "languages": languages,
            "count": len(languages),
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
                            "list_voices", "get_voice", "create_voice", "clone_voice",
                            "update_voice", "delete_voice",
                            "text_to_speech", "text_to_speech_bytes",
                            "create_embedding",
                            "list_languages",
                        ],
                    },
                    "voice_id": {"type": "string"},
                    "text": {"type": "string"},
                    "model_id": {"type": "string"},
                    "name": {"type": "string"},
                    "language": {"type": "string"},
                    "speed": {"type": "number", "minimum": 0.5, "maximum": 2.0},
                },
                "required": ["action"],
            },
        }
