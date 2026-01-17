"""
HTTP Requests Tool for ARGUS.

Make HTTP requests to fetch web content.
"""

from __future__ import annotations

import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class RequestsTool(BaseTool):
    """
    HTTP requests tool for fetching web content.
    
    Example:
        >>> tool = RequestsTool()
        >>> result = tool(url="https://api.example.com/data", method="GET")
    """
    
    name = "http_request"
    description = "Make HTTP requests (GET, POST, etc.) to fetch web content or call APIs."
    category = ToolCategory.EXTERNAL_API
    
    def __init__(
        self,
        timeout: float = 30.0,
        headers: Optional[dict] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        self.timeout = timeout
        self.default_headers = headers or {"User-Agent": "ARGUS-Agent/1.0"}
    
    def execute(
        self,
        url: str = "",
        method: str = "GET",
        headers: Optional[dict] = None,
        data: Optional[dict] = None,
        json: Optional[dict] = None,
        params: Optional[dict] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Make HTTP request.
        
        Args:
            url: URL to request
            method: HTTP method
            headers: Request headers
            data: Form data
            json: JSON body
            params: Query parameters
            
        Returns:
            ToolResult with response
        """
        try:
            import requests
        except ImportError:
            return ToolResult.from_error("requests not installed")
        
        if not url:
            return ToolResult.from_error("URL is required")
        
        try:
            req_headers = {**self.default_headers}
            if headers:
                req_headers.update(headers)
            
            response = requests.request(
                method=method.upper(),
                url=url,
                headers=req_headers,
                data=data,
                json=json,
                params=params,
                timeout=self.timeout,
            )
            
            # Try to parse JSON
            try:
                content = response.json()
                content_type = "json"
            except ValueError:
                content = response.text[:10000]  # Truncate
                content_type = "text"
            
            return ToolResult.from_data({
                "url": url,
                "status_code": response.status_code,
                "content_type": content_type,
                "content": content,
                "headers": dict(response.headers),
            })
            
        except requests.Timeout:
            return ToolResult.from_error(f"Request timed out: {url}")
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return ToolResult.from_error(f"Request failed: {e}")
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to request"},
                    "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]},
                    "headers": {"type": "object", "description": "Request headers"},
                    "json": {"type": "object", "description": "JSON body"},
                },
                "required": ["url"],
            },
        }
