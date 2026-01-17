"""
Weather Tool for ARGUS.

Get current weather and forecasts via OpenWeatherMap.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class WeatherTool(BaseTool):
    """
    OpenWeatherMap weather tool.
    
    Example:
        >>> tool = WeatherTool(api_key="...")
        >>> result = tool(location="New York", units="metric")
    """
    
    name = "weather"
    description = "Get current weather and forecasts for any location."
    category = ToolCategory.EXTERNAL_API
    
    BASE_URL = "https://api.openweathermap.org/data/2.5"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        self.api_key = api_key or os.getenv("OPENWEATHERMAP_API_KEY")
    
    def execute(
        self,
        location: str = "",
        action: str = "current",
        units: str = "metric",
        **kwargs: Any,
    ) -> ToolResult:
        """
        Get weather data.
        
        Args:
            location: City name or coordinates
            action: 'current' or 'forecast'
            units: 'metric', 'imperial', or 'standard'
            
        Returns:
            ToolResult with weather data
        """
        try:
            import requests
        except ImportError:
            return ToolResult.from_error("requests not installed")
        
        if not self.api_key:
            return ToolResult.from_error("OPENWEATHERMAP_API_KEY not set")
        
        if not location:
            return ToolResult.from_error("Location is required")
        
        try:
            if action == "current":
                url = f"{self.BASE_URL}/weather"
            elif action == "forecast":
                url = f"{self.BASE_URL}/forecast"
            else:
                return ToolResult.from_error(f"Unknown action: {action}")
            
            response = requests.get(
                url,
                params={
                    "q": location,
                    "appid": self.api_key,
                    "units": units,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            
            if action == "current":
                return ToolResult.from_data({
                    "location": data.get("name"),
                    "country": data.get("sys", {}).get("country"),
                    "weather": data.get("weather", [{}])[0].get("description"),
                    "temperature": data.get("main", {}).get("temp"),
                    "feels_like": data.get("main", {}).get("feels_like"),
                    "humidity": data.get("main", {}).get("humidity"),
                    "wind_speed": data.get("wind", {}).get("speed"),
                    "units": units,
                })
            else:
                forecasts = [
                    {
                        "datetime": f.get("dt_txt"),
                        "temp": f.get("main", {}).get("temp"),
                        "weather": f.get("weather", [{}])[0].get("description"),
                    }
                    for f in data.get("list", [])[:10]
                ]
                return ToolResult.from_data({
                    "location": data.get("city", {}).get("name"),
                    "forecasts": forecasts,
                    "units": units,
                })
                
        except Exception as e:
            logger.error(f"Weather error: {e}")
            return ToolResult.from_error(f"Weather error: {e}")
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name or location"},
                    "action": {"type": "string", "enum": ["current", "forecast"]},
                    "units": {"type": "string", "enum": ["metric", "imperial", "standard"]},
                },
                "required": ["location"],
            },
        }
