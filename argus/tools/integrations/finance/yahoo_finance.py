"""
Yahoo Finance Tool for ARGUS.

Get stock prices, company info, and financial data.
"""

from __future__ import annotations

import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class YahooFinanceTool(BaseTool):
    """
    Yahoo Finance data tool.
    
    Example:
        >>> tool = YahooFinanceTool()
        >>> result = tool(symbol="AAPL", action="quote")
    """
    
    name = "yahoo_finance"
    description = "Get stock quotes, company info, historical prices, and financial news."
    category = ToolCategory.DATA
    
    def __init__(self, config: Optional[ToolConfig] = None):
        super().__init__(config)
    
    def execute(
        self,
        symbol: str = "",
        action: str = "quote",
        period: str = "1mo",
        **kwargs: Any,
    ) -> ToolResult:
        """
        Get financial data.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            action: 'quote', 'info', 'history', 'news'
            period: Time period for history ('1d', '5d', '1mo', '3mo', '1y')
            
        Returns:
            ToolResult with financial data
        """
        try:
            import yfinance as yf
        except ImportError:
            return ToolResult.from_error("yfinance not installed. Run: pip install yfinance")
        
        if not symbol:
            return ToolResult.from_error("Symbol is required")
        
        try:
            ticker = yf.Ticker(symbol)
            
            if action == "quote":
                info = ticker.info
                return ToolResult.from_data({
                    "symbol": symbol,
                    "name": info.get("longName"),
                    "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                    "change": info.get("regularMarketChange"),
                    "change_percent": info.get("regularMarketChangePercent"),
                    "volume": info.get("regularMarketVolume"),
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "52_week_high": info.get("fiftyTwoWeekHigh"),
                    "52_week_low": info.get("fiftyTwoWeekLow"),
                })
            
            elif action == "info":
                info = ticker.info
                return ToolResult.from_data({
                    "symbol": symbol,
                    "name": info.get("longName"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "country": info.get("country"),
                    "employees": info.get("fullTimeEmployees"),
                    "description": info.get("longBusinessSummary", "")[:1000],
                    "website": info.get("website"),
                })
            
            elif action == "history":
                hist = ticker.history(period=period)
                if hist.empty:
                    return ToolResult.from_error("No historical data found")
                records = hist.reset_index().head(100).to_dict("records")
                # Convert timestamps
                for r in records:
                    if "Date" in r:
                        r["Date"] = str(r["Date"])
                    if "Datetime" in r:
                        r["Datetime"] = str(r["Datetime"])
                return ToolResult.from_data({
                    "symbol": symbol,
                    "period": period,
                    "data": records,
                    "count": len(records),
                })
            
            elif action == "news":
                news = ticker.news
                formatted = [
                    {
                        "title": n.get("title"),
                        "publisher": n.get("publisher"),
                        "link": n.get("link"),
                        "type": n.get("type"),
                    }
                    for n in (news or [])[:10]
                ]
                return ToolResult.from_data({"symbol": symbol, "news": formatted})
            
            else:
                return ToolResult.from_error(f"Unknown action: {action}")
                
        except Exception as e:
            logger.error(f"Yahoo Finance error: {e}")
            return ToolResult.from_error(f"Yahoo Finance error: {e}")
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    "action": {"type": "string", "enum": ["quote", "info", "history", "news"]},
                    "period": {"type": "string", "default": "1mo"},
                },
                "required": ["symbol"],
            },
        }
