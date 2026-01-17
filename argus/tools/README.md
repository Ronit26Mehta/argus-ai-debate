# ARGUS Tools Module

## Overview

The `tools/` module provides 19+ pre-built tool integrations that agents can use for search, data retrieval, code execution, and more.

## Structure

```
tools/
├── __init__.py
├── base.py            # BaseTool interface
├── registry.py        # Tool registry
├── executor.py        # Tool executor
├── cache.py           # Result caching
├── guardrails.py      # Safety guardrails
└── integrations/      # Pre-built tools
    ├── search/        # Search tools (6)
    ├── web/           # Web tools (4)
    ├── productivity/  # Productivity tools (5)
    ├── database/      # Database tools (2)
    └── finance/       # Finance tools (2)
```

---

## Available Tools (19)

### Search Tools

| Tool | Description | API Key |
|------|-------------|---------|
| `DuckDuckGoTool` | Web search | None (free) |
| `WikipediaTool` | Wikipedia lookup | None (free) |
| `ArxivTool` | Scientific papers | None (free) |
| `TavilyTool` | AI-optimized search | `TAVILY_API_KEY` |
| `BraveTool` | Privacy search | `BRAVE_API_KEY` |
| `ExaTool` | Neural search | `EXA_API_KEY` |

### Web Tools

| Tool | Description | API Key |
|------|-------------|---------|
| `RequestsTool` | HTTP requests | None |
| `WebScraperTool` | Web scraping | None |
| `JinaReaderTool` | URL to markdown | `JINA_API_KEY` (optional) |
| `YouTubeTool` | Video transcripts | None |

### Productivity Tools

| Tool | Description | API Key |
|------|-------------|---------|
| `FileSystemTool` | File operations | None |
| `PythonReplTool` | Code execution | None |
| `ShellTool` | Shell commands | None |
| `GitHubTool` | GitHub API | `GITHUB_TOKEN` |
| `JsonTool` | JSON parsing | None |

### Database Tools

| Tool | Description | API Key |
|------|-------------|---------|
| `SqlTool` | SQL queries | Connection string |
| `PandasTool` | DataFrame analysis | None |

### Finance Tools

| Tool | Description | API Key |
|------|-------------|---------|
| `YahooFinanceTool` | Stock data | None (free) |
| `WeatherTool` | Weather data | `OPENWEATHERMAP_API_KEY` |

---

## Usage Examples

### Search Tools

```python
from argus.tools.integrations import (
    DuckDuckGoTool, WikipediaTool, ArxivTool, TavilyTool
)

# DuckDuckGo - Free web search
search = DuckDuckGoTool()
result = search(query="machine learning trends 2024", max_results=10)

for item in result.data["results"]:
    print(f"📰 {item['title']}")
    print(f"   {item['url']}")
    print(f"   {item['snippet'][:100]}...")

# Wikipedia
wiki = WikipediaTool()

# Search for articles
result = wiki(query="Artificial Intelligence", action="search")
print(result.data["results"])

# Get article summary
result = wiki(query="Machine Learning", action="summary", sentences=5)
print(result.data["summary"])

# Get full page
result = wiki(query="Neural Network", action="page")
print(result.data["content"][:500])

# ArXiv papers
arxiv = ArxivTool()
result = arxiv(query="transformer attention mechanism", max_results=5)

for paper in result.data["results"]:
    print(f"📄 {paper['title']}")
    print(f"   Authors: {', '.join(paper['authors'])}")
    print(f"   PDF: {paper['pdf_url']}")
    print(f"   {paper['summary'][:200]}...")

# Tavily (AI-optimized, requires API key)
tavily = TavilyTool(api_key="tvly-...")
result = tavily(
    query="latest developments in AGI",
    search_depth="advanced",
    include_answer=True,
)
print(f"Answer: {result.data['answer']}")
```

### Web Tools

```python
from argus.tools.integrations import (
    RequestsTool, WebScraperTool, JinaReaderTool, YouTubeTool
)

# HTTP Requests
http = RequestsTool()
result = http(url="https://api.github.com/repos/python/cpython", method="GET")
print(f"Status: {result.data['status_code']}")
print(f"Stars: {result.data['content']['stargazers_count']}")

# Web Scraping
scraper = WebScraperTool()

# Extract text
result = scraper(url="https://example.com", extract="text")
print(result.data["content"])

# Extract links
result = scraper(url="https://example.com", extract="links")
for link in result.data["content"]:
    print(f"{link['text']}: {link['href']}")

# CSS selector
result = scraper(
    url="https://example.com",
    extract="selector",
    selector="h1.title"
)

# Jina Reader (URL to clean markdown)
jina = JinaReaderTool()
result = jina(url="https://example.com")
print(result.data["content"])  # Clean markdown text

# YouTube Transcripts
youtube = YouTubeTool()
result = youtube(
    video_url="https://youtube.com/watch?v=dQw4w9WgXcQ",
    language="en",
)
print(result.data["full_text"])
```

### Productivity Tools

```python
from argus.tools.integrations import (
    FileSystemTool, PythonReplTool, ShellTool, GitHubTool, JsonTool
)

# File System
fs = FileSystemTool(base_dir="./data")

# List directory
result = fs(action="list", path=".")
for item in result.data["items"]:
    print(f"{item['type']}: {item['name']}")

# Read file
result = fs(action="read", path="config.json")
print(result.data["content"])

# Write file
result = fs(action="write", path="output.txt", content="Hello World")

# Python REPL
repl = PythonReplTool()

# Execute code
result = repl(code="x = 10; y = 20; print(x + y)")
print(result.data["output"])  # 30

# Evaluate expression
result = repl(code="sum([1, 2, 3, 4, 5])")
print(result.data["result"])  # 15

# Complex computation
result = repl(code="""
import math
primes = [n for n in range(2, 100) if all(n % i != 0 for i in range(2, int(math.sqrt(n))+1))]
print(f"Found {len(primes)} primes")
print(primes[:10])
""")

# Shell Commands
shell = ShellTool()
result = shell(command="ls -la")
print(result.data["stdout"])

# GitHub
gh = GitHubTool(token="ghp_...")

# Search repos
result = gh(action="search_repos", query="machine learning python")
for repo in result.data["results"]:
    print(f"⭐ {repo['stars']} - {repo['name']}")

# Get repo info
result = gh(action="get_repo", owner="python", repo="cpython")

# JSON Tool
json_tool = JsonTool()

# Parse JSON
result = json_tool(action="parse", data='{"name": "John", "age": 30}')
print(result.data["parsed"])

# Format JSON
result = json_tool(action="format", data='{"a":1,"b":2}')
print(result.data["formatted"])

# Get nested value
result = json_tool(
    action="get",
    data='{"user": {"name": "John", "address": {"city": "NYC"}}}',
    path="user.address.city",
)
print(result.data["value"])  # NYC
```

### Database Tools

```python
from argus.tools.integrations import SqlTool, PandasTool

# SQL Database
sql = SqlTool(connection_string="sqlite:///mydb.db")

# Query
result = sql(query="SELECT * FROM users LIMIT 10")
for row in result.data["rows"]:
    print(row)

# With parameters
result = sql(
    query="SELECT * FROM users WHERE age > :min_age",
    params={"min_age": 25},
)

# Pandas DataFrame
df_tool = PandasTool()

# Load CSV
result = df_tool(action="read_csv", path="data.csv")
print(f"Shape: {result.data['shape']}")

# View data
result = df_tool(action="head", n=5)
print(result.data["rows"])

# Describe statistics
result = df_tool(action="describe")
print(result.data["statistics"])

# Query DataFrame
result = df_tool(action="query", query="age > 25 and salary > 50000")
```

### Finance Tools

```python
from argus.tools.integrations import YahooFinanceTool, WeatherTool

# Yahoo Finance
yf = YahooFinanceTool()

# Get quote
result = yf(symbol="AAPL", action="quote")
print(f"Apple: ${result.data['price']}")
print(f"Change: {result.data['change_percent']}%")

# Company info
result = yf(symbol="GOOGL", action="info")
print(f"Sector: {result.data['sector']}")
print(f"Employees: {result.data['employees']}")

# Historical data
result = yf(symbol="MSFT", action="history", period="1mo")
for day in result.data["data"][:5]:
    print(f"{day['Date']}: ${day['Close']:.2f}")

# News
result = yf(symbol="NVDA", action="news")
for article in result.data["news"]:
    print(f"📰 {article['title']}")

# Weather
weather = WeatherTool(api_key="...")

# Current weather
result = weather(location="New York", action="current")
print(f"Temperature: {result.data['temperature']}°C")
print(f"Weather: {result.data['weather']}")

# Forecast
result = weather(location="London", action="forecast")
for forecast in result.data["forecasts"]:
    print(f"{forecast['datetime']}: {forecast['temp']}°C")
```

---

## Creating Custom Tools

```python
from argus.tools import BaseTool, ToolResult, ToolCategory

class MyCustomTool(BaseTool):
    """My custom tool implementation."""
    
    name = "my_tool"
    description = "Does something useful"
    category = ToolCategory.UTILITY
    
    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
    
    def execute(self, query: str, **kwargs) -> ToolResult:
        try:
            # Your implementation
            result = {"data": f"Processed: {query}"}
            return ToolResult.from_data(result)
        except Exception as e:
            return ToolResult.from_error(str(e))
    
    def get_schema(self) -> dict:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query input"},
                },
                "required": ["query"],
            },
        }

# Use custom tool
tool = MyCustomTool()
result = tool(query="hello world")
print(result.data)
```

---

## Tool Registry

```python
from argus.tools import ToolRegistry, register_tool, get_tool, list_tools

# Register custom tool
register_tool(MyCustomTool())

# Get tool by name
tool = get_tool("my_tool")

# List all tools
all_tools = list_tools()
print(all_tools)
```
