"""
Web Connector for ARGUS.

Fetches content from web URLs with configurable options and robots.txt compliance.

Example:
    >>> connector = WebConnector()
    >>> result = connector.fetch("https://example.com/article")
    >>> 
    >>> for doc in result.documents:
    ...     print(doc.title)
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional, Any
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass, field

from pydantic import Field

from argus.knowledge.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectorResult,
)
from argus.core.models import Document, SourceType

logger = logging.getLogger(__name__)


# =============================================================================
# Robots.txt Parser
# =============================================================================

@dataclass
class RobotRule:
    """A single rule from robots.txt."""
    path: str
    allowed: bool


@dataclass
class RobotsTxtRules:
    """Parsed robots.txt rules for a user-agent."""
    user_agent: str
    rules: list[RobotRule] = field(default_factory=list)
    crawl_delay: Optional[float] = None
    sitemaps: list[str] = field(default_factory=list)
    
    def is_allowed(self, path: str) -> bool:
        """Check if a path is allowed by these rules.
        
        Args:
            path: URL path to check
            
        Returns:
            True if allowed, False if disallowed
        """
        if not self.rules:
            return True
        
        # Normalize path
        if not path:
            path = "/"
        if not path.startswith("/"):
            path = "/" + path
        
        # Find the most specific matching rule
        # Rules are matched by longest path prefix
        best_match: Optional[RobotRule] = None
        best_match_len = -1
        
        for rule in self.rules:
            rule_path = rule.path
            
            # Handle wildcard patterns
            if "*" in rule_path:
                # Convert to regex pattern
                pattern = rule_path.replace("*", ".*")
                if pattern.endswith("$"):
                    pattern = pattern[:-1] + "$"
                else:
                    pattern = pattern + ".*"
                try:
                    if re.match(pattern, path):
                        if len(rule_path) > best_match_len:
                            best_match = rule
                            best_match_len = len(rule_path)
                except re.error:
                    pass
            elif path.startswith(rule_path):
                if len(rule_path) > best_match_len:
                    best_match = rule
                    best_match_len = len(rule_path)
        
        if best_match is None:
            return True
        
        return best_match.allowed


class RobotsTxtParser:
    """Parser for robots.txt files with caching.
    
    Parses robots.txt content and provides URL access checking.
    Caches parsed results for efficiency.
    
    Example:
        >>> parser = RobotsTxtParser()
        >>> rules = parser.parse(robots_txt_content, "MyBot/1.0")
        >>> if rules.is_allowed("/some/path"):
        ...     # OK to crawl
        ...     pass
    """
    
    def __init__(self, cache_ttl: int = 3600):
        """Initialize the parser.
        
        Args:
            cache_ttl: Cache TTL in seconds (default 1 hour)
        """
        self.cache_ttl = cache_ttl
        self._cache: dict[str, tuple[RobotsTxtRules, float]] = {}
    
    def parse(self, content: str, user_agent: str = "*") -> RobotsTxtRules:
        """Parse robots.txt content.
        
        Args:
            content: Raw robots.txt content
            user_agent: User-agent to match rules for
            
        Returns:
            RobotsTxtRules for the specified user-agent
        """
        lines = content.split("\n")
        
        # Track rules for all user-agents we find
        all_rules: dict[str, RobotsTxtRules] = {}
        current_agents: list[str] = []
        sitemaps: list[str] = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            
            # Parse directive
            if ":" not in line:
                continue
            
            directive, _, value = line.partition(":")
            directive = directive.strip().lower()
            value = value.strip()
            
            if directive == "user-agent":
                # Start new user-agent block
                agent = value.lower()
                if agent not in all_rules:
                    all_rules[agent] = RobotsTxtRules(user_agent=agent)
                current_agents = [agent]
            
            elif directive == "disallow" and current_agents:
                if value:  # Empty disallow means allow all
                    for agent in current_agents:
                        all_rules[agent].rules.append(
                            RobotRule(path=value, allowed=False)
                        )
            
            elif directive == "allow" and current_agents:
                if value:
                    for agent in current_agents:
                        all_rules[agent].rules.append(
                            RobotRule(path=value, allowed=True)
                        )
            
            elif directive == "crawl-delay" and current_agents:
                try:
                    delay = float(value)
                    for agent in current_agents:
                        all_rules[agent].crawl_delay = delay
                except ValueError:
                    pass
            
            elif directive == "sitemap":
                sitemaps.append(value)
        
        # Add sitemaps to all rules
        for rules in all_rules.values():
            rules.sitemaps = sitemaps
        
        # Find matching rules for the specified user-agent
        user_agent_lower = user_agent.lower()
        
        # Try exact match first
        for agent, rules in all_rules.items():
            if agent in user_agent_lower or user_agent_lower in agent:
                return rules
        
        # Fall back to wildcard
        if "*" in all_rules:
            return all_rules["*"]
        
        # No rules found, allow everything
        return RobotsTxtRules(user_agent=user_agent)
    
    def fetch_and_parse(
        self,
        base_url: str,
        user_agent: str = "*",
        session: Any = None,
        timeout: float = 10.0,
    ) -> RobotsTxtRules:
        """Fetch and parse robots.txt from a URL.
        
        Args:
            base_url: Base URL of the site
            user_agent: User-agent to match
            session: Optional HTTP session
            timeout: Request timeout
            
        Returns:
            Parsed rules (empty rules on error, allowing access)
        """
        parsed = urlparse(base_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        
        # Check cache
        cache_key = f"{robots_url}:{user_agent}"
        if cache_key in self._cache:
            rules, cached_at = self._cache[cache_key]
            if time.time() - cached_at < self.cache_ttl:
                return rules
        
        try:
            if session:
                response = session.get(robots_url, timeout=timeout)
                if response.status_code == 200:
                    content = response.text
                else:
                    content = ""
            else:
                import urllib.request
                try:
                    with urllib.request.urlopen(robots_url, timeout=timeout) as resp:
                        content = resp.read().decode("utf-8", errors="ignore")
                except Exception:
                    content = ""
            
            rules = self.parse(content, user_agent)
            
        except Exception as e:
            logger.debug(f"Could not fetch robots.txt from {robots_url}: {e}")
            # On error, allow access (fail open)
            rules = RobotsTxtRules(user_agent=user_agent)
        
        # Cache result
        self._cache[cache_key] = (rules, time.time())
        return rules
    
    def is_allowed(
        self,
        url: str,
        user_agent: str = "*",
        session: Any = None,
        timeout: float = 10.0,
    ) -> bool:
        """Check if a URL is allowed by robots.txt.
        
        Args:
            url: Full URL to check
            user_agent: User-agent string
            session: Optional HTTP session
            timeout: Request timeout
            
        Returns:
            True if allowed
        """
        parsed = urlparse(url)
        rules = self.fetch_and_parse(
            f"{parsed.scheme}://{parsed.netloc}",
            user_agent=user_agent,
            session=session,
            timeout=timeout,
        )
        return rules.is_allowed(parsed.path)
    
    def clear_cache(self) -> None:
        """Clear the robots.txt cache."""
        self._cache.clear()


# =============================================================================
# Web Connector Configuration
# =============================================================================

class WebConnectorConfig(ConnectorConfig):
    """Configuration for web connector.
    
    Attributes:
        user_agent: HTTP User-Agent header
        follow_redirects: Whether to follow redirects
        max_content_length: Maximum content length in bytes
        extract_links: Whether to extract links from content
        respect_robots_txt: Whether to check robots.txt before fetching
        robots_cache_ttl: Cache TTL for robots.txt in seconds
    """
    user_agent: str = Field(
        default="ARGUS-Bot/1.0",
        description="HTTP User-Agent header",
    )
    follow_redirects: bool = Field(
        default=True,
        description="Follow HTTP redirects",
    )
    max_content_length: int = Field(
        default=10_000_000,  # 10MB
        ge=1000,
        description="Maximum content length in bytes",
    )
    extract_links: bool = Field(
        default=False,
        description="Extract links from content",
    )
    respect_robots_txt: bool = Field(
        default=True,
        description="Respect robots.txt rules",
    )
    robots_cache_ttl: int = Field(
        default=3600,
        ge=0,
        description="Robots.txt cache TTL in seconds",
    )


# =============================================================================
# Web Connector
# =============================================================================

class WebConnector(BaseConnector):
    """Connector for fetching web content with robots.txt compliance.
    
    Fetches and parses web pages into Document objects.
    Supports HTML content extraction using BeautifulSoup.
    Respects robots.txt rules when configured.
    
    Example:
        >>> connector = WebConnector()
        >>> result = connector.fetch("https://example.com")
        >>> 
        >>> if result.success and result.documents:
        ...     doc = result.documents[0]
        ...     print(f"Title: {doc.title}")
        ...     print(f"Content length: {len(doc.content)}")
    """
    
    name = "web"
    description = "Fetch content from web URLs with robots.txt compliance"
    version = "1.1.0"
    
    def __init__(self, config: Optional[WebConnectorConfig] = None):
        """Initialize web connector.
        
        Args:
            config: Web connector configuration
        """
        super().__init__(config or WebConnectorConfig())
        self._session = None
        self._robots_parser = RobotsTxtParser(
            cache_ttl=self.web_config.robots_cache_ttl
        )
    
    @property
    def web_config(self) -> WebConnectorConfig:
        """Get typed config."""
        return self.config  # type: ignore
    
    def _get_session(self):
        """Get or create HTTP session."""
        if self._session is None:
            try:
                import httpx
                self._session = httpx.Client(
                    follow_redirects=self.web_config.follow_redirects,
                    timeout=self.config.timeout,
                    headers={"User-Agent": self.web_config.user_agent},
                )
            except ImportError:
                logger.warning("httpx not available, using urllib")
        return self._session
    
    def _check_robots_txt(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt.
        
        Args:
            url: URL to check
            
        Returns:
            True if allowed, False if disallowed
        """
        if not self.web_config.respect_robots_txt:
            return True
        
        return self._robots_parser.is_allowed(
            url=url,
            user_agent=self.web_config.user_agent,
            session=self._get_session(),
            timeout=min(self.config.timeout, 10.0),
        )
    
    def _get_crawl_delay(self, url: str) -> Optional[float]:
        """Get crawl delay from robots.txt.
        
        Args:
            url: URL to check
            
        Returns:
            Crawl delay in seconds, or None
        """
        if not self.web_config.respect_robots_txt:
            return None
        
        parsed = urlparse(url)
        rules = self._robots_parser.fetch_and_parse(
            f"{parsed.scheme}://{parsed.netloc}",
            user_agent=self.web_config.user_agent,
            session=self._get_session(),
        )
        return rules.crawl_delay
    
    def fetch(
        self,
        query: str,  # URL in this case
        max_results: int = 1,
        **kwargs: Any,
    ) -> ConnectorResult:
        """Fetch content from URL with robots.txt compliance.
        
        Args:
            query: URL to fetch
            max_results: Not used for web connector
            **kwargs: Additional options
                - skip_robots_check: Skip robots.txt check for this request
            
        Returns:
            ConnectorResult with document
        """
        url = query
        skip_robots = kwargs.get("skip_robots_check", False)
        
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return ConnectorResult.from_error(f"Invalid URL: {url}")
        
        if parsed.scheme not in ("http", "https"):
            return ConnectorResult.from_error(
                f"Unsupported URL scheme: {parsed.scheme}"
            )
        
        # Check robots.txt
        if not skip_robots and not self._check_robots_txt(url):
            logger.info(f"URL blocked by robots.txt: {url}")
            return ConnectorResult(
                success=False,
                error=f"Access blocked by robots.txt: {url}",
                metadata={"blocked_by": "robots.txt"},
            )
        
        # Respect crawl delay if specified
        if not skip_robots:
            delay = self._get_crawl_delay(url)
            if delay and delay > 0:
                logger.debug(f"Respecting crawl-delay of {delay}s")
                time.sleep(delay)
        
        try:
            # Fetch using httpx or fallback
            session = self._get_session()
            if session:
                response = session.get(url)
                content = response.text
                status = response.status_code
                content_type = response.headers.get("content-type", "")
            else:
                # Fallback to urllib
                import urllib.request
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": self.web_config.user_agent}
                )
                with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                    content = resp.read().decode("utf-8", errors="ignore")
                    status = resp.status
                    content_type = resp.headers.get("Content-Type", "")
            
            if status != 200:
                return ConnectorResult.from_error(f"HTTP {status}")
            
            # Check content length
            if len(content) > self.web_config.max_content_length:
                return ConnectorResult.from_error(
                    f"Content too large: {len(content)} bytes "
                    f"(max: {self.web_config.max_content_length})"
                )
            
            # Parse HTML
            title, text, links = self._parse_html(content)
            
            # Build metadata
            metadata: dict[str, Any] = {
                "domain": parsed.netloc,
                "content_length": len(content),
                "content_type": content_type,
            }
            
            if self.web_config.extract_links and links:
                metadata["links"] = links
            
            # Create document
            doc = Document(
                url=url,
                title=title or parsed.netloc,
                content=text,
                source_type=SourceType.HTML,
                metadata=metadata,
            )
            
            return ConnectorResult(
                success=True,
                documents=[doc],
                metadata={
                    "url": url,
                    "status": status,
                    "robots_checked": not skip_robots,
                },
            )
            
        except Exception as e:
            logger.error(f"Web fetch failed for {url}: {e}")
            return ConnectorResult.from_error(str(e))
    
    def fetch_multiple(
        self,
        urls: list[str],
        max_concurrent: int = 5,
        **kwargs: Any,
    ) -> list[ConnectorResult]:
        """Fetch multiple URLs.
        
        Args:
            urls: List of URLs to fetch
            max_concurrent: Maximum concurrent requests (not used in sync mode)
            **kwargs: Additional options passed to fetch()
            
        Returns:
            List of ConnectorResults
        """
        results = []
        for url in urls:
            result = self.fetch(url, **kwargs)
            results.append(result)
        return results
    
    def _parse_html(
        self,
        html: str,
    ) -> tuple[Optional[str], str, list[str]]:
        """Parse HTML content.
        
        Args:
            html: Raw HTML content
            
        Returns:
            (title, text content, links)
        """
        links: list[str] = []
        
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html, "lxml")
            
            # Extract links before removing elements
            if self.web_config.extract_links:
                for a_tag in soup.find_all("a", href=True):
                    href = a_tag["href"]
                    if href and not href.startswith(("#", "javascript:", "mailto:")):
                        links.append(href)
            
            # Remove scripts and styles
            for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            
            # Get title
            title = None
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)
            
            # Try to get main content
            main_content = (
                soup.find("main") or
                soup.find("article") or
                soup.find(class_=re.compile(r"content|main|article", re.I)) or
                soup.body or
                soup
            )
            
            # Get text
            text = main_content.get_text(separator="\n", strip=True)
            
            # Clean up whitespace
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r" {2,}", " ", text)
            
            return title, text.strip(), links
            
        except ImportError:
            # Fallback: basic regex extraction
            title_match = re.search(
                r"<title>(.*?)</title>",
                html,
                re.IGNORECASE | re.DOTALL
            )
            title = title_match.group(1).strip() if title_match else None
            
            # Extract links
            link_pattern = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
            links = [
                m for m in link_pattern.findall(html)
                if not m.startswith(("#", "javascript:", "mailto:"))
            ]
            
            # Remove tags
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text)
            
            return title, text.strip(), links
    
    def get_sitemaps(self, base_url: str) -> list[str]:
        """Get sitemap URLs from robots.txt.
        
        Args:
            base_url: Base URL of the site
            
        Returns:
            List of sitemap URLs
        """
        rules = self._robots_parser.fetch_and_parse(
            base_url,
            user_agent=self.web_config.user_agent,
            session=self._get_session(),
        )
        return rules.sitemaps
    
    def test_connection(self) -> bool:
        """Test connection to a known URL.
        
        Returns:
            True if successful
        """
        try:
            result = self.fetch(
                "https://httpbin.org/get",
                skip_robots_check=True
            )
            return result.success
        except Exception:
            return False
    
    def close(self) -> None:
        """Close HTTP session and clear caches."""
        if self._session:
            self._session.close()
            self._session = None
        self._robots_parser.clear_cache()
