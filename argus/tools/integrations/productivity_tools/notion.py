"""
Notion Tool for ARGUS.

Documentation, knowledge management, and database operations.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class NotionTool(BaseTool):
    """
    Notion - Documentation and knowledge management.
    
    Features:
    - Page CRUD operations
    - Database operations
    - Block manipulation
    - Search functionality
    - User management
    
    Example:
        >>> tool = NotionTool()
        >>> result = tool(action="create_page", parent_id="...", title="Meeting Notes")
        >>> result = tool(action="query_database", database_id="...")
    """
    
    name = "notion"
    description = "Documentation and knowledge management with Notion"
    category = ToolCategory.PRODUCTIVITY
    version = "1.0.0"
    
    BASE_URL = "https://api.notion.com/v1"
    NOTION_VERSION = "2022-06-28"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.api_key = api_key or os.getenv("NOTION_API_KEY")
        
        self._session = None
        
        if not self.api_key:
            logger.warning("No Notion API key provided")
        
        logger.debug("Notion tool initialized")
    
    def _get_session(self):
        """Get HTTP session with authentication."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Notion-Version": self.NOTION_VERSION,
            })
        return self._session
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> dict:
        """Make API request to Notion."""
        session = self._get_session()
        url = f"{self.BASE_URL}{endpoint}"
        
        response = session.request(
            method=method,
            url=url,
            json=data,
            params=params,
        )
        
        if response.status_code >= 400:
            try:
                error_data = response.json()
                raise Exception(error_data.get("message", f"HTTP {response.status_code}"))
            except ValueError:
                raise Exception(f"HTTP {response.status_code}")
        
        return response.json() if response.text else {}
    
    def execute(
        self,
        action: str = "search",
        page_id: Optional[str] = None,
        database_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        parent_type: str = "page_id",
        title: Optional[str] = None,
        content: Optional[str] = None,
        properties: Optional[dict] = None,
        filter: Optional[dict] = None,
        sorts: Optional[list] = None,
        query: Optional[str] = None,
        blocks: Optional[list] = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute Notion operations."""
        actions = {
            # Page operations
            "get_page": self._get_page,
            "create_page": self._create_page,
            "update_page": self._update_page,
            "archive_page": self._archive_page,
            
            # Database operations
            "get_database": self._get_database,
            "query_database": self._query_database,
            "create_database": self._create_database,
            
            # Block operations
            "get_blocks": self._get_blocks,
            "append_blocks": self._append_blocks,
            "delete_block": self._delete_block,
            
            # Search
            "search": self._search,
            
            # Users
            "list_users": self._list_users,
            "get_user": self._get_user,
            "me": self._me,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            if not self.api_key:
                return ToolResult.from_error("Notion API key not configured")
            
            return actions[action](
                page_id=page_id,
                database_id=database_id,
                parent_id=parent_id,
                parent_type=parent_type,
                title=title,
                content=content,
                properties=properties,
                filter=filter,
                sorts=sorts,
                query=query,
                blocks=blocks,
                limit=limit,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Notion error: {e}")
            return ToolResult.from_error(f"Notion error: {e}")
    
    def _extract_title(self, page: dict) -> str:
        """Extract title from page properties."""
        props = page.get("properties", {})
        
        # Try common title property names
        for key in ["Name", "Title", "title", "name"]:
            if key in props:
                prop = props[key]
                if prop.get("type") == "title":
                    title_content = prop.get("title", [])
                    if title_content:
                        return title_content[0].get("plain_text", "")
        
        # Try first title type property
        for prop in props.values():
            if prop.get("type") == "title":
                title_content = prop.get("title", [])
                if title_content:
                    return title_content[0].get("plain_text", "")
        
        return ""
    
    # Page operations
    def _get_page(
        self,
        page_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get page details."""
        if not page_id:
            return ToolResult.from_error("page_id is required")
        
        response = self._request("GET", f"/pages/{page_id}")
        
        return ToolResult.from_data({
            "page": {
                "id": response.get("id"),
                "title": self._extract_title(response),
                "url": response.get("url"),
                "created_time": response.get("created_time"),
                "last_edited_time": response.get("last_edited_time"),
                "properties": response.get("properties"),
            }
        })
    
    def _create_page(
        self,
        parent_id: Optional[str] = None,
        parent_type: str = "page_id",
        title: Optional[str] = None,
        content: Optional[str] = None,
        properties: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a new page."""
        if not parent_id:
            return ToolResult.from_error("parent_id is required")
        
        data = {
            "parent": {parent_type: parent_id},
        }
        
        # Handle properties
        if properties:
            data["properties"] = properties
        elif title:
            # Default title property for pages/databases
            if parent_type == "database_id":
                data["properties"] = {
                    "Name": {
                        "title": [{"text": {"content": title}}]
                    }
                }
            else:
                data["properties"] = {
                    "title": {
                        "title": [{"text": {"content": title}}]
                    }
                }
        
        # Handle content as blocks
        if content:
            data["children"] = [
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": content}}]
                    }
                }
            ]
        
        response = self._request("POST", "/pages", data=data)
        
        return ToolResult.from_data({
            "page_id": response.get("id"),
            "url": response.get("url"),
            "created": True,
        })
    
    def _update_page(
        self,
        page_id: Optional[str] = None,
        properties: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Update page properties."""
        if not page_id:
            return ToolResult.from_error("page_id is required")
        if not properties:
            return ToolResult.from_error("properties is required")
        
        data = {"properties": properties}
        
        response = self._request("PATCH", f"/pages/{page_id}", data=data)
        
        return ToolResult.from_data({
            "page_id": response.get("id"),
            "updated": True,
        })
    
    def _archive_page(
        self,
        page_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Archive (delete) a page."""
        if not page_id:
            return ToolResult.from_error("page_id is required")
        
        self._request("PATCH", f"/pages/{page_id}", data={"archived": True})
        
        return ToolResult.from_data({
            "page_id": page_id,
            "archived": True,
        })
    
    # Database operations
    def _get_database(
        self,
        database_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get database details."""
        if not database_id:
            return ToolResult.from_error("database_id is required")
        
        response = self._request("GET", f"/databases/{database_id}")
        
        # Extract property schema
        properties = {}
        for name, prop in response.get("properties", {}).items():
            properties[name] = {
                "type": prop.get("type"),
                "id": prop.get("id"),
            }
        
        return ToolResult.from_data({
            "database": {
                "id": response.get("id"),
                "title": self._extract_title(response),
                "url": response.get("url"),
                "properties": properties,
            }
        })
    
    def _query_database(
        self,
        database_id: Optional[str] = None,
        filter: Optional[dict] = None,
        sorts: Optional[list] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """Query a database."""
        if not database_id:
            return ToolResult.from_error("database_id is required")
        
        data = {"page_size": min(limit, 100)}
        
        if filter:
            data["filter"] = filter
        
        if sorts:
            data["sorts"] = sorts
        
        response = self._request("POST", f"/databases/{database_id}/query", data=data)
        
        results = []
        for page in response.get("results", []):
            results.append({
                "id": page.get("id"),
                "title": self._extract_title(page),
                "url": page.get("url"),
                "properties": page.get("properties"),
            })
        
        return ToolResult.from_data({
            "database_id": database_id,
            "results": results,
            "count": len(results),
            "has_more": response.get("has_more", False),
        })
    
    def _create_database(
        self,
        parent_id: Optional[str] = None,
        title: Optional[str] = None,
        properties: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a new database."""
        if not parent_id:
            return ToolResult.from_error("parent_id is required")
        if not title:
            return ToolResult.from_error("title is required")
        
        data = {
            "parent": {"page_id": parent_id},
            "title": [{"type": "text", "text": {"content": title}}],
            "properties": properties or {
                "Name": {"title": {}},
            },
        }
        
        response = self._request("POST", "/databases", data=data)
        
        return ToolResult.from_data({
            "database_id": response.get("id"),
            "url": response.get("url"),
            "created": True,
        })
    
    # Block operations
    def _get_blocks(
        self,
        page_id: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """Get blocks (content) of a page."""
        if not page_id:
            return ToolResult.from_error("page_id (block_id) is required")
        
        response = self._request(
            "GET",
            f"/blocks/{page_id}/children",
            params={"page_size": min(limit, 100)},
        )
        
        blocks = []
        for block in response.get("results", []):
            block_type = block.get("type")
            block_data = {
                "id": block.get("id"),
                "type": block_type,
                "has_children": block.get("has_children", False),
            }
            
            # Extract text content for common block types
            if block_type in block:
                type_data = block[block_type]
                rich_text = type_data.get("rich_text", [])
                if rich_text:
                    block_data["text"] = "".join(
                        t.get("plain_text", "") for t in rich_text
                    )
            
            blocks.append(block_data)
        
        return ToolResult.from_data({
            "page_id": page_id,
            "blocks": blocks,
            "count": len(blocks),
        })
    
    def _append_blocks(
        self,
        page_id: Optional[str] = None,
        blocks: Optional[list] = None,
        content: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Append blocks to a page."""
        if not page_id:
            return ToolResult.from_error("page_id (block_id) is required")
        
        children = []
        
        if blocks:
            children = blocks
        elif content:
            # Convert simple content to paragraph block
            children = [
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": content}}]
                    }
                }
            ]
        else:
            return ToolResult.from_error("blocks or content is required")
        
        response = self._request(
            "PATCH",
            f"/blocks/{page_id}/children",
            data={"children": children},
        )
        
        return ToolResult.from_data({
            "page_id": page_id,
            "appended": len(response.get("results", [])),
        })
    
    def _delete_block(
        self,
        **kwargs,
    ) -> ToolResult:
        """Delete a block."""
        block_id = kwargs.get("block_id")
        if not block_id:
            return ToolResult.from_error("block_id is required")
        
        self._request("DELETE", f"/blocks/{block_id}")
        
        return ToolResult.from_data({
            "block_id": block_id,
            "deleted": True,
        })
    
    # Search
    def _search(
        self,
        query: Optional[str] = None,
        filter: Optional[dict] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """Search pages and databases."""
        data = {"page_size": min(limit, 100)}
        
        if query:
            data["query"] = query
        
        if filter:
            data["filter"] = filter
        
        response = self._request("POST", "/search", data=data)
        
        results = []
        for item in response.get("results", []):
            results.append({
                "id": item.get("id"),
                "object": item.get("object"),
                "title": self._extract_title(item),
                "url": item.get("url"),
            })
        
        return ToolResult.from_data({
            "query": query,
            "results": results,
            "count": len(results),
        })
    
    # User operations
    def _list_users(self, limit: int = 100, **kwargs) -> ToolResult:
        """List users."""
        response = self._request(
            "GET",
            "/users",
            params={"page_size": min(limit, 100)},
        )
        
        users = [
            {
                "id": u.get("id"),
                "name": u.get("name"),
                "type": u.get("type"),
                "avatar_url": u.get("avatar_url"),
            }
            for u in response.get("results", [])
        ]
        
        return ToolResult.from_data({
            "users": users,
            "count": len(users),
        })
    
    def _get_user(self, **kwargs) -> ToolResult:
        """Get user details."""
        user_id = kwargs.get("user_id")
        if not user_id:
            return ToolResult.from_error("user_id is required")
        
        response = self._request("GET", f"/users/{user_id}")
        
        return ToolResult.from_data({
            "user": {
                "id": response.get("id"),
                "name": response.get("name"),
                "type": response.get("type"),
                "avatar_url": response.get("avatar_url"),
            }
        })
    
    def _me(self, **kwargs) -> ToolResult:
        """Get the bot user."""
        response = self._request("GET", "/users/me")
        
        return ToolResult.from_data({
            "bot": {
                "id": response.get("id"),
                "name": response.get("name"),
                "type": response.get("type"),
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
                            "get_page", "create_page", "update_page", "archive_page",
                            "get_database", "query_database", "create_database",
                            "get_blocks", "append_blocks", "delete_block",
                            "search",
                            "list_users", "get_user", "me",
                        ],
                    },
                    "page_id": {"type": "string"},
                    "database_id": {"type": "string"},
                    "parent_id": {"type": "string"},
                    "parent_type": {"type": "string", "enum": ["page_id", "database_id"]},
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "properties": {"type": "object"},
                    "filter": {"type": "object"},
                    "sorts": {"type": "array"},
                    "query": {"type": "string"},
                    "blocks": {"type": "array"},
                    "limit": {"type": "integer", "default": 100},
                },
                "required": ["action"],
            },
        }
