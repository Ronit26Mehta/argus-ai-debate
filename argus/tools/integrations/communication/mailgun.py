"""
Mailgun Tool for ARGUS.

Email sending and management integration.
"""

from __future__ import annotations

import os
import base64
import logging
from typing import Optional, Any

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class MailgunTool(BaseTool):
    """
    Mailgun - Email sending and management.
    
    Features:
    - Send emails (text and HTML)
    - Email templates
    - Mailing lists management
    - Event tracking
    - Domain validation
    
    Example:
        >>> tool = MailgunTool(domain="mg.example.com")
        >>> result = tool(action="send", to="user@example.com", subject="Hello", text="Hi there!")
    """
    
    name = "mailgun"
    description = "Email sending and management with Mailgun"
    category = ToolCategory.COMMUNICATION
    version = "1.0.0"
    
    BASE_URL = "https://api.mailgun.net/v3"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        domain: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.api_key = api_key or os.getenv("MAILGUN_API_KEY")
        self.domain = domain or os.getenv("MAILGUN_DOMAIN")
        
        self._session = None
        
        if not self.api_key:
            logger.warning("No Mailgun API key provided")
        
        logger.debug(f"Mailgun tool initialized (domain={self.domain})")
    
    def _get_session(self):
        """Get HTTP session with authentication."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            
            auth_str = f"api:{self.api_key}"
            auth_bytes = base64.b64encode(auth_str.encode()).decode()
            
            self._session.headers.update({
                "Authorization": f"Basic {auth_bytes}",
            })
        return self._session
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
        files: Optional[dict] = None,
    ) -> dict:
        """Make API request to Mailgun."""
        session = self._get_session()
        url = f"{self.BASE_URL}{endpoint}"
        
        response = session.request(
            method=method,
            url=url,
            data=data,
            params=params,
            files=files,
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
        action: str = "send",
        to: Optional[str] = None,
        to_list: Optional[list] = None,
        from_email: Optional[str] = None,
        subject: Optional[str] = None,
        text: Optional[str] = None,
        html: Optional[str] = None,
        cc: Optional[str] = None,
        bcc: Optional[str] = None,
        reply_to: Optional[str] = None,
        template: Optional[str] = None,
        template_variables: Optional[dict] = None,
        tags: Optional[list] = None,
        list_address: Optional[str] = None,
        member_email: Optional[str] = None,
        member_name: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute Mailgun operations."""
        actions = {
            # Email operations
            "send": self._send_email,
            "send_template": self._send_template,
            
            # Mailing list operations
            "list_mailing_lists": self._list_mailing_lists,
            "create_mailing_list": self._create_mailing_list,
            "get_mailing_list": self._get_mailing_list,
            "delete_mailing_list": self._delete_mailing_list,
            "add_member": self._add_member,
            "list_members": self._list_members,
            "remove_member": self._remove_member,
            
            # Event operations
            "get_events": self._get_events,
            
            # Domain operations
            "list_domains": self._list_domains,
            "get_domain": self._get_domain,
            "verify_domain": self._verify_domain,
            
            # Stats
            "get_stats": self._get_stats,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            if not self.api_key:
                return ToolResult.from_error("Mailgun API key not configured")
            
            return actions[action](
                to=to,
                to_list=to_list,
                from_email=from_email,
                subject=subject,
                text=text,
                html=html,
                cc=cc,
                bcc=bcc,
                reply_to=reply_to,
                template=template,
                template_variables=template_variables,
                tags=tags,
                list_address=list_address,
                member_email=member_email,
                member_name=member_name,
                event_type=event_type,
                limit=limit,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Mailgun error: {e}")
            return ToolResult.from_error(f"Mailgun error: {e}")
    
    # Email operations
    def _send_email(
        self,
        to: Optional[str] = None,
        to_list: Optional[list] = None,
        from_email: Optional[str] = None,
        subject: Optional[str] = None,
        text: Optional[str] = None,
        html: Optional[str] = None,
        cc: Optional[str] = None,
        bcc: Optional[str] = None,
        reply_to: Optional[str] = None,
        tags: Optional[list] = None,
        **kwargs,
    ) -> ToolResult:
        """Send an email."""
        if not self.domain:
            return ToolResult.from_error("Mailgun domain not configured")
        
        recipients = to_list or ([to] if to else None)
        if not recipients:
            return ToolResult.from_error("to or to_list is required")
        if not subject:
            return ToolResult.from_error("subject is required")
        if not text and not html:
            return ToolResult.from_error("text or html is required")
        
        data = {
            "from": from_email or f"mailgun@{self.domain}",
            "to": recipients if isinstance(recipients, list) else [recipients],
            "subject": subject,
        }
        
        if text:
            data["text"] = text
        if html:
            data["html"] = html
        if cc:
            data["cc"] = cc
        if bcc:
            data["bcc"] = bcc
        if reply_to:
            data["h:Reply-To"] = reply_to
        if tags:
            data["o:tag"] = tags
        
        response = self._request("POST", f"/{self.domain}/messages", data=data)
        
        return ToolResult.from_data({
            "id": response.get("id"),
            "message": response.get("message"),
            "sent": True,
        })
    
    def _send_template(
        self,
        to: Optional[str] = None,
        to_list: Optional[list] = None,
        from_email: Optional[str] = None,
        subject: Optional[str] = None,
        template: Optional[str] = None,
        template_variables: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Send an email using a template."""
        if not self.domain:
            return ToolResult.from_error("Mailgun domain not configured")
        
        recipients = to_list or ([to] if to else None)
        if not recipients:
            return ToolResult.from_error("to or to_list is required")
        if not template:
            return ToolResult.from_error("template is required")
        
        data = {
            "from": from_email or f"mailgun@{self.domain}",
            "to": recipients if isinstance(recipients, list) else [recipients],
            "subject": subject or "",
            "template": template,
        }
        
        if template_variables:
            import json
            data["h:X-Mailgun-Variables"] = json.dumps(template_variables)
        
        response = self._request("POST", f"/{self.domain}/messages", data=data)
        
        return ToolResult.from_data({
            "id": response.get("id"),
            "message": response.get("message"),
            "sent": True,
        })
    
    # Mailing list operations
    def _list_mailing_lists(self, limit: int = 100, **kwargs) -> ToolResult:
        """List all mailing lists."""
        response = self._request(
            "GET",
            "/lists/pages",
            params={"limit": limit},
        )
        
        lists = [
            {
                "address": ml.get("address"),
                "name": ml.get("name"),
                "description": ml.get("description"),
                "members_count": ml.get("members_count"),
            }
            for ml in response.get("items", [])
        ]
        
        return ToolResult.from_data({
            "mailing_lists": lists,
            "count": len(lists),
        })
    
    def _create_mailing_list(
        self,
        list_address: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a mailing list."""
        if not list_address:
            return ToolResult.from_error("list_address is required")
        
        data = {
            "address": list_address,
        }
        
        name = kwargs.get("name")
        description = kwargs.get("description")
        
        if name:
            data["name"] = name
        if description:
            data["description"] = description
        
        response = self._request("POST", "/lists", data=data)
        
        return ToolResult.from_data({
            "list": response.get("list"),
            "created": True,
        })
    
    def _get_mailing_list(
        self,
        list_address: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get mailing list details."""
        if not list_address:
            return ToolResult.from_error("list_address is required")
        
        response = self._request("GET", f"/lists/{list_address}")
        
        return ToolResult.from_data({"list": response.get("list")})
    
    def _delete_mailing_list(
        self,
        list_address: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete a mailing list."""
        if not list_address:
            return ToolResult.from_error("list_address is required")
        
        self._request("DELETE", f"/lists/{list_address}")
        
        return ToolResult.from_data({
            "list_address": list_address,
            "deleted": True,
        })
    
    def _add_member(
        self,
        list_address: Optional[str] = None,
        member_email: Optional[str] = None,
        member_name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Add a member to a mailing list."""
        if not list_address:
            return ToolResult.from_error("list_address is required")
        if not member_email:
            return ToolResult.from_error("member_email is required")
        
        data = {
            "address": member_email,
            "subscribed": True,
        }
        
        if member_name:
            data["name"] = member_name
        
        response = self._request(
            "POST",
            f"/lists/{list_address}/members",
            data=data,
        )
        
        return ToolResult.from_data({
            "member": response.get("member"),
            "added": True,
        })
    
    def _list_members(
        self,
        list_address: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """List members of a mailing list."""
        if not list_address:
            return ToolResult.from_error("list_address is required")
        
        response = self._request(
            "GET",
            f"/lists/{list_address}/members/pages",
            params={"limit": limit},
        )
        
        members = [
            {
                "address": m.get("address"),
                "name": m.get("name"),
                "subscribed": m.get("subscribed"),
            }
            for m in response.get("items", [])
        ]
        
        return ToolResult.from_data({
            "list_address": list_address,
            "members": members,
            "count": len(members),
        })
    
    def _remove_member(
        self,
        list_address: Optional[str] = None,
        member_email: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Remove a member from a mailing list."""
        if not list_address:
            return ToolResult.from_error("list_address is required")
        if not member_email:
            return ToolResult.from_error("member_email is required")
        
        self._request("DELETE", f"/lists/{list_address}/members/{member_email}")
        
        return ToolResult.from_data({
            "list_address": list_address,
            "member_email": member_email,
            "removed": True,
        })
    
    # Event operations
    def _get_events(
        self,
        event_type: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> ToolResult:
        """Get email events."""
        if not self.domain:
            return ToolResult.from_error("Mailgun domain not configured")
        
        params = {"limit": limit}
        
        if event_type:
            params["event"] = event_type
        
        response = self._request(
            "GET",
            f"/{self.domain}/events",
            params=params,
        )
        
        events = [
            {
                "id": e.get("id"),
                "event": e.get("event"),
                "timestamp": e.get("timestamp"),
                "recipient": e.get("recipient"),
                "message_id": e.get("message", {}).get("headers", {}).get("message-id"),
            }
            for e in response.get("items", [])
        ]
        
        return ToolResult.from_data({
            "events": events,
            "count": len(events),
        })
    
    # Domain operations
    def _list_domains(self, limit: int = 100, **kwargs) -> ToolResult:
        """List all domains."""
        response = self._request(
            "GET",
            "/domains",
            params={"limit": limit},
        )
        
        domains = [
            {
                "name": d.get("name"),
                "state": d.get("state"),
                "type": d.get("type"),
                "created_at": d.get("created_at"),
            }
            for d in response.get("items", [])
        ]
        
        return ToolResult.from_data({
            "domains": domains,
            "count": len(domains),
        })
    
    def _get_domain(self, **kwargs) -> ToolResult:
        """Get domain details."""
        domain = kwargs.get("domain_name") or self.domain
        if not domain:
            return ToolResult.from_error("domain is required")
        
        response = self._request("GET", f"/domains/{domain}")
        
        return ToolResult.from_data({
            "domain": response.get("domain"),
            "receiving_dns_records": response.get("receiving_dns_records"),
            "sending_dns_records": response.get("sending_dns_records"),
        })
    
    def _verify_domain(self, **kwargs) -> ToolResult:
        """Verify domain DNS records."""
        domain = kwargs.get("domain_name") or self.domain
        if not domain:
            return ToolResult.from_error("domain is required")
        
        response = self._request("PUT", f"/domains/{domain}/verify")
        
        return ToolResult.from_data({
            "domain": response.get("domain"),
            "verified": True,
        })
    
    # Stats
    def _get_stats(self, limit: int = 100, **kwargs) -> ToolResult:
        """Get email statistics."""
        if not self.domain:
            return ToolResult.from_error("Mailgun domain not configured")
        
        event_types = kwargs.get("event_types", ["accepted", "delivered", "failed"])
        
        response = self._request(
            "GET",
            f"/{self.domain}/stats/total",
            params={
                "event": event_types,
                "duration": kwargs.get("duration", "1m"),
            },
        )
        
        return ToolResult.from_data({
            "stats": response.get("stats", []),
            "start": response.get("start"),
            "end": response.get("end"),
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
                            "send", "send_template",
                            "list_mailing_lists", "create_mailing_list",
                            "get_mailing_list", "delete_mailing_list",
                            "add_member", "list_members", "remove_member",
                            "get_events",
                            "list_domains", "get_domain", "verify_domain",
                            "get_stats",
                        ],
                    },
                    "to": {"type": "string"},
                    "to_list": {"type": "array", "items": {"type": "string"}},
                    "from_email": {"type": "string"},
                    "subject": {"type": "string"},
                    "text": {"type": "string"},
                    "html": {"type": "string"},
                    "template": {"type": "string"},
                    "template_variables": {"type": "object"},
                    "list_address": {"type": "string"},
                    "member_email": {"type": "string"},
                    "event_type": {"type": "string"},
                    "limit": {"type": "integer", "default": 100},
                },
                "required": ["action"],
            },
        }
