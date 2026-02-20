"""
AgentMail Tool for ARGUS.

Create email inboxes for AI agents to send, receive, and manage messages.
Provides a full email management system for AI agents.
"""

from __future__ import annotations

import os
import uuid
import logging
import smtplib
import imaplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


@dataclass
class EmailMessage:
    """Represents an email message."""
    id: str
    from_addr: str
    to_addr: str
    subject: str
    body: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    read: bool = False
    attachments: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "from": self.from_addr,
            "to": self.to_addr,
            "subject": self.subject,
            "body": self.body[:500] + "..." if len(self.body) > 500 else self.body,
            "timestamp": self.timestamp.isoformat(),
            "read": self.read,
            "attachments": self.attachments,
        }


@dataclass 
class AgentInbox:
    """Virtual inbox for an AI agent."""
    inbox_id: str
    agent_name: str
    email_address: str
    messages: list[EmailMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_message(self, msg: EmailMessage) -> None:
        self.messages.append(msg)
    
    def get_unread(self) -> list[EmailMessage]:
        return [m for m in self.messages if not m.read]
    
    def mark_read(self, message_id: str) -> bool:
        for m in self.messages:
            if m.id == message_id:
                m.read = True
                return True
        return False


class AgentMailTool(BaseTool):
    """
    AgentMail - Email management system for AI agents.
    
    Create virtual inboxes, send/receive emails, and manage agent communications.
    Supports both virtual (in-memory) and real SMTP/IMAP connections.
    
    Example:
        >>> tool = AgentMailTool()
        >>> result = tool(action="create_inbox", agent_name="research_agent")
        >>> result = tool(action="send", inbox_id="...", to="user@example.com", 
        ...               subject="Report", body="Here's the report...")
        >>> result = tool(action="receive", inbox_id="...")
    """
    
    name = "agentmail"
    description = "Create email inboxes for AI agents to send, receive, and manage messages"
    category = ToolCategory.EXTERNAL_API
    version = "1.0.0"
    
    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: int = 587,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        imap_host: Optional[str] = None,
        imap_port: int = 993,
        domain: str = "agent.local",
        use_real_smtp: bool = False,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        # SMTP configuration
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST")
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user or os.getenv("SMTP_USER")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD")
        
        # IMAP configuration  
        self.imap_host = imap_host or os.getenv("IMAP_HOST")
        self.imap_port = imap_port
        
        self.domain = domain
        self.use_real_smtp = use_real_smtp and self.smtp_host is not None
        
        # Virtual inbox storage (in-memory)
        self._inboxes: dict[str, AgentInbox] = {}
        
        logger.debug(f"AgentMail initialized (real_smtp={self.use_real_smtp})")
    
    def execute(
        self,
        action: str = "list_inboxes",
        inbox_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        to: Optional[str] = None,
        subject: Optional[str] = None,
        body: Optional[str] = None,
        message_id: Optional[str] = None,
        limit: int = 20,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Execute AgentMail operations.
        
        Args:
            action: One of 'create_inbox', 'delete_inbox', 'list_inboxes', 
                   'send', 'receive', 'get_message', 'mark_read', 'search'
            inbox_id: Inbox identifier
            agent_name: Name for new agent inbox
            to: Recipient email address
            subject: Email subject
            body: Email body
            message_id: Specific message ID
            limit: Max messages to return
            
        Returns:
            ToolResult with operation result
        """
        actions = {
            "create_inbox": self._create_inbox,
            "delete_inbox": self._delete_inbox,
            "list_inboxes": self._list_inboxes,
            "send": self._send_email,
            "receive": self._receive_emails,
            "get_message": self._get_message,
            "mark_read": self._mark_read,
            "search": self._search_messages,
            "get_stats": self._get_stats,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            return actions[action](
                inbox_id=inbox_id,
                agent_name=agent_name,
                to=to,
                subject=subject,
                body=body,
                message_id=message_id,
                limit=limit,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"AgentMail error: {e}")
            return ToolResult.from_error(f"AgentMail error: {e}")
    
    def _create_inbox(
        self,
        agent_name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a new inbox for an agent."""
        if not agent_name:
            return ToolResult.from_error("agent_name is required")
        
        inbox_id = str(uuid.uuid4())[:8]
        email_address = f"{agent_name.lower().replace(' ', '_')}@{self.domain}"
        
        inbox = AgentInbox(
            inbox_id=inbox_id,
            agent_name=agent_name,
            email_address=email_address,
        )
        self._inboxes[inbox_id] = inbox
        
        logger.info(f"Created inbox {inbox_id} for agent {agent_name}")
        
        return ToolResult.from_data({
            "inbox_id": inbox_id,
            "agent_name": agent_name,
            "email_address": email_address,
            "message": f"Inbox created successfully",
        })
    
    def _delete_inbox(
        self,
        inbox_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete an inbox."""
        if not inbox_id or inbox_id not in self._inboxes:
            return ToolResult.from_error("Invalid inbox_id")
        
        del self._inboxes[inbox_id]
        return ToolResult.from_data({"message": f"Inbox {inbox_id} deleted"})
    
    def _list_inboxes(self, **kwargs) -> ToolResult:
        """List all agent inboxes."""
        inboxes = [
            {
                "inbox_id": inbox.inbox_id,
                "agent_name": inbox.agent_name,
                "email_address": inbox.email_address,
                "message_count": len(inbox.messages),
                "unread_count": len(inbox.get_unread()),
                "created_at": inbox.created_at.isoformat(),
            }
            for inbox in self._inboxes.values()
        ]
        return ToolResult.from_data({"inboxes": inboxes, "count": len(inboxes)})
    
    def _send_email(
        self,
        inbox_id: Optional[str] = None,
        to: Optional[str] = None,
        subject: Optional[str] = None,
        body: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Send an email from an agent inbox."""
        if not inbox_id or inbox_id not in self._inboxes:
            return ToolResult.from_error("Invalid inbox_id")
        if not to:
            return ToolResult.from_error("Recipient 'to' is required")
        if not subject:
            return ToolResult.from_error("Subject is required")
        if not body:
            return ToolResult.from_error("Body is required")
        
        inbox = self._inboxes[inbox_id]
        message_id = str(uuid.uuid4())[:12]
        
        if self.use_real_smtp:
            try:
                msg = MIMEMultipart()
                msg["From"] = inbox.email_address
                msg["To"] = to
                msg["Subject"] = subject
                msg.attach(MIMEText(body, "plain"))
                
                with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                    server.starttls()
                    if self.smtp_user and self.smtp_password:
                        server.login(self.smtp_user, self.smtp_password)
                    server.send_message(msg)
                
                logger.info(f"Email sent via SMTP: {message_id}")
            except Exception as e:
                return ToolResult.from_error(f"SMTP error: {e}")
        else:
            # Virtual send - store in sent folder simulation
            logger.info(f"Virtual email sent: {message_id} to {to}")
        
        return ToolResult.from_data({
            "message_id": message_id,
            "from": inbox.email_address,
            "to": to,
            "subject": subject,
            "status": "sent",
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    def _receive_emails(
        self,
        inbox_id: Optional[str] = None,
        limit: int = 20,
        unread_only: bool = False,
        **kwargs,
    ) -> ToolResult:
        """Receive emails for an inbox."""
        if not inbox_id or inbox_id not in self._inboxes:
            return ToolResult.from_error("Invalid inbox_id")
        
        inbox = self._inboxes[inbox_id]
        
        # Try real IMAP if configured
        if self.imap_host and self.smtp_user and self.smtp_password:
            try:
                with imaplib.IMAP4_SSL(self.imap_host, self.imap_port) as mail:
                    mail.login(self.smtp_user, self.smtp_password)
                    mail.select("INBOX")
                    
                    search_criteria = "UNSEEN" if unread_only else "ALL"
                    _, message_numbers = mail.search(None, search_criteria)
                    
                    messages = []
                    for num in message_numbers[0].split()[-limit:]:
                        _, msg_data = mail.fetch(num, "(RFC822)")
                        email_body = msg_data[0][1]
                        msg = email.message_from_bytes(email_body)
                        
                        body_text = ""
                        if msg.is_multipart():
                            for part in msg.walk():
                                if part.get_content_type() == "text/plain":
                                    body_text = part.get_payload(decode=True).decode()
                                    break
                        else:
                            body_text = msg.get_payload(decode=True).decode()
                        
                        email_msg = EmailMessage(
                            id=str(uuid.uuid4())[:12],
                            from_addr=msg["From"],
                            to_addr=msg["To"],
                            subject=msg["Subject"] or "",
                            body=body_text,
                        )
                        inbox.add_message(email_msg)
                        messages.append(email_msg.to_dict())
                    
                    return ToolResult.from_data({
                        "inbox_id": inbox_id,
                        "messages": messages,
                        "count": len(messages),
                    })
            except Exception as e:
                logger.warning(f"IMAP fetch failed: {e}, using virtual inbox")
        
        # Return virtual inbox messages
        messages = inbox.get_unread() if unread_only else inbox.messages
        messages = messages[-limit:]
        
        return ToolResult.from_data({
            "inbox_id": inbox_id,
            "messages": [m.to_dict() for m in messages],
            "count": len(messages),
        })
    
    def _get_message(
        self,
        inbox_id: Optional[str] = None,
        message_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get a specific message by ID."""
        if not inbox_id or inbox_id not in self._inboxes:
            return ToolResult.from_error("Invalid inbox_id")
        if not message_id:
            return ToolResult.from_error("message_id is required")
        
        inbox = self._inboxes[inbox_id]
        for msg in inbox.messages:
            if msg.id == message_id:
                return ToolResult.from_data({"message": msg.to_dict()})
        
        return ToolResult.from_error(f"Message {message_id} not found")
    
    def _mark_read(
        self,
        inbox_id: Optional[str] = None,
        message_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Mark a message as read."""
        if not inbox_id or inbox_id not in self._inboxes:
            return ToolResult.from_error("Invalid inbox_id")
        if not message_id:
            return ToolResult.from_error("message_id is required")
        
        inbox = self._inboxes[inbox_id]
        if inbox.mark_read(message_id):
            return ToolResult.from_data({"message": f"Message {message_id} marked as read"})
        return ToolResult.from_error(f"Message {message_id} not found")
    
    def _search_messages(
        self,
        inbox_id: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 20,
        **kwargs,
    ) -> ToolResult:
        """Search messages in an inbox."""
        if not inbox_id or inbox_id not in self._inboxes:
            return ToolResult.from_error("Invalid inbox_id")
        if not query:
            return ToolResult.from_error("Search query is required")
        
        inbox = self._inboxes[inbox_id]
        query_lower = query.lower()
        
        matches = [
            m for m in inbox.messages
            if query_lower in m.subject.lower() or query_lower in m.body.lower()
        ][:limit]
        
        return ToolResult.from_data({
            "inbox_id": inbox_id,
            "query": query,
            "messages": [m.to_dict() for m in matches],
            "count": len(matches),
        })
    
    def _get_stats(
        self,
        inbox_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get statistics for an inbox or all inboxes."""
        if inbox_id:
            if inbox_id not in self._inboxes:
                return ToolResult.from_error("Invalid inbox_id")
            inbox = self._inboxes[inbox_id]
            return ToolResult.from_data({
                "inbox_id": inbox_id,
                "agent_name": inbox.agent_name,
                "total_messages": len(inbox.messages),
                "unread_messages": len(inbox.get_unread()),
                "created_at": inbox.created_at.isoformat(),
            })
        
        total_messages = sum(len(i.messages) for i in self._inboxes.values())
        total_unread = sum(len(i.get_unread()) for i in self._inboxes.values())
        
        return ToolResult.from_data({
            "total_inboxes": len(self._inboxes),
            "total_messages": total_messages,
            "total_unread": total_unread,
        })
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create_inbox", "delete_inbox", "list_inboxes", 
                                "send", "receive", "get_message", "mark_read", 
                                "search", "get_stats"],
                        "description": "Operation to perform",
                    },
                    "inbox_id": {"type": "string", "description": "Inbox identifier"},
                    "agent_name": {"type": "string", "description": "Name for new agent"},
                    "to": {"type": "string", "description": "Recipient email"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"},
                    "message_id": {"type": "string", "description": "Message ID"},
                    "limit": {"type": "integer", "description": "Max results"},
                },
                "required": ["action"],
            },
        }
