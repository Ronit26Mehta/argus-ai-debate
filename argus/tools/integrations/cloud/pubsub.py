"""
Pub/Sub Tool for ARGUS.

Publish, pull, and acknowledge messages from Google Cloud Pub/Sub.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from argus.tools.base import BaseTool, ToolResult, ToolConfig, ToolCategory

logger = logging.getLogger(__name__)


class PubSubTool(BaseTool):
    """
    Google Cloud Pub/Sub - Message queue operations.
    
    Features:
    - Publish messages to topics
    - Pull messages from subscriptions
    - Acknowledge message processing
    - Topic and subscription management
    
    Example:
        >>> tool = PubSubTool(project_id="my-project")
        >>> result = tool(action="publish", topic="my-topic", message="Hello!")
        >>> result = tool(action="pull", subscription="my-sub")
    """
    
    name = "pubsub"
    description = "Publish, pull, and acknowledge messages from Google Cloud Pub/Sub"
    category = ToolCategory.EXTERNAL_API
    version = "1.0.0"
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
        config: Optional[ToolConfig] = None,
    ):
        super().__init__(config)
        
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.credentials_path = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self._publisher = None
        self._subscriber = None
        
        logger.debug(f"PubSub initialized for project {self.project_id}")
    
    def _get_publisher(self):
        """Lazy-load publisher client."""
        if self._publisher is None:
            try:
                from google.cloud import pubsub_v1
                self._publisher = pubsub_v1.PublisherClient()
            except ImportError:
                raise ImportError("google-cloud-pubsub not installed. Run: pip install google-cloud-pubsub")
        return self._publisher
    
    def _get_subscriber(self):
        """Lazy-load subscriber client."""
        if self._subscriber is None:
            try:
                from google.cloud import pubsub_v1
                self._subscriber = pubsub_v1.SubscriberClient()
            except ImportError:
                raise ImportError("google-cloud-pubsub not installed. Run: pip install google-cloud-pubsub")
        return self._subscriber
    
    def execute(
        self,
        action: str = "list_topics",
        topic: Optional[str] = None,
        subscription: Optional[str] = None,
        message: Optional[str] = None,
        messages: Optional[list] = None,
        attributes: Optional[dict] = None,
        ack_ids: Optional[list] = None,
        max_messages: int = 10,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Execute Pub/Sub operations.
        
        Args:
            action: Operation to perform
            topic: Topic name
            subscription: Subscription name
            message: Message data to publish
            messages: List of messages to publish in batch
            attributes: Message attributes
            ack_ids: Acknowledgment IDs
            max_messages: Max messages to pull
            
        Returns:
            ToolResult with operation result
        """
        actions = {
            "publish": self._publish,
            "publish_batch": self._publish_batch,
            "pull": self._pull,
            "acknowledge": self._acknowledge,
            "list_topics": self._list_topics,
            "list_subscriptions": self._list_subscriptions,
            "create_topic": self._create_topic,
            "delete_topic": self._delete_topic,
            "create_subscription": self._create_subscription,
            "delete_subscription": self._delete_subscription,
            "get_topic": self._get_topic,
            "get_subscription": self._get_subscription,
        }
        
        if action not in actions:
            return ToolResult.from_error(
                f"Unknown action: {action}. Available: {list(actions.keys())}"
            )
        
        try:
            return actions[action](
                topic=topic,
                subscription=subscription,
                message=message,
                messages=messages or [],
                attributes=attributes or {},
                ack_ids=ack_ids or [],
                max_messages=max_messages,
                **kwargs,
            )
        except ImportError as e:
            return ToolResult.from_error(str(e))
        except Exception as e:
            logger.error(f"PubSub error: {e}")
            return ToolResult.from_error(f"PubSub error: {e}")
    
    def _topic_path(self, topic: str) -> str:
        """Get full topic path."""
        return f"projects/{self.project_id}/topics/{topic}"
    
    def _subscription_path(self, subscription: str) -> str:
        """Get full subscription path."""
        return f"projects/{self.project_id}/subscriptions/{subscription}"
    
    def _publish(
        self,
        topic: Optional[str] = None,
        message: Optional[str] = None,
        attributes: Optional[dict] = None,
        **kwargs,
    ) -> ToolResult:
        """Publish a message to a topic."""
        if not topic:
            return ToolResult.from_error("topic is required")
        if not message:
            return ToolResult.from_error("message is required")
        
        publisher = self._get_publisher()
        topic_path = self._topic_path(topic)
        
        # Encode message
        if isinstance(message, dict):
            data = json.dumps(message).encode("utf-8")
        else:
            data = str(message).encode("utf-8")
        
        # Publish
        future = publisher.publish(
            topic_path,
            data,
            **(attributes or {}),
        )
        message_id = future.result()
        
        return ToolResult.from_data({
            "message_id": message_id,
            "topic": topic,
            "published": True,
        })
    
    def _publish_batch(
        self,
        topic: Optional[str] = None,
        messages: Optional[list] = None,
        **kwargs,
    ) -> ToolResult:
        """Publish multiple messages in batch."""
        if not topic:
            return ToolResult.from_error("topic is required")
        if not messages:
            return ToolResult.from_error("messages list is required")
        
        publisher = self._get_publisher()
        topic_path = self._topic_path(topic)
        
        futures = []
        for msg in messages:
            if isinstance(msg, dict):
                data = msg.get("data", "")
                attrs = msg.get("attributes", {})
            else:
                data = str(msg)
                attrs = {}
            
            if isinstance(data, dict):
                data = json.dumps(data).encode("utf-8")
            else:
                data = str(data).encode("utf-8")
            
            future = publisher.publish(topic_path, data, **attrs)
            futures.append(future)
        
        message_ids = [f.result() for f in futures]
        
        return ToolResult.from_data({
            "message_ids": message_ids,
            "topic": topic,
            "count": len(message_ids),
        })
    
    def _pull(
        self,
        subscription: Optional[str] = None,
        max_messages: int = 10,
        **kwargs,
    ) -> ToolResult:
        """Pull messages from a subscription."""
        if not subscription:
            return ToolResult.from_error("subscription is required")
        
        subscriber = self._get_subscriber()
        subscription_path = self._subscription_path(subscription)
        
        from google.cloud import pubsub_v1
        
        response = subscriber.pull(
            request={
                "subscription": subscription_path,
                "max_messages": max_messages,
            },
            timeout=30.0,
        )
        
        messages = []
        for msg in response.received_messages:
            messages.append({
                "ack_id": msg.ack_id,
                "message_id": msg.message.message_id,
                "data": msg.message.data.decode("utf-8"),
                "attributes": dict(msg.message.attributes),
                "publish_time": msg.message.publish_time.isoformat() if msg.message.publish_time else None,
            })
        
        return ToolResult.from_data({
            "subscription": subscription,
            "messages": messages,
            "count": len(messages),
        })
    
    def _acknowledge(
        self,
        subscription: Optional[str] = None,
        ack_ids: Optional[list] = None,
        **kwargs,
    ) -> ToolResult:
        """Acknowledge messages."""
        if not subscription:
            return ToolResult.from_error("subscription is required")
        if not ack_ids:
            return ToolResult.from_error("ack_ids list is required")
        
        subscriber = self._get_subscriber()
        subscription_path = self._subscription_path(subscription)
        
        subscriber.acknowledge(
            request={
                "subscription": subscription_path,
                "ack_ids": ack_ids,
            }
        )
        
        return ToolResult.from_data({
            "subscription": subscription,
            "acknowledged": len(ack_ids),
        })
    
    def _list_topics(self, **kwargs) -> ToolResult:
        """List all topics."""
        publisher = self._get_publisher()
        project_path = f"projects/{self.project_id}"
        
        topics = []
        for topic in publisher.list_topics(request={"project": project_path}):
            topics.append({
                "name": topic.name,
                "short_name": topic.name.split("/")[-1],
            })
        
        return ToolResult.from_data({
            "topics": topics,
            "count": len(topics),
        })
    
    def _list_subscriptions(
        self,
        topic: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """List subscriptions."""
        subscriber = self._get_subscriber()
        project_path = f"projects/{self.project_id}"
        
        subscriptions = []
        for sub in subscriber.list_subscriptions(request={"project": project_path}):
            if topic and topic not in sub.topic:
                continue
            subscriptions.append({
                "name": sub.name,
                "short_name": sub.name.split("/")[-1],
                "topic": sub.topic.split("/")[-1],
                "ack_deadline_seconds": sub.ack_deadline_seconds,
            })
        
        return ToolResult.from_data({
            "subscriptions": subscriptions,
            "count": len(subscriptions),
        })
    
    def _create_topic(
        self,
        topic: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create a new topic."""
        if not topic:
            return ToolResult.from_error("topic name is required")
        
        publisher = self._get_publisher()
        topic_path = self._topic_path(topic)
        
        created_topic = publisher.create_topic(request={"name": topic_path})
        
        return ToolResult.from_data({
            "topic": topic,
            "full_name": created_topic.name,
            "created": True,
        })
    
    def _delete_topic(
        self,
        topic: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete a topic."""
        if not topic:
            return ToolResult.from_error("topic name is required")
        
        publisher = self._get_publisher()
        topic_path = self._topic_path(topic)
        
        publisher.delete_topic(request={"topic": topic_path})
        
        return ToolResult.from_data({
            "topic": topic,
            "deleted": True,
        })
    
    def _create_subscription(
        self,
        subscription: Optional[str] = None,
        topic: Optional[str] = None,
        ack_deadline: int = 10,
        **kwargs,
    ) -> ToolResult:
        """Create a subscription."""
        if not subscription:
            return ToolResult.from_error("subscription name is required")
        if not topic:
            return ToolResult.from_error("topic is required")
        
        subscriber = self._get_subscriber()
        subscription_path = self._subscription_path(subscription)
        topic_path = self._topic_path(topic)
        
        created_sub = subscriber.create_subscription(
            request={
                "name": subscription_path,
                "topic": topic_path,
                "ack_deadline_seconds": ack_deadline,
            }
        )
        
        return ToolResult.from_data({
            "subscription": subscription,
            "topic": topic,
            "full_name": created_sub.name,
            "created": True,
        })
    
    def _delete_subscription(
        self,
        subscription: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Delete a subscription."""
        if not subscription:
            return ToolResult.from_error("subscription name is required")
        
        subscriber = self._get_subscriber()
        subscription_path = self._subscription_path(subscription)
        
        subscriber.delete_subscription(request={"subscription": subscription_path})
        
        return ToolResult.from_data({
            "subscription": subscription,
            "deleted": True,
        })
    
    def _get_topic(
        self,
        topic: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get topic details."""
        if not topic:
            return ToolResult.from_error("topic name is required")
        
        publisher = self._get_publisher()
        topic_path = self._topic_path(topic)
        
        topic_obj = publisher.get_topic(request={"topic": topic_path})
        
        return ToolResult.from_data({
            "name": topic_obj.name,
            "short_name": topic,
        })
    
    def _get_subscription(
        self,
        subscription: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Get subscription details."""
        if not subscription:
            return ToolResult.from_error("subscription name is required")
        
        subscriber = self._get_subscriber()
        subscription_path = self._subscription_path(subscription)
        
        sub = subscriber.get_subscription(request={"subscription": subscription_path})
        
        return ToolResult.from_data({
            "name": sub.name,
            "short_name": subscription,
            "topic": sub.topic.split("/")[-1],
            "ack_deadline_seconds": sub.ack_deadline_seconds,
            "retain_acked_messages": sub.retain_acked_messages,
        })
    
    def get_schema(self) -> dict[str, Any]:
        return {
            **super().get_schema(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["publish", "publish_batch", "pull", "acknowledge",
                                "list_topics", "list_subscriptions", "create_topic",
                                "delete_topic", "create_subscription", "delete_subscription",
                                "get_topic", "get_subscription"],
                    },
                    "topic": {"type": "string"},
                    "subscription": {"type": "string"},
                    "message": {"type": "string"},
                    "messages": {"type": "array"},
                    "attributes": {"type": "object"},
                    "ack_ids": {"type": "array"},
                    "max_messages": {"type": "integer", "default": 10},
                },
                "required": ["action"],
            },
        }
