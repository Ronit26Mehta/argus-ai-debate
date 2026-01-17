"""
ARGUS Human-in-the-Loop Callbacks.

Real-time feedback collection from human reviewers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Callable

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of feedback."""
    APPROVAL = "approval"
    RATING = "rating"
    ANNOTATION = "annotation"
    CORRECTION = "correction"


@dataclass
class Feedback:
    """Feedback from a human reviewer."""
    feedback_id: str
    feedback_type: FeedbackType
    content: Any
    agent_name: Optional[str] = None
    action_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "feedback_type": self.feedback_type.value,
            "content": self.content,
            "agent_name": self.agent_name,
            "action_name": self.action_name,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class BaseCallback(ABC):
    """Abstract base class for HITL callbacks."""
    
    def __init__(self):
        self._feedback: list[Feedback] = []
    
    @abstractmethod
    def collect(self, **kwargs: Any) -> Feedback:
        pass
    
    def get_feedback(self, limit: int = 100) -> list[Feedback]:
        return self._feedback[-limit:]


class FeedbackCallback(BaseCallback):
    """General feedback collection callback."""
    
    def __init__(self, on_feedback: Optional[Callable[[Feedback], None]] = None):
        super().__init__()
        self.on_feedback = on_feedback
    
    def collect(
        self,
        feedback_type: FeedbackType,
        content: Any,
        agent_name: Optional[str] = None,
        action_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Feedback:
        import uuid
        feedback = Feedback(
            feedback_id=str(uuid.uuid4()),
            feedback_type=feedback_type,
            content=content,
            agent_name=agent_name,
            action_name=action_name,
            metadata=kwargs,
        )
        self._feedback.append(feedback)
        if self.on_feedback:
            self.on_feedback(feedback)
        logger.debug(f"Collected {feedback_type.value} feedback: {feedback.feedback_id}")
        return feedback


class RatingCallback(BaseCallback):
    """Collect numeric ratings (1-5)."""
    
    def __init__(self, min_rating: int = 1, max_rating: int = 5):
        super().__init__()
        self.min_rating = min_rating
        self.max_rating = max_rating
    
    def collect(
        self,
        rating: int,
        agent_name: Optional[str] = None,
        action_name: Optional[str] = None,
        comment: Optional[str] = None,
        **kwargs: Any,
    ) -> Feedback:
        import uuid
        if not self.min_rating <= rating <= self.max_rating:
            raise ValueError(f"Rating must be between {self.min_rating} and {self.max_rating}")
        feedback = Feedback(
            feedback_id=str(uuid.uuid4()),
            feedback_type=FeedbackType.RATING,
            content={"rating": rating, "comment": comment},
            agent_name=agent_name,
            action_name=action_name,
            metadata=kwargs,
        )
        self._feedback.append(feedback)
        logger.debug(f"Collected rating {rating} for {action_name or 'unknown'}")
        return feedback
    
    def get_average_rating(self, action_name: Optional[str] = None) -> float:
        ratings = [f.content["rating"] for f in self._feedback 
                   if f.feedback_type == FeedbackType.RATING 
                   and (action_name is None or f.action_name == action_name)]
        return sum(ratings) / len(ratings) if ratings else 0.0


class AnnotationCallback(BaseCallback):
    """Collect text annotations on agent outputs."""
    
    def collect(
        self,
        annotation: str,
        target_id: str,
        agent_name: Optional[str] = None,
        action_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Feedback:
        import uuid
        feedback = Feedback(
            feedback_id=str(uuid.uuid4()),
            feedback_type=FeedbackType.ANNOTATION,
            content={"annotation": annotation, "target_id": target_id},
            agent_name=agent_name,
            action_name=action_name,
            metadata=kwargs,
        )
        self._feedback.append(feedback)
        logger.debug(f"Collected annotation for target {target_id}")
        return feedback


class CorrectionCallback(BaseCallback):
    """Record corrections to agent outputs for learning."""
    
    def collect(
        self,
        original: str,
        corrected: str,
        agent_name: Optional[str] = None,
        action_name: Optional[str] = None,
        reason: Optional[str] = None,
        **kwargs: Any,
    ) -> Feedback:
        import uuid
        feedback = Feedback(
            feedback_id=str(uuid.uuid4()),
            feedback_type=FeedbackType.CORRECTION,
            content={"original": original, "corrected": corrected, "reason": reason},
            agent_name=agent_name,
            action_name=action_name,
            metadata=kwargs,
        )
        self._feedback.append(feedback)
        logger.debug(f"Recorded correction for {action_name or 'unknown'}")
        return feedback
    
    def get_corrections_for_agent(self, agent_name: str) -> list[Feedback]:
        return [f for f in self._feedback if f.agent_name == agent_name]


class CallbackManager:
    """Manages multiple callback types."""
    
    def __init__(self):
        self.feedback_callback = FeedbackCallback()
        self.rating_callback = RatingCallback()
        self.annotation_callback = AnnotationCallback()
        self.correction_callback = CorrectionCallback()
    
    def get_all_feedback(self, limit: int = 100) -> list[Feedback]:
        all_feedback = (
            self.feedback_callback.get_feedback() +
            self.rating_callback.get_feedback() +
            self.annotation_callback.get_feedback() +
            self.correction_callback.get_feedback()
        )
        all_feedback.sort(key=lambda f: f.timestamp, reverse=True)
        return all_feedback[:limit]
