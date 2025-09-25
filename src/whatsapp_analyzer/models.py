"""
Data models for WhatsApp chat analysis.
"""
from dataclasses import dataclass
from datetime import datetime as dt
from typing import List, Optional, Dict, Any
from enum import Enum
import pandas as pd


class MessageType(Enum):
    """Enumeration for different message types."""
    REGULAR = "Regular Message"
    MEDIA = "Media"
    QUESTION = "Question"
    LINK = "Link"
    SHORT_REACTION = "Short Reaction"
    SYSTEM = "System"


class SentimentLabel(Enum):
    """Enumeration for sentiment labels."""
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"


class ThreatSeverity(Enum):
    """Enumeration for threat severity levels."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class ChatMessage:
    """Represents a single chat message."""
    datetime: str
    author: str
    message: str
    datetime_parsed: Optional[dt] = None
    sentiment: Optional[float] = None
    message_type: Optional[MessageType] = None
    thread_id: Optional[int] = None


@dataclass
class UserMetrics:
    """Metrics for a specific user."""
    username: str
    message_count: int
    avg_sentiment: float
    most_active_hour: int
    emoji_usage: Dict[str, int]
    message_types: Dict[MessageType, int]


@dataclass
class ChatAnalysisResult:
    """Complete analysis result for a chat."""
    total_messages: int
    total_users: int
    date_range: tuple
    overall_sentiment: float
    sentiment_label: SentimentLabel
    most_active_user: str
    peak_activity_hour: int
    user_metrics: List[UserMetrics]
    message_types_distribution: Dict[MessageType, int]
    emoji_analysis: pd.DataFrame
    topics: Optional[List[str]] = None
    interaction_matrix: Optional[pd.DataFrame] = None


@dataclass
class ThreatAnalysisResult:
    """Result of threat analysis."""
    threats_found: int
    severity_distribution: Dict[ThreatSeverity, int]
    threat_details: List[Dict[str, Any]]
    analysis_summary: str
