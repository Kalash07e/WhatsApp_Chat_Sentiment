"""
Main chat analyzer that orchestrates all analysis components.
"""
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime

from .parser import ChatParser
from .sentiment import SentimentAnalyzer
from .threat_detector import ThreatDetector
from ..models import ChatAnalysisResult, MessageType, SentimentLabel, UserMetrics
from ..utils.emoji_analyzer import EmojiAnalyzer
from ..utils.topic_modeling import TopicModeler
from ..utils.interaction_analyzer import InteractionAnalyzer


class ChatAnalyzer:
    """Main analyzer class that coordinates all analysis components."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.parser = ChatParser()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.threat_detector = ThreatDetector(api_key)
        self.emoji_analyzer = EmojiAnalyzer()
        self.topic_modeler = TopicModeler()
        self.interaction_analyzer = InteractionAnalyzer()
    
    def analyze_chat(self, chat_text: str) -> ChatAnalysisResult:
        """
        Perform complete analysis of a WhatsApp chat export.
        
        Args:
            chat_text: Raw text from WhatsApp export
            
        Returns:
            ChatAnalysisResult containing all analysis results
        """
        # Parse the chat
        df = self.parser.parse_chat_export(chat_text)
        
        # Validate parsed data
        is_valid, issues = self.parser.validate_parsed_data(df)
        if not is_valid:
            raise ValueError(f"Chat parsing failed: {', '.join(issues)}")
        
        # Process datetime and filter valid messages
        df = self._preprocess_dataframe(df)
        
        # Perform sentiment analysis
        df['sentiment'] = df['message'].apply(self.sentiment_analyzer.analyze_message_sentiment)
        
        # Classify message types
        df['message_type'] = df['message'].apply(self._classify_message_type)
        
        # Generate analysis results
        return self._generate_analysis_result(df)
    
    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the DataFrame with datetime parsing and filtering."""
        # Parse datetime
        df['datetime_parsed'] = pd.to_datetime(df['datetime'], errors='coerce', dayfirst=True)
        
        # Remove rows with invalid datetime
        df = df.dropna(subset=['datetime_parsed']).reset_index(drop=True)
        
        if df.empty:
            raise ValueError("No valid messages with parseable dates found")
        
        # Sort by datetime
        df = df.sort_values('datetime_parsed').reset_index(drop=True)
        
        return df
    
    def _classify_message_type(self, message: str) -> MessageType:
        """Classify message type based on content."""
        if not message or not message.strip():
            return MessageType.SYSTEM
        
        message_lower = message.lower()
        
        if "<media omitted>" in message_lower or "document omitted" in message_lower:
            return MessageType.MEDIA
        elif "?" in message:
            return MessageType.QUESTION
        elif any(url_indicator in message_lower for url_indicator in ["http", "www.", ".com", ".org"]):
            return MessageType.LINK
        elif len(message.split()) < 4:
            return MessageType.SHORT_REACTION
        else:
            return MessageType.REGULAR
    
    def _generate_analysis_result(self, df: pd.DataFrame) -> ChatAnalysisResult:
        """Generate comprehensive analysis result."""
        user_df = df[df['author'] != 'System']
        
        if user_df.empty:
            raise ValueError("No user messages found for analysis")
        
        # Basic metrics
        total_messages = len(user_df)
        total_users = len(user_df['author'].unique())
        date_range = (user_df['datetime_parsed'].min(), user_df['datetime_parsed'].max())
        overall_sentiment = user_df['sentiment'].mean()
        sentiment_label = self.sentiment_analyzer.get_sentiment_label(overall_sentiment)
        most_active_user = user_df['author'].value_counts().idxmax()
        peak_activity_hour = user_df['datetime_parsed'].dt.hour.value_counts().idxmax()
        
        # User metrics
        user_metrics = self._generate_user_metrics(user_df)
        
        # Message type distribution
        message_types_dist = dict(user_df['message_type'].value_counts())
        
        # Emoji analysis
        emoji_analysis = self.emoji_analyzer.analyze_emojis(user_df)
        
        # Topic modeling (optional)
        topics = None
        if len(user_df) >= 10:  # Only if enough messages
            topics = self.topic_modeler.extract_topics(user_df['message'].tolist())
        
        # Interaction matrix
        interaction_matrix = self.interaction_analyzer.get_interaction_matrix(df)
        
        return ChatAnalysisResult(
            total_messages=total_messages,
            total_users=total_users,
            date_range=date_range,
            overall_sentiment=overall_sentiment,
            sentiment_label=sentiment_label,
            most_active_user=most_active_user,
            peak_activity_hour=peak_activity_hour,
            user_metrics=user_metrics,
            message_types_distribution=message_types_dist,
            emoji_analysis=emoji_analysis,
            topics=topics,
            interaction_matrix=interaction_matrix
        )
    
    def _generate_user_metrics(self, df: pd.DataFrame) -> list[UserMetrics]:
        """Generate metrics for each user."""
        user_metrics = []
        
        for username in df['author'].unique():
            user_df = df[df['author'] == username]
            
            metrics = UserMetrics(
                username=username,
                message_count=len(user_df),
                avg_sentiment=user_df['sentiment'].mean(),
                most_active_hour=user_df['datetime_parsed'].dt.hour.value_counts().idxmax() if len(user_df) > 0 else 0,
                emoji_usage=dict(self.emoji_analyzer.get_user_emoji_usage(user_df)),
                message_types=dict(user_df['message_type'].value_counts())
            )
            user_metrics.append(metrics)
        
        # Sort by message count
        user_metrics.sort(key=lambda x: x.message_count, reverse=True)
        
        return user_metrics
    
    def analyze_threats(self, df: pd.DataFrame):
        """Analyze threats in the chat using AI."""
        user_df = df[df['author'] != 'System']
        return self.threat_detector.analyze_threats(user_df)
