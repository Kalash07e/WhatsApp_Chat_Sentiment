"""
Sentiment analysis functionality.
"""
import re
from typing import Set
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ..models import SentimentLabel


class SentimentAnalyzer:
    """Handles sentiment analysis for chat messages."""
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.hinglish_words = self._load_hinglish_words()
    
    def _load_hinglish_words(self) -> Set[str]:
        """Load Hinglish words for language detection."""
        return {
            "gussa", "pyar", "pyaar", "dhoka", "acha", "accha", "bura", 
            "yaar", "mujhe", "tum", "tumhe", "kya", "kyun", "nahi", 
            "nahin", "haan", "theek", "sahi", "galat", "paagal", 
            "dost", "bhai", "behen", "mummy", "papa", "beta"
        }
    
    def analyze_message_sentiment(self, message: str) -> float:
        """
        Analyze sentiment of a single message.
        
        Args:
            message: The message text to analyze
            
        Returns:
            Sentiment score between -1 (negative) and 1 (positive)
        """
        if not message.strip():
            return 0.0
        
        # Check for Hinglish content
        words = set(re.findall(r'\b\w+\b', message.lower()))
        has_hinglish = any(word in self.hinglish_words for word in words)
        
        if has_hinglish:
            # Use VADER for Hinglish content as it handles mixed language better
            return self.vader_analyzer.polarity_scores(message)['compound']
        else:
            # Use TextBlob for pure English content
            return TextBlob(message).sentiment.polarity
    
    def get_sentiment_label(self, sentiment_score: float) -> SentimentLabel:
        """
        Convert sentiment score to categorical label.
        
        Args:
            sentiment_score: Numerical sentiment score
            
        Returns:
            SentimentLabel enum value
        """
        if sentiment_score > 0.05:
            return SentimentLabel.POSITIVE
        elif sentiment_score < -0.05:
            return SentimentLabel.NEGATIVE
        else:
            return SentimentLabel.NEUTRAL
    
    def analyze_sentiment_distribution(self, sentiment_scores: list) -> dict:
        """
        Analyze the distribution of sentiment scores.
        
        Args:
            sentiment_scores: List of sentiment scores
            
        Returns:
            Dictionary with sentiment distribution statistics
        """
        if not sentiment_scores:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
        
        import numpy as np
        
        positive_count = sum(1 for score in sentiment_scores if score > 0.05)
        negative_count = sum(1 for score in sentiment_scores if score < -0.05)
        neutral_count = len(sentiment_scores) - positive_count - negative_count
        
        return {
            'mean': np.mean(sentiment_scores),
            'median': np.median(sentiment_scores),
            'std': np.std(sentiment_scores),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'positive_percentage': (positive_count / len(sentiment_scores)) * 100,
            'negative_percentage': (negative_count / len(sentiment_scores)) * 100,
            'neutral_percentage': (neutral_count / len(sentiment_scores)) * 100
        }
