"""
Unit tests for the sentiment analysis module.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from whatsapp_analyzer.core.sentiment import SentimentAnalyzer
from whatsapp_analyzer.models import SentimentLabel


class TestSentimentAnalyzer:
    """Test cases for SentimentAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
    
    def test_positive_sentiment(self):
        """Test positive sentiment analysis."""
        message = "I love this! It's absolutely amazing!"
        score = self.analyzer.analyze_message_sentiment(message)
        assert score > 0.1  # Should be clearly positive
    
    def test_negative_sentiment(self):
        """Test negative sentiment analysis."""
        message = "This is terrible! I hate it so much."
        score = self.analyzer.analyze_message_sentiment(message)
        assert score < -0.1  # Should be clearly negative
    
    def test_neutral_sentiment(self):
        """Test neutral sentiment analysis."""
        message = "The weather is okay today."
        score = self.analyzer.analyze_message_sentiment(message)
        assert -0.1 <= score <= 0.1  # Should be neutral
    
    def test_empty_message(self):
        """Test empty message handling."""
        score = self.analyzer.analyze_message_sentiment("")
        assert score == 0.0
        
        score = self.analyzer.analyze_message_sentiment("   ")
        assert score == 0.0
    
    def test_hinglish_detection(self):
        """Test Hinglish content detection."""
        hinglish_message = "Yaar, this is really acha!"
        english_message = "This is really good!"
        
        # Both should work, but may use different analyzers internally
        hinglish_score = self.analyzer.analyze_message_sentiment(hinglish_message)
        english_score = self.analyzer.analyze_message_sentiment(english_message)
        
        # Both should be positive
        assert hinglish_score > 0
        assert english_score > 0
    
    def test_get_sentiment_label(self):
        """Test sentiment label conversion."""
        assert self.analyzer.get_sentiment_label(0.5) == SentimentLabel.POSITIVE
        assert self.analyzer.get_sentiment_label(-0.5) == SentimentLabel.NEGATIVE
        assert self.analyzer.get_sentiment_label(0.02) == SentimentLabel.NEUTRAL
        assert self.analyzer.get_sentiment_label(-0.02) == SentimentLabel.NEUTRAL
    
    def test_analyze_sentiment_distribution(self):
        """Test sentiment distribution analysis."""
        sentiment_scores = [0.5, -0.3, 0.1, -0.8, 0.2, 0.0]
        
        distribution = self.analyzer.analyze_sentiment_distribution(sentiment_scores)
        
        assert 'mean' in distribution
        assert 'positive_count' in distribution
        assert 'negative_count' in distribution
        assert 'neutral_count' in distribution
        
        # Check counts
        assert distribution['positive_count'] == 3  # 0.5, 0.1, 0.2
        assert distribution['negative_count'] == 2  # -0.3, -0.8
        assert distribution['neutral_count'] == 1   # 0.0
    
    def test_empty_sentiment_distribution(self):
        """Test sentiment distribution with empty list."""
        distribution = self.analyzer.analyze_sentiment_distribution([])
        
        assert distribution['mean'] == 0.0
        assert distribution['positive_count'] == 0
        assert distribution['negative_count'] == 0
        assert distribution['neutral_count'] == 0
