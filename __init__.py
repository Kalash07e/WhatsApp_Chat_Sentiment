"""
WhatsApp Chat Sentiment Analysis Package

A comprehensive tool for analyzing WhatsApp chat exports with sentiment analysis,
threat detection, and advanced analytics capabilities.
"""

__version__ = "1.0.0"
__author__ = "Kalash Bhargava"

from .core.analyzer import ChatAnalyzer
from .core.sentiment import SentimentAnalyzer
from .core.threat_detector import ThreatDetector

__all__ = ['ChatAnalyzer', 'SentimentAnalyzer', 'ThreatDetector']
