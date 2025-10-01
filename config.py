"""
Configuration settings for WhatsApp Chat Analyzer
"""
import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class AppConfig:
    """Application configuration settings."""
    
    # Streamlit settings
    PAGE_TITLE: str = "WhatsApp Chat Analysis"
    PAGE_ICON: str = "🕵️"
    LAYOUT: str = "wide"
    
    # Analysis settings
    MAX_THREAD_GAP_MINUTES: int = 5
    WINDOW_SIZE_MULTIPLIER: int = 10
    MIN_WINDOW_SIZE: int = 10
    
    # Topic modeling settings
    TOPIC_MODEL_COMPONENTS: int = 5
    MIN_TOPIC_MESSAGES: int = 5
    
    # Word cloud settings
    WORDCLOUD_WIDTH: int = 1200
    WORDCLOUD_HEIGHT: int = 500
    
    # AI settings
    AI_MAX_CHARS: int = 20000
    AI_REQUEST_TIMEOUT: int = 120
    
    # Chart settings
    PIE_CHART_TOP_AUTHORS: int = 7
    
    # Theme settings
    DEFAULT_THEME: str = "dark"


@dataclass
class ThemeConfig:
    """Theme configuration for the application."""
    
    DARK_THEME: Dict[str, str] = None
    LIGHT_THEME: Dict[str, str] = None
    
    def __post_init__(self):
        self.DARK_THEME = {
            "--bg-color": "#0d1117",
            "--primary-color": "#58a6ff",
            "--secondary-color": "#161b22",
            "--card-color": "#172030",
            "--text-color": "#c9d1d9",
            "--text-muted": "#8b949e",
            "--border-color": "#30363d",
            "--danger-color": "#f85149",
            "--warning-color": "#d29922",
            "--success-color": "#3fb950",
            "--info-color": "#38bdf8",
        }
        
        self.LIGHT_THEME = {
            "--bg-color": "#f7f9fc",
            "--primary-color": "#0b63e5",
            "--secondary-color": "#ffffff",
            "--card-color": "#ffffff",
            "--text-color": "#0b1220",
            "--text-muted": "#5b6b78",
            "--border-color": "#e6eef8",
            "--danger-color": "#d32f2f",
            "--warning-color": "#f39c12",
            "--success-color": "#1eae53",
            "--info-color": "#2aa7ff",
        }


class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.app = AppConfig()
        self.theme = ThemeConfig()
        self._load_environment_vars()
    
    def _load_environment_vars(self):
        """Load configuration from environment variables."""
        # Google AI API Key
        self.GOOGLE_AI_API_KEY = os.getenv('GOOGLE_AI_API_KEY', '')
        
        # Optional: Override default settings with environment variables
        self.app.AI_MAX_CHARS = int(os.getenv('AI_MAX_CHARS', self.app.AI_MAX_CHARS))
        self.app.AI_REQUEST_TIMEOUT = int(os.getenv('AI_REQUEST_TIMEOUT', self.app.AI_REQUEST_TIMEOUT))


# Global config instance
config = Config()
