# API Documentation

## Core Modules

### ChatAnalyzer

The main analyzer class that orchestrates all analysis components.

```python
from whatsapp_analyzer import ChatAnalyzer

analyzer = ChatAnalyzer(api_key="your_google_ai_key")
result = analyzer.analyze_chat(chat_text)
```

#### Methods

- `analyze_chat(chat_text: str) -> ChatAnalysisResult`: Perform complete chat analysis
- `analyze_threats(df: pd.DataFrame) -> ThreatAnalysisResult`: Analyze threats using AI

### SentimentAnalyzer

Handles sentiment analysis for chat messages using hybrid VADER + TextBlob approach.

```python
from whatsapp_analyzer.core.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer()
score = analyzer.analyze_message_sentiment("I love this!")
label = analyzer.get_sentiment_label(score)
```

#### Methods

- `analyze_message_sentiment(message: str) -> float`: Analyze single message sentiment
- `get_sentiment_label(score: float) -> SentimentLabel`: Convert score to label
- `analyze_sentiment_distribution(scores: list) -> dict`: Analyze score distribution

### ThreatDetector

AI-powered threat detection using Google Gemini.

```python
from whatsapp_analyzer.core.threat_detector import ThreatDetector

detector = ThreatDetector(api_key="your_key")
threats = detector.analyze_threats(messages_df)
```

#### Methods

- `analyze_threats(messages_df: pd.DataFrame) -> ThreatAnalysisResult`: Analyze chat for threats

## Data Models

### ChatAnalysisResult

Complete analysis result containing:
- `total_messages: int`
- `total_users: int`
- `overall_sentiment: float`
- `sentiment_label: SentimentLabel`
- `user_metrics: List[UserMetrics]`
- `message_types_distribution: Dict`
- `topics: Optional[List[str]]`

### ThreatAnalysisResult

Threat analysis results containing:
- `threats_found: int`
- `severity_distribution: Dict[ThreatSeverity, int]`
- `threat_details: List[Dict]`
- `analysis_summary: str`

## Configuration

Configuration is managed through `whatsapp_analyzer.config`:

```python
from whatsapp_analyzer.config import config

# Access configuration
max_chars = config.app.AI_MAX_CHARS
theme_vars = config.theme.DARK_THEME
```

## Error Handling

All modules include comprehensive error handling:

```python
try:
    result = analyzer.analyze_chat(chat_text)
except ValueError as e:
    print(f"Chat parsing failed: {e}")
except RuntimeError as e:
    print(f"Analysis failed: {e}")
```
