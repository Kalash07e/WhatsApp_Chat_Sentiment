"""
Emoji analysis utilities.
"""
import re
import pandas as pd
from collections import Counter
from typing import Dict, List


class EmojiAnalyzer:
    """Handles emoji extraction and analysis from chat messages."""
    
    def __init__(self):
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\u2600-\u26FF\u2700-\u27BF"
            "]+", flags=re.UNICODE
        )
    
    def extract_emojis(self, text: str) -> List[str]:
        """Extract all emojis from a text."""
        if not text:
            return []
        return self.emoji_pattern.findall(text)
    
    def analyze_emojis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze emoji usage across all messages.
        
        Args:
            df: DataFrame with columns ['message', 'author', 'sentiment']
            
        Returns:
            DataFrame with emoji analysis results
        """
        records = []
        
        for _, row in df.iterrows():
            emojis = self.extract_emojis(row['message'])
            for emoji in emojis:
                records.append({
                    'emoji': emoji,
                    'author': row['author'],
                    'sentiment': row.get('sentiment', 0.0)
                })
        
        if not records:
            return pd.DataFrame(columns=['emoji', 'count', 'avg_sentiment'])
        
        # Create DataFrame and analyze
        emoji_df = pd.DataFrame(records)
        
        # Group by emoji and calculate statistics
        summary = emoji_df.groupby('emoji').agg({
            'emoji': 'size',
            'sentiment': 'mean'
        }).rename(columns={
            'emoji': 'count',
            'sentiment': 'avg_sentiment'
        }).reset_index()
        
        return summary.sort_values('count', ascending=False)
    
    def get_user_emoji_usage(self, user_df: pd.DataFrame) -> Dict[str, int]:
        """Get emoji usage statistics for a specific user."""
        all_emojis = []
        
        for _, row in user_df.iterrows():
            emojis = self.extract_emojis(row['message'])
            all_emojis.extend(emojis)
        
        return dict(Counter(all_emojis))
    
    def get_top_emojis(self, df: pd.DataFrame, top_n: int = 10) -> List[tuple]:
        """Get top N most used emojis."""
        emoji_analysis = self.analyze_emojis(df)
        
        if emoji_analysis.empty:
            return []
        
        return emoji_analysis.head(top_n)[['emoji', 'count']].values.tolist()
    
    def get_emoji_sentiment_correlation(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get correlation between emojis and sentiment."""
        emoji_analysis = self.analyze_emojis(df)
        
        if emoji_analysis.empty:
            return {}
        
        return dict(zip(emoji_analysis['emoji'], emoji_analysis['avg_sentiment']))
