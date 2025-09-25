"""
Chat parsing functionality for WhatsApp exports.
"""
import re
import pandas as pd
from typing import List, Tuple
from ..models import ChatMessage
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ChatParser:
    """Handles parsing of WhatsApp chat exports."""
    
    def __init__(self):
        self.pattern_author = re.compile(
            r'^\[?(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?::\d{2})?)(?:\s*(AM|PM|am|pm))?\]?\s*(?:-)?\s*([^:]+?):\s*(.*)$'
        )
        self.pattern_system = re.compile(
            r'^\[?(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?::\d{2})?)(?:\s*(AM|PM|am|pm))?\]?\s*(?:-)?\s*(.*)$'
        )
    
    def preprocess_text(self, chat_text: str) -> str:
        """Preprocess the raw chat text."""
        # Remove Unicode control characters
        chat_text = chat_text.replace('\u200e', '').replace('\ufeff', '').replace('\u202f', ' ')
        return chat_text
    
    def parse_chat_export(self, chat_text: str) -> pd.DataFrame:
        """
        Parse WhatsApp chat export text into structured DataFrame.
        
        Args:
            chat_text: Raw text from WhatsApp export
            
        Returns:
            DataFrame with columns: datetime, author, message
        """
        logger.info("Starting chat parsing...")
        
        chat_text = self.preprocess_text(chat_text)
        dates, authors, messages = [], [], []
        last_idx = -1
        
        for line_num, line in enumerate(chat_text.splitlines()):
            raw = line.rstrip("\n").strip()
            
            if not raw:
                if last_idx >= 0:
                    messages[last_idx] += "\n"
                continue
            
            # Try to match author message pattern
            m_author = self.pattern_author.match(raw)
            if m_author:
                dt = self._format_datetime(m_author.group(1), m_author.group(2), m_author.group(3))
                dates.append(dt)
                authors.append(m_author.group(4).strip())
                messages.append(m_author.group(5).strip())
                last_idx += 1
                continue
            
            # Try to match system message pattern
            m_system = self.pattern_system.match(raw)
            if m_system:
                dt = self._format_datetime(m_system.group(1), m_system.group(2), m_system.group(3))
                if ':' not in m_system.group(4):
                    dates.append(dt)
                    authors.append("System")
                    messages.append(m_system.group(4).strip())
                    last_idx += 1
                    continue
            
            # Continuation of previous message
            if last_idx >= 0:
                messages[last_idx] += " " + raw
            else:
                # Orphan line - treat as system message
                dates.append("")
                authors.append("System")
                messages.append(raw)
                last_idx += 1
        
        df = pd.DataFrame({
            'datetime': dates, 
            'author': authors, 
            'message': messages
        })
        
        logger.info(f"Parsed {len(df)} messages from {len(df['author'].unique())} unique authors")
        return df
    
    def _format_datetime(self, date_part: str, time_part: str, ampm: str = None) -> str:
        """Format datetime components into consistent string."""
        formatted_time = time_part
        if ampm:
            formatted_time += f" {ampm}"
        return f"{date_part}, {formatted_time}"
    
    def validate_parsed_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate the parsed data for common issues.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if df.empty:
            issues.append("DataFrame is empty")
        
        if 'datetime' not in df.columns:
            issues.append("Missing 'datetime' column")
        
        if 'author' not in df.columns:
            issues.append("Missing 'author' column")
        
        if 'message' not in df.columns:
            issues.append("Missing 'message' column")
        
        if len(df) > 0 and df['datetime'].str.strip().eq('').all():
            issues.append("All datetime entries are empty")
        
        # Check for reasonable number of unique authors (between 2 and 100)
        unique_authors = len(df[df['author'] != 'System']['author'].unique())
        if unique_authors < 2:
            issues.append(f"Too few unique authors: {unique_authors}")
        elif unique_authors > 100:
            issues.append(f"Unusually high number of authors: {unique_authors}")
        
        return len(issues) == 0, issues
