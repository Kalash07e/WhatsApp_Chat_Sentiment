"""
Unit tests for the chat parser module.
"""
import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from whatsapp_analyzer.core.parser import ChatParser


class TestChatParser:
    """Test cases for ChatParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = ChatParser()
        
    def test_preprocess_text(self):
        """Test text preprocessing."""
        test_text = "\u200eHello\ufeff World\u202f Test"
        expected = "Hello World Test"
        result = self.parser.preprocess_text(test_text)
        assert result == expected
    
    def test_parse_simple_chat(self):
        """Test parsing of simple chat format."""
        chat_text = """1/1/23, 10:00 AM - John: Hello there
1/1/23, 10:05 AM - Jane: Hi John!
1/1/23, 10:10 AM - John: How are you?"""
        
        df = self.parser.parse_chat_export(chat_text)
        
        assert len(df) == 3
        assert df.iloc[0]['author'] == 'John'
        assert df.iloc[1]['author'] == 'Jane'
        assert df.iloc[0]['message'] == 'Hello there'
        assert df.iloc[1]['message'] == 'Hi John!'
    
    def test_parse_multiline_message(self):
        """Test parsing of multiline messages."""
        chat_text = """1/1/23, 10:00 AM - John: This is a long message
that continues on the next line
and even more lines"""
        
        df = self.parser.parse_chat_export(chat_text)
        
        assert len(df) == 1
        expected_message = "This is a long message that continues on the next line and even more lines"
        assert df.iloc[0]['message'] == expected_message
    
    def test_parse_system_message(self):
        """Test parsing of system messages."""
        chat_text = """1/1/23, 10:00 AM - Messages to this chat and calls are now secured with end-to-end encryption
1/1/23, 10:05 AM - John: Hello"""
        
        df = self.parser.parse_chat_export(chat_text)
        
        assert len(df) == 2
        assert df.iloc[0]['author'] == 'System'
        assert df.iloc[1]['author'] == 'John'
    
    def test_validate_parsed_data_valid(self):
        """Test validation of valid parsed data."""
        df = pd.DataFrame({
            'datetime': ['1/1/23, 10:00 AM', '1/1/23, 10:05 AM'],
            'author': ['John', 'Jane'],
            'message': ['Hello', 'Hi']
        })
        
        is_valid, issues = self.parser.validate_parsed_data(df)
        assert is_valid
        assert len(issues) == 0
    
    def test_validate_parsed_data_invalid(self):
        """Test validation of invalid parsed data."""
        # Empty DataFrame
        df_empty = pd.DataFrame()
        is_valid, issues = self.parser.validate_parsed_data(df_empty)
        assert not is_valid
        assert "DataFrame is empty" in issues
        
        # Missing columns
        df_missing = pd.DataFrame({'datetime': ['1/1/23'], 'author': ['John']})
        is_valid, issues = self.parser.validate_parsed_data(df_missing)
        assert not is_valid
        assert "Missing 'message' column" in issues
    
    def test_format_datetime(self):
        """Test datetime formatting."""
        result = self.parser._format_datetime("1/1/23", "10:00", "AM")
        expected = "1/1/23, 10:00 AM"
        assert result == expected
        
        result = self.parser._format_datetime("1/1/23", "22:00")
        expected = "1/1/23, 22:00"
        assert result == expected
