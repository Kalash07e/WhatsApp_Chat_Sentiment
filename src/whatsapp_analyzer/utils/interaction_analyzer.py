"""
Interaction analysis utilities.
"""
import pandas as pd
from typing import Dict, List, Tuple
from ..config import config


class InteractionAnalyzer:
    """Analyzes user interactions and conversation patterns."""
    
    def __init__(self):
        self.max_thread_gap_minutes = config.app.MAX_THREAD_GAP_MINUTES
    
    def get_interaction_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create an interaction matrix showing who responds to whom.
        
        Args:
            df: DataFrame with chat messages
            
        Returns:
            Square matrix with sender-receiver interaction counts
        """
        # Get unique authors excluding system messages
        authors = sorted(df[df['author'] != 'System']['author'].unique())
        
        if len(authors) < 2:
            return pd.DataFrame()
        
        # Initialize interaction matrix
        matrix = pd.DataFrame(0, index=authors, columns=authors)
        
        # Count interactions (consecutive messages between different users)
        for i in range(1, len(df)):
            current_author = df.iloc[i]['author']
            previous_author = df.iloc[i-1]['author']
            
            # Skip system messages and self-interactions
            if (current_author != 'System' and 
                previous_author != 'System' and 
                current_author != previous_author and
                current_author in authors and 
                previous_author in authors):
                
                matrix.loc[current_author, previous_author] += 1
        
        return matrix
    
    def detect_conversation_threads(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect conversation threads based on time gaps and author changes.
        
        Args:
            df: DataFrame with chat messages
            
        Returns:
            DataFrame with additional 'thread_id' column
        """
        if df.empty:
            return df.copy()
        
        df_sorted = df.sort_values('datetime_parsed').copy()
        
        # Calculate time differences and author changes
        df_sorted['next_author'] = df_sorted['author'].shift(-1)
        df_sorted['next_time'] = df_sorted['datetime_parsed'].shift(-1)
        df_sorted['time_diff_minutes'] = (
            df_sorted['next_time'] - df_sorted['datetime_parsed']
        ).dt.total_seconds() / 60.0
        
        # Detect thread boundaries
        df_sorted['is_thread_boundary'] = (
            (df_sorted['time_diff_minutes'] > self.max_thread_gap_minutes) |
            df_sorted['message'].str.contains(r'^\>', regex=True, na=False) |
            (df_sorted['author'] != df_sorted['next_author'])
        )
        
        # Assign thread IDs
        df_sorted['thread_id'] = df_sorted['is_thread_boundary'].cumsum()
        
        # Clean up temporary columns
        result = df_sorted[['datetime', 'author', 'message', 'thread_id']].copy()
        
        return result
    
    def get_response_times(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Calculate response times between users.
        
        Args:
            df: DataFrame with chat messages
            
        Returns:
            Dictionary mapping user pairs to list of response times in minutes
        """
        response_times = {}
        
        for i in range(1, len(df)):
            current_row = df.iloc[i]
            previous_row = df.iloc[i-1]
            
            current_author = current_row['author']
            previous_author = previous_row['author']
            
            # Skip if same author or system messages
            if (current_author == previous_author or 
                current_author == 'System' or 
                previous_author == 'System'):
                continue
            
            # Calculate response time
            time_diff = (current_row['datetime_parsed'] - 
                        previous_row['datetime_parsed']).total_seconds() / 60.0
            
            # Only consider reasonable response times (< 24 hours)
            if 0 < time_diff < 1440:
                pair_key = f"{previous_author} -> {current_author}"
                
                if pair_key not in response_times:
                    response_times[pair_key] = []
                
                response_times[pair_key].append(time_diff)
        
        return response_times
    
    def get_conversation_statistics(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Get overall conversation statistics.
        
        Args:
            df: DataFrame with chat messages
            
        Returns:
            Dictionary with conversation statistics
        """
        user_df = df[df['author'] != 'System']
        
        if user_df.empty:
            return {}
        
        # Basic statistics
        total_messages = len(user_df)
        unique_users = len(user_df['author'].unique())
        date_range = (user_df['datetime_parsed'].min(), user_df['datetime_parsed'].max())
        duration_days = (date_range[1] - date_range[0]).days
        
        # Activity patterns
        hourly_activity = user_df['datetime_parsed'].dt.hour.value_counts().to_dict()
        daily_activity = user_df['datetime_parsed'].dt.date.value_counts().to_dict()
        
        # User statistics
        user_message_counts = user_df['author'].value_counts().to_dict()
        
        return {
            'total_messages': total_messages,
            'unique_users': unique_users,
            'duration_days': duration_days,
            'messages_per_day': total_messages / max(duration_days, 1),
            'most_active_hour': max(hourly_activity.items(), key=lambda x: x[1])[0],
            'user_message_counts': user_message_counts,
            'hourly_distribution': hourly_activity,
            'daily_distribution': daily_activity
        }
