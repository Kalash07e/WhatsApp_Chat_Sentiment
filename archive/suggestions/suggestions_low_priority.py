# LOW PRIORITY BUT VALUABLE IMPROVEMENTS

#### 6. **Security and Privacy Enhancements**

# SUGGESTION 10: Data Privacy Protection
def anonymize_chat_data(df):
    """Anonymize user data while preserving analysis quality"""
    # Create user mapping
    unique_authors = df[df['author'] != 'System']['author'].unique()
    author_mapping = {author: f"User_{i+1}" for i, author in enumerate(unique_authors)}
    
    # Replace names in messages
    df_anon = df.copy()
    for original, anonymous in author_mapping.items():
        df_anon['author'] = df_anon['author'].replace(original, anonymous)
        # Also replace names mentioned in messages
        df_anon['message'] = df_anon['message'].str.replace(original, anonymous, case=False)
    
    return df_anon, author_mapping

# SUGGESTION 11: Secure File Handling
def secure_file_upload():
    """Enhanced security for file uploads"""
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB limit
    ALLOWED_EXTENSIONS = ['.txt']
    
    uploaded_file = st.file_uploader(
        "Upload WhatsApp Chat",
        type=['txt'],
        help="Only .txt files up to 50MB are allowed"
    )
    
    if uploaded_file:
        # Check file size
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error("File too large! Please upload files smaller than 50MB.")
            return None
            
        # Validate file extension
        if not any(uploaded_file.name.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
            st.error("Invalid file type! Only .txt files are allowed.")
            return None
            
        # Check for potentially malicious content
        content_preview = str(uploaded_file.read(1000))  # Read first 1KB
        uploaded_file.seek(0)  # Reset file pointer
        
        if any(keyword in content_preview.lower() for keyword in ['<script>', 'javascript:', 'eval(']):
            st.error("File contains potentially unsafe content.")
            return None
            
        return uploaded_file
    return None

#### 7. **Advanced Analytics Features**

# SUGGESTION 12: Conversation Quality Metrics
def analyze_conversation_quality(df):
    """Assess the quality and health of conversations"""
    user_df = df[df['author'] != 'System']
    
    metrics = {}
    
    # Engagement Score
    unique_users = len(user_df['author'].unique())
    total_messages = len(user_df)
    avg_messages_per_user = total_messages / unique_users if unique_users > 0 else 0
    
    # Response Rate
    user_df_sorted = user_df.sort_values('datetime_parsed')
    user_df_sorted['responded_to_previous'] = (
        user_df_sorted['author'] != user_df_sorted['author'].shift(1)
    ).astype(int)
    response_rate = user_df_sorted['responded_to_previous'].mean()
    
    # Conversation Depth (average message length)
    avg_message_length = user_df['message'].str.len().mean()
    
    # Balanced Participation
    user_message_counts = user_df['author'].value_counts()
    participation_std = user_message_counts.std()
    participation_balance = 1 - (participation_std / user_message_counts.mean()) if user_message_counts.mean() > 0 else 0
    
    metrics = {
        'engagement_score': min(avg_messages_per_user / 10, 1.0),  # Normalize to 0-1
        'response_rate': response_rate,
        'conversation_depth': min(avg_message_length / 50, 1.0),  # Normalize to 0-1
        'participation_balance': max(0, participation_balance)
    }
    
    # Overall Quality Score
    overall_quality = sum(metrics.values()) / len(metrics)
    
    return metrics, overall_quality

# SUGGESTION 13: Advanced Text Analytics
def advanced_text_analysis(df):
    """Perform sophisticated text analysis"""
    from collections import Counter
    import re
    
    user_df = df[df['author'] != 'System']
    all_text = ' '.join(user_df['message'].astype(str))
    
    # Language complexity analysis
    sentences = all_text.split('.')
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    
    # Vocabulary richness
    words = re.findall(r'\b\w+\b', all_text.lower())
    unique_words = len(set(words))
    total_words = len(words)
    vocabulary_richness = unique_words / total_words if total_words > 0 else 0
    
    # Most common phrases (bigrams and trigrams)
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
    
    common_bigrams = Counter(bigrams).most_common(10)
    common_trigrams = Counter(trigrams).most_common(5)
    
    return {
        'avg_sentence_length': avg_sentence_length,
        'vocabulary_richness': vocabulary_richness,
        'common_bigrams': common_bigrams,
        'common_trigrams': common_trigrams,
        'total_words': total_words,
        'unique_words': unique_words
    }

# SUGGESTION 14: Predictive Analytics
def predict_conversation_trends(df):
    """Predict conversation patterns and sentiment trends"""
    user_df = df[df['author'] != 'System'].copy()
    
    if len(user_df) < 50:  # Need sufficient data
        return None
        
    # Create time-based features
    user_df['hour'] = user_df['datetime_parsed'].dt.hour
    user_df['day_of_week'] = user_df['datetime_parsed'].dt.dayofweek
    user_df['is_weekend'] = user_df['day_of_week'].isin([5, 6]).astype(int)
    
    # Simple trend analysis
    hourly_sentiment = user_df.groupby('hour')['sentiment'].mean()
    
    # Find peak activity hours
    hourly_counts = user_df.groupby('hour').size()
    peak_hours = hourly_counts.nlargest(3).index.tolist()
    
    # Weekend vs weekday patterns
    weekend_sentiment = user_df[user_df['is_weekend'] == 1]['sentiment'].mean()
    weekday_sentiment = user_df[user_df['is_weekend'] == 0]['sentiment'].mean()
    
    return {
        'peak_activity_hours': peak_hours,
        'weekend_sentiment': weekend_sentiment,
        'weekday_sentiment': weekday_sentiment,
        'hourly_sentiment_pattern': hourly_sentiment.to_dict(),
        'sentiment_trend': 'improving' if user_df['sentiment'].tail(100).mean() > user_df['sentiment'].head(100).mean() else 'declining'
    }

#### 8. **Mobile-Friendly Enhancements**

# SUGGESTION 15: Responsive Design
def apply_mobile_friendly_css():
    """Apply CSS for better mobile experience"""
    st.markdown("""
    <style>
    @media (max-width: 768px) {
        .main-container {
            padding: 1rem 0.5rem;
        }
        
        .metric-card {
            margin-bottom: 1rem;
            font-size: 0.9rem;
        }
        
        .chart-container {
            height: 300px !important;
        }
        
        .stPlotlyChart {
            height: 300px !important;
        }
        
        .sidebar .sidebar-content {
            width: 100% !important;
        }
    }
    
    @media (max-width: 480px) {
        .metric-card {
            padding: 0.8rem;
            font-size: 0.8rem;
        }
        
        .main-title {
            font-size: 1.5rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

#### 9. **API Integration Possibilities**

# SUGGESTION 16: External API Integration
def integrate_external_apis():
    """Framework for integrating external APIs"""
    
    # Language Detection API
    def detect_language(text_sample):
        """Detect the primary language of the chat"""
        # This would integrate with services like Google Translate API
        # or use libraries like langdetect
        pass
    
    # Advanced Sentiment API
    def advanced_sentiment_analysis(messages):
        """Use advanced sentiment APIs like AWS Comprehend"""
        # Integration with cloud sentiment analysis services
        pass
    
    # Spam/Toxicity Detection
    def detect_harmful_content(messages):
        """Detect spam, toxic, or harmful content"""
        # Integration with content moderation APIs
        pass
    
    return {
        'language_detection': detect_language,
        'advanced_sentiment': advanced_sentiment_analysis,
        'content_moderation': detect_harmful_content
    }

#### 10. **Testing and Quality Assurance**

# SUGGESTION 17: Comprehensive Testing Framework
def create_test_suite():
    """Create comprehensive test cases"""
    
    # Sample test data
    test_data = {
        'basic_chat': [
            "12/25/23, 10:30 AM - John: Hello everyone!",
            "12/25/23, 10:31 AM - Jane: Hi John! How are you?",
            "12/25/23, 10:32 AM - John: I'm great! 😊"
        ],
        'edge_cases': [
            "Invalid date format",
            "12/25/23, 10:30 AM - : Empty author",
            "12/25/23, 10:30 AM - John: "  # Empty message
        ]
    }
    
    def test_preprocessing():
        """Test chat preprocessing function"""
        # Test normal cases
        df = preprocess_chat(test_data['basic_chat'])
        assert df is not None
        assert len(df) == 3
        assert 'sentiment' in df.columns
        
        # Test edge cases
        df_edge = preprocess_chat(test_data['edge_cases'])
        # Should handle gracefully without crashing
        
    def test_sentiment_analysis():
        """Test sentiment analysis accuracy"""
        positive_message = "I love this! It's amazing! 😊"
        negative_message = "This is terrible. I hate it. 😠"
        
        # Test sentiment scoring
        # Add assertions for expected sentiment ranges
        
    return {
        'test_preprocessing': test_preprocessing,
        'test_sentiment': test_sentiment_analysis
    }
