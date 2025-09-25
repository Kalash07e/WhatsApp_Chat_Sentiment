# SUGGESTION 1: Add Real-Time Sentiment Tracking
def display_sentiment_timeline(df):
    """Display real-time sentiment changes with interactive timeline"""
    st.subheader("📈 Live Sentiment Timeline")
    
    user_df = df[df['author'] != 'System'].copy()
    if len(user_df) < 2:
        st.info("Need at least 2 messages for timeline analysis")
        return
        
    # Create hourly/daily sentiment aggregation
    user_df['hour'] = user_df['datetime_parsed'].dt.hour
    user_df['date'] = user_df['datetime_parsed'].dt.date
    
    # Hourly sentiment tracking
    hourly_sentiment = user_df.groupby('hour')['sentiment'].agg(['mean', 'count']).reset_index()
    
    # Create interactive timeline
    fig = go.Figure()
    
    # Add sentiment line
    fig.add_trace(go.Scatter(
        x=hourly_sentiment['hour'],
        y=hourly_sentiment['mean'],
        mode='lines+markers',
        name='Average Sentiment',
        line=dict(color='#58a6ff', width=3),
        marker=dict(size=8),
        hovertemplate='<b>Hour:</b> %{x}:00<br><b>Sentiment:</b> %{y:.3f}<br><b>Messages:</b> %{customdata}<extra></extra>',
        customdata=hourly_sentiment['count']
    ))
    
    # Add sentiment zones
    fig.add_hline(y=0.1, line_dash="dash", line_color="green", annotation_text="Positive Zone")
    fig.add_hline(y=-0.1, line_dash="dash", line_color="red", annotation_text="Negative Zone")
    fig.add_hrect(y0=-0.1, y1=0.1, fillcolor="yellow", opacity=0.1, annotation_text="Neutral Zone")
    
    fig.update_layout(
        title="Sentiment Changes Throughout the Day",
        xaxis_title="Hour of Day",
        yaxis_title="Average Sentiment Score",
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# SUGGESTION 2: Advanced User Behavior Analysis
def analyze_user_response_patterns(df):
    """Analyze how users respond to each other"""
    st.subheader("🔄 User Response Patterns")
    
    user_df = df[df['author'] != 'System'].copy()
    if len(user_df) < 10:
        st.info("Need at least 10 messages for response pattern analysis")
        return
    
    # Calculate response times and patterns
    user_df = user_df.sort_values('datetime_parsed')
    user_df['next_author'] = user_df['author'].shift(-1)
    user_df['time_diff'] = user_df['datetime_parsed'].diff().dt.total_seconds() / 60
    
    # Find rapid responses (within 2 minutes)
    rapid_responses = user_df[user_df['time_diff'] <= 2]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Quick Responses", len(rapid_responses), "Within 2 mins")
    with col2:
        avg_response = user_df['time_diff'].mean()
        st.metric("Avg Response Time", f"{avg_response:.1f} min")
    with col3:
        longest_gap = user_df['time_diff'].max()
        st.metric("Longest Silence", f"{longest_gap/60:.1f} hours")

# SUGGESTION 3: Emotion Detection Beyond Sentiment
def analyze_emotions(df):
    """Detect specific emotions using advanced NLP"""
    st.subheader("😊 Emotion Analysis")
    
    # Define emotion keywords
    emotion_keywords = {
        'Joy': ['happy', 'excited', 'amazing', 'awesome', 'great', '😊', '😄', '🎉'],
        'Anger': ['angry', 'annoyed', 'frustrated', 'mad', '😠', '😡'],
        'Sadness': ['sad', 'disappointed', 'hurt', 'cry', '😢', '😭'],
        'Fear': ['scared', 'worried', 'nervous', 'afraid', '😰', '😨'],
        'Love': ['love', 'adore', 'heart', 'kiss', '❤️', '😍', '🥰'],
        'Surprise': ['wow', 'omg', 'shocked', 'unbelievable', '😲', '😱']
    }
    
    user_df = df[df['author'] != 'System']
    emotion_scores = {emotion: 0 for emotion in emotion_keywords}
    
    for _, row in user_df.iterrows():
        message_lower = row['message'].lower()
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                emotion_scores[emotion] += 1
    
    # Create emotion radar chart
    fig = go.Figure(data=go.Scatterpolar(
        r=list(emotion_scores.values()),
        theta=list(emotion_scores.keys()),
        fill='toself',
        name='Emotions'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(emotion_scores.values())])),
        showlegend=False,
        title="Emotional Profile of the Chat",
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
