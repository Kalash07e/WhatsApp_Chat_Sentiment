"""
Chart generation for visualization.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from typing import Optional
from ..config import config


class ChartGenerator:
    """Generates various charts for data visualization."""
    
    def __init__(self):
        self.config = config
    
    def create_user_pie_chart(self, df: pd.DataFrame) -> Optional[go.Figure]:
        """Create pie chart showing most active users."""
        author_counts = df[df['author'] != 'System']['author'].value_counts()
        
        if author_counts.empty:
            return None
        
        # Limit to top authors
        top_authors = author_counts.head(self.config.app.PIE_CHART_TOP_AUTHORS)
        
        if len(author_counts) > self.config.app.PIE_CHART_TOP_AUTHORS:
            top_authors['Others'] = author_counts[self.config.app.PIE_CHART_TOP_AUTHORS:].sum()
        
        fig = px.pie(
            values=top_authors.values,
            names=top_authors.index,
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Blues_r,
            title="Most Active Users"
        )
        
        fig.update_layout(
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_x=0.5
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont_size=14,
            hovertemplate='<b>%{label}</b><br>%{value} messages<br>%{percent}'
        )
        
        return fig
    
    def create_message_type_pie_chart(self, message_types: dict) -> Optional[go.Figure]:
        """Create pie chart showing message type distribution."""
        if not message_types:
            return None
        
        fig = px.pie(
            values=list(message_types.values()),
            names=list(message_types.keys()),
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Greens_r,
            title="Message Types"
        )
        
        fig.update_layout(
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_x=0.5
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont_size=14,
            hovertemplate='<b>%{label}</b><br>%{value} messages<br>%{percent}'
        )
        
        return fig
    
    def create_sentiment_timeline(self, df: pd.DataFrame) -> Optional[go.Figure]:
        """Create sentiment timeline chart."""
        user_df = df[df['author'] != 'System'].copy()
        
        if user_df.empty or len(user_df) <= 1:
            return None
        
        user_df = user_df.sort_values('datetime_parsed')
        
        # Calculate rolling average
        window_size = max(
            self.config.app.MIN_WINDOW_SIZE,
            len(user_df) // self.config.app.WINDOW_SIZE_MULTIPLIER
        )
        
        user_df['sentiment_rolling_avg'] = (
            user_df['sentiment']
            .rolling(window=window_size, min_periods=1)
            .mean()
        )
        
        fig = px.line(
            user_df,
            x='datetime_parsed',
            y='sentiment_rolling_avg',
            title='Conversation Sentiment Over Time (Rolling Average)',
            labels={
                'sentiment_rolling_avg': 'Sentiment Score',
                'datetime_parsed': 'Date'
            }
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="grey")
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        
        return fig
    
    def create_interaction_heatmap(self, interaction_matrix: pd.DataFrame) -> Optional[plt.Figure]:
        """Create interaction heatmap using matplotlib/seaborn."""
        if interaction_matrix.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            interaction_matrix,
            annot=True,
            fmt='d',
            cmap="Blues",
            ax=ax,
            cbar=False,
            annot_kws={"color": "white"}
        )
        
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        
        plt.xticks(rotation=45, ha='right', c='white', fontsize=10)
        plt.yticks(rotation=0, c='white', fontsize=10)
        plt.title('User Interaction Matrix', color='white')
        
        return fig
    
    def create_word_cloud(self, text: str) -> Optional[plt.Figure]:
        """Create word cloud visualization."""
        if not text.strip():
            return None
        
        wordcloud = WordCloud(
            width=self.config.app.WORDCLOUD_WIDTH,
            height=self.config.app.WORDCLOUD_HEIGHT,
            background_color=None,
            mode="RGBA",
            stopwords=STOPWORDS,
            collocations=False
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        
        return fig
    
    def create_hourly_activity_chart(self, df: pd.DataFrame) -> Optional[go.Figure]:
        """Create hourly activity distribution chart."""
        user_df = df[df['author'] != 'System']
        
        if user_df.empty:
            return None
        
        hourly_counts = user_df['datetime_parsed'].dt.hour.value_counts().sort_index()
        
        fig = go.Figure(data=go.Bar(
            x=hourly_counts.index,
            y=hourly_counts.values,
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title='Message Activity by Hour',
            xaxis_title='Hour of Day',
            yaxis_title='Number of Messages',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        
        return fig
    
    def create_user_sentiment_comparison(self, user_metrics: list) -> Optional[go.Figure]:
        """Create user sentiment comparison chart."""
        if not user_metrics or len(user_metrics) < 2:
            return None
        
        usernames = [metric.username for metric in user_metrics[:10]]  # Top 10 users
        sentiments = [metric.avg_sentiment for metric in user_metrics[:10]]
        message_counts = [metric.message_count for metric in user_metrics[:10]]
        
        fig = go.Figure(data=go.Scatter(
            x=sentiments,
            y=usernames,
            mode='markers',
            marker=dict(
                size=[min(count/10, 50) for count in message_counts],
                color=sentiments,
                colorscale='RdYlBu',
                showscale=True,
                colorbar=dict(title="Sentiment Score")
            ),
            text=[f"{name}<br>Messages: {count}" for name, count in zip(usernames, message_counts)],
            hovertemplate='%{text}<br>Sentiment: %{x:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='User Sentiment vs Activity',
            xaxis_title='Average Sentiment Score',
            yaxis_title='Users',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        
        return fig
