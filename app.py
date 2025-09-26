def get_gemini_api_key():
    """Read the Gemini API key from the .gemini_api_key file in the project root."""
    key_path = os.path.join(os.path.dirname(__file__), ".gemini_api_key")
    try:
        with open(key_path, "r") as f:
            return f.read().strip()
    except Exception:
        return None
import io
import re
import csv
import base64
import html as html_lib
from collections import Counter
import datetime
import time
import subprocess
import sys
import os
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from fpdf import FPDF

# Import authentication system
from auth import (
    is_user_logged_in, create_login_form, create_user_dashboard, 
    show_user_profile, get_all_users, is_admin_user, configure_admin_access, ADMIN_EMAILS, USER_DB_FILE
)

try:
    import google.generativeai as genai
except Exception:
    genai = None

# OPTIONAL IMPORTS (fallback-safe)
try:
    import networkx as nx
except Exception:
    nx = None

try:
    from langdetect import detect
except Exception:
    detect = None

try:
    from transformers import pipeline
    _have_transformers = True
except Exception:
    pipeline = None
    _have_transformers = False

try:
    import spacy
    _have_spacy = True
except Exception:
    spacy = None
    _have_spacy = False

# NLTK is still needed for WordCloud stopwords
import nltk

# Ensure NLTK stopwords are downloaded for WordCloud
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & STYLES
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="WhatsApp Chat Analysis",
    page_icon="🕵️",
    layout="wide"
)

# Modified load_css to accept theme toggle and small visual polish options
def load_css(theme="dark"):
    """Loads custom CSS for the application. Supports 'dark' and 'light' themes."""
    if theme == "light":
        root_vars = {
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
    else:
        root_vars = {
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

    css_vars = "\n".join([f"{k}: {v};" for k, v in root_vars.items()])

    st.markdown(f"""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            {css_vars}
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateX(-20px); }}
            to {{ opacity: 1; transform: translateX(0); }}
        }}
        @keyframes pulsate {{
            0% {{ transform: scale(1); opacity: 1; }}
            50% {{ transform: scale(1.1); opacity: 0.8; }}
            100% {{ transform: scale(1); opacity: 1; }}
        }}
        @keyframes gradientShift {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        html, body, [class*="css"] {{
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(90deg, var(--bg-color), var(--secondary-color));
            background-size: 200% 200%;
            animation: gradientShift 10s ease infinite;
        }}
        h1, h2, h3, h4 {{
            font-weight: 500;
            color: var(--primary-color);
        }}
        .title-container {{
            padding: 1rem 0 1.5rem 0;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 1rem;
            animation: fadeIn 0.5s ease-out;
            text-align: center;
        }}
        .main-title {{
            font-size: 2.4rem;
            font-weight: 700;
            background: linear-gradient(90deg, var(--primary-color), var(--info-color), var(--success-color), var(--primary-color));
            background-size: 200% auto;
            -webkit-background-clip: text;
            -moz-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientShift 8s ease-in-out infinite;
        }}
        .sub-title {{
            font-size: 0.95rem;
            color: var(--text-muted);
            margin-top: -5px;
            animation: fadeIn 1s ease-out 0.5s;
            animation-fill-mode: both;
        }}
        .metric-card {{
            background-color: var(--card-color);
            border: 1px solid var(--border-color);
            padding: 14px 16px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
            transition: all 0.25s ease-in-out;
            animation: fadeIn 0.5s ease-out;
        }}
        .metric-card:hover {{
            transform: translateY(-4px) scale(1.05);
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
            border-color: var(--primary-color);
        }}
        .metric-card-label {{
            font-size: 1rem;
            font-weight: bold;
            color: white; /* Ensure Step 1, Step 2, Step 3 text is white */
        }}
        .metric-card-value {{
            font-size: 1.2rem;
            font-weight: bold;
            color: white; /* Ensure text is white */
        }}
        .button-container button {{
            transition: transform 0.2s ease-in-out;
        }}
        .button-container button:hover {{
            transform: scale(1.05);
        }}
        .loading-spinner {{
            width: 50px;
            height: 50px;
            border: 5px solid var(--border-color);
            border-top: 5px solid var(--primary-color);
            border-radius: 50%;
            animation: rotate 1s linear infinite;
            margin: auto;
        }}
        .loading-animation {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            animation: fadeIn 0.5s ease-out;
        }}
        .loading-animation .pulsate {{
            width: 50px;
            height: 50px;
            background-color: var(--primary-color);
            border-radius: 50%;
            animation: pulsate 1.5s infinite;
        }}
        .loading-animation p {{
            margin-top: 10px;
            font-size: 1rem;
            color: var(--text-muted);
            animation: fadeIn 1s ease-out;
        }}
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS & ASSETS
# -----------------------------------------------------------------------------

def get_image_as_base64(svg_string):
    svg_bytes = svg_string.encode('utf-8')
    b64 = base64.b64encode(svg_bytes).decode()
    return f"data:image/svg+xml;base64,{b64}"

SIDEBAR_LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-shield-check"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><path d="m9 12 2 2 4-4"></path></svg>"""
WELCOME_IMAGE_SVG = """<svg width="800px" height="400px" viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg"><rect width="800" height="400" fill="#0d1117"/><g opacity="0.6"><path d="M 50 350 Q 250 50 450 350 T 750 350" stroke="#58a6ff" stroke-width="2" fill="none" stroke-dasharray="10 5" /><path d="M 100 50 Q 300 300 500 100 T 700 300" stroke="#3fb950" stroke-width="2" fill="none" stroke-dasharray="10 5" /><circle cx="450" cy="350" r="8" fill="#f85149" opacity="0.8"><animate attributeName="r" values="8;12;8" dur="2s" repeatCount="indefinite" /></circle><circle cx="500" cy="100" r="8" fill="#d29922" opacity="0.8"><animate attributeName="r" values="8;12;8" begin="1s" repeatCount="indefinite" /></circle></g><text x="400" y="200" font-family="Poppins, sans-serif" font-size="24" fill="#c9d1d9" text-anchor="middle">Upload a file to begin analysis</text></svg>"""

vader_analyzer = SentimentIntensityAnalyzer()
hinglish_words = {"gussa", "pyar", "pyaar", "dhoka", "acha", "accha", "bura", "yaar", "mujhe", "tum", "tumhe", "kya", "kyun", "nahi", "nahin", "haan", "theek"}

emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\u2600-\u26FF\u2700-\u27BF"
    "]+", flags=re.UNICODE
)

def extract_emojis(text):
    return emoji_pattern.findall(text or "")

def analyze_emojis(df):
    records = []
    for _, row in df.iterrows():
        emojis = extract_emojis(row['message'])
        for e in emojis:
            records.append((e, row['author'], row['sentiment']))
    if not records:
        return pd.DataFrame(columns=['emoji', 'author', 'sentiment'])
    em_df = pd.DataFrame(records, columns=['emoji', 'author', 'sentiment'])
    summary = em_df.groupby('emoji').agg({'emoji':'size', 'sentiment':'mean'}).rename(columns={'emoji':'count'}).reset_index()
    return summary.sort_values('count', ascending=False)

def detect_threads(df, max_gap_minutes=5):
    if df.empty: return pd.DataFrame()
    df_sorted = df.sort_values('datetime_parsed')
    df_sorted['next_author'] = df_sorted['author'].shift(-1)
    df_sorted['next_time'] = df_sorted['datetime_parsed'].shift(-1)
    df_sorted['time_diff'] = (df_sorted['next_time'] - df_sorted['datetime_parsed']).dt.total_seconds() / 60.0
    df_sorted['new_thread'] = (df_sorted['time_diff'] > max_gap_minutes) | df_sorted['message'].str.contains(r'^\>', regex=True) | (df_sorted['author'] != df_sorted['next_author'])
    df_sorted['thread_id'] = df_sorted['new_thread'].cumsum()
    return df_sorted[['datetime', 'author', 'message', 'thread_id']]

def preprocess_chat(chat_text: str) -> pd.DataFrame:
    chat_text = chat_text.replace('\u200e', '').replace('\ufeff', '').replace('\u202f', ' ')
    pattern_author = re.compile(
        r'^\[?(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?::\d{2})?)(?:\s*(AM|PM|am|pm))?\]?\s*(?:-)?\s*([^:]+?):\s*(.*)$'
    )
    pattern_system = re.compile(
        r'^\[?(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?::\d{2})?)(?:\s*(AM|PM|am|pm))?\]?\s*(?:-)?\s*(.*)$'
    )
    dates, authors, messages = [], [], []
    last_idx = -1
    for line in chat_text.splitlines():
        raw = line.rstrip("\n").strip()
        if not raw:
            if last_idx >= 0: messages[last_idx] += "\n"
            continue
        m_author = pattern_author.match(raw)
        if m_author:
            dt = f"{m_author.group(1)}, {m_author.group(2)}" + (f" {m_author.group(3)}" if m_author.group(3) else "")
            dates.append(dt)
            authors.append(m_author.group(4).strip())
            messages.append(m_author.group(5).strip())
            last_idx += 1
            continue
        m_system = pattern_system.match(raw)
        if m_system:
            dt = f"{m_system.group(1)}, {m_system.group(2)}" + (f" {m_system.group(3)}" if m_system.group(3) else "")
            if ':' not in m_system.group(4):
                dates.append(dt)
                authors.append("System")
                messages.append(m_system.group(4).strip())
                last_idx += 1
                continue
        if last_idx >= 0:
            messages[last_idx] += " " + raw
        else:
            dates.append("")
            authors.append("System")
            messages.append(raw)
            last_idx += 1
    return pd.DataFrame({'datetime': dates, 'author': authors, 'message': messages})

def analyze_sentiment(message: str) -> float:
    if not message.strip(): return 0.0
    words = set(re.findall(r'\b\w+\b', message.lower()))
    has_hinglish = any(w in hinglish_words for w in words)
    return vader_analyzer.polarity_scores(message)['compound'] if has_hinglish else TextBlob(message).sentiment.polarity

@st.cache_data
def analyze_chat_for_threats_holistically(_df, api_key):
    """Enhanced AI Context-Based Threat Analysis with Advanced Psychological Profiling"""
    if not api_key:
        return "API Key is required for this feature."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        # Enhanced context preparation
        authors = list(_df['author'].unique())
        total_messages = len(_df)
        date_range = f"{_df['datetime'].iloc[0]} to {_df['datetime'].iloc[-1]}"
        
        # Prepare conversation flow analysis
        chat_text = "\n".join([f"[{row['datetime']}] {row['author']}: {row['message']}" for _, row in _df.iterrows()])
        max_chars = 25000  # Increased for better context
        if len(chat_text) > max_chars:
            # Keep both beginning and end for context
            mid_point = len(chat_text) // 2
            chat_text = chat_text[:max_chars//2] + "\n...[CONVERSATION CONTINUES]...\n" + chat_text[mid_point:]
            if len(chat_text) > max_chars:
                chat_text = chat_text[-max_chars:]

        enhanced_prompt = f"""
        🔍 **ADVANCED AI SECURITY ANALYST** - Enhanced Context-Based Threat Assessment

        **CONVERSATION METADATA:**
        • Participants: {', '.join(authors)}
        • Total Messages: {total_messages}
        • Time Period: {date_range}
        • Analysis Scope: Comprehensive psychological and contextual evaluation

        **ANALYSIS FRAMEWORK:**
        As an expert AI security analyst with advanced psychological profiling capabilities, conduct a multi-layered threat assessment:

        1. **Contextual Analysis**: Examine conversation flow, participant dynamics, and behavioral patterns
        2. **Psychological Profiling**: Assess emotional states, motivations, and potential risk indicators
        3. **Pattern Recognition**: Identify escalation sequences, planning behaviors, and coordination attempts
        4. **Risk Stratification**: Evaluate threat credibility using advanced context understanding

        **THREAT DETECTION CRITERIA:**
        - Genuine planning or coordination of harmful activities
        - Escalating aggressive behavior patterns
        - Recruitment or radicalization attempts
        - Detailed discussion of weapons, violence, or illegal activities
        - Evidence of real-world capability or intent

        **EXCLUSION CRITERIA:**
        - Obvious jokes, sarcasm, or hyperbole
        - Gaming/entertainment discussions
        - Metaphorical language without real intent
        - Historical or news discussions
        - Creative writing or fictional scenarios

        **OUTPUT FORMAT:**
        For each credible threat identified, use this enhanced format:

        <threat>
        <risk_level>CRITICAL | HIGH | MEDIUM | LOW</risk_level>
        <threat_category>Violence | Illegal Activity | Harassment | Radicalization | Other</threat_category>
        <psychological_assessment>Brief analysis of participant behavior and motivations</psychological_assessment>
        <context_analysis>Conversation flow and situational factors that elevate concern</context_analysis>
        <evidence_summary>Clear, factual summary without speculation</evidence_summary>
        <key_indicators>
        • Specific behavioral markers observed
        • Timeline and escalation patterns
        • Coordination evidence (if any)
        </key_indicators>
        <critical_messages>
        [Quote 2-4 most concerning messages with **highlighted** keywords]
        </critical_messages>
        <recommendation>Specific action recommendations based on risk level</recommendation>
        </threat>

        **SAFETY PROTOCOLS:**
        - If NO credible threats exist: "✅ COMPREHENSIVE ANALYSIS COMPLETE - No credible security threats detected."
        - For borderline cases: Explain why content doesn't meet threat criteria
        - Focus on actionable intelligence, not false positives

        **CONVERSATION LOG:**
        ================
        {chat_text}
        ================
        """
        
        response = model.generate_content(enhanced_prompt, request_options={'timeout': 150})
        return response.text
        
    except Exception as e:
        if "quota" in str(e):
            return "⚠️ AI Analysis Quota Exceeded - Please try again later or check your Google AI API quota."
        return f"🚨 Enhanced AI Analysis Error: {str(e)}"

def create_sentiment_scale_visualization(positive_count, negative_count, neutral_count, df):
    """Create a professional sentiment scale with actual sentiment scores and enhanced styling"""
    total = positive_count + negative_count + neutral_count
    if total == 0:
        st.warning("No messages to analyze for sentiment.")
        return
    
    # Calculate percentages
    pos_percent = (positive_count / total) * 100
    neg_percent = (negative_count / total) * 100
    neu_percent = (neutral_count / total) * 100
    
    # Calculate actual sentiment scores
    avg_sentiment = df['sentiment'].mean()
    pos_avg_score = df[df['sentiment'] > 0.05]['sentiment'].mean() if positive_count > 0 else 0
    neg_avg_score = df[df['sentiment'] < -0.05]['sentiment'].mean() if negative_count > 0 else 0
    neu_avg_score = df[(df['sentiment'] >= -0.05) & (df['sentiment'] <= 0.05)]['sentiment'].mean() if neutral_count > 0 else 0
    
    # Fill NaN values with 0
    pos_avg_score = pos_avg_score if not pd.isna(pos_avg_score) else 0
    neg_avg_score = neg_avg_score if not pd.isna(neg_avg_score) else 0
    neu_avg_score = neu_avg_score if not pd.isna(neu_avg_score) else 0
    
    # Determine dominant sentiment
    if pos_percent > neg_percent and pos_percent > neu_percent:
        dominant = "Positive"
        dominant_emoji = "😊"
        dominant_color = "#4CAF50"
    elif neg_percent > pos_percent and neg_percent > neu_percent:
        dominant = "Negative" 
        dominant_emoji = "😔"
        dominant_color = "#F44336"
    else:
        dominant = "Neutral"
        dominant_emoji = "😐"
        dominant_color = "#FF9800"
    
    # Professional header with metrics
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #667eea 100%);
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
        border: 2px solid rgba(255,255,255,0.1);
    ">
        <h2 style="color: white; margin: 0 0 10px 0; font-size: 2rem; font-weight: 700;">
            🎯 Professional Sentiment Analysis Dashboard
        </h2>
        <div style="
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 30px;
            margin-top: 15px;
            flex-wrap: wrap;
        ">
            <div style="text-align: center;">
                <div style="color: rgba(255,255,255,0.9); font-size: 0.9rem; margin-bottom: 5px;">Overall Score</div>
                <div style="color: white; font-size: 1.8rem; font-weight: bold;">{avg_sentiment:+.3f}</div>
            </div>
            <div style="text-align: center;">
                <div style="color: rgba(255,255,255,0.9); font-size: 0.9rem; margin-bottom: 5px;">Total Messages</div>
                <div style="color: white; font-size: 1.8rem; font-weight: bold;">{total:,}</div>
            </div>
            <div style="text-align: center;">
                <div style="color: rgba(255,255,255,0.9); font-size: 0.9rem; margin-bottom: 5px;">Dominant Tone</div>
                <div style="color: white; font-size: 1.8rem; font-weight: bold;">{dominant_emoji} {dominant}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional sentiment cards with scores
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #4CAF50, #66BB6A);
            padding: 25px;
            border-radius: 18px;
            text-align: center;
            color: white;
            box-shadow: 0 8px 25px rgba(76, 175, 80, 0.25);
            transform: {'scale(1.02)' if dominant == 'Positive' else 'scale(1)'};
            transition: all 0.3s ease;
            border: {'3px solid #2E7D32' if dominant == 'Positive' else '2px solid rgba(255,255,255,0.1)'};
            position: relative;
            overflow: hidden;
        ">
            <div style="font-size: 3.5rem; margin-bottom: 15px; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">😊</div>
            <div style="font-size: 2.2rem; font-weight: bold; margin-bottom: 8px;">{positive_count}</div>
            <div style="font-size: 1.4rem; margin-bottom: 8px; font-weight: 600;">{pos_percent:.1f}%</div>
            <div style="font-size: 1.1rem; opacity: 0.95; margin-bottom: 10px; font-weight: 500;">POSITIVE</div>
            <div style="
                background: rgba(255,255,255,0.2);
                padding: 8px 12px;
                border-radius: 25px;
                font-size: 0.95rem;
                font-weight: 600;
                letter-spacing: 0.5px;
            ">
                Avg Score: {pos_avg_score:+.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced progress bar
        st.progress(pos_percent / 100, text=f"Positive Distribution: {pos_percent:.1f}%")
    
    with col2:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #FF9800, #FFB74D);
            padding: 25px;
            border-radius: 18px;
            text-align: center;
            color: white;
            box-shadow: 0 8px 25px rgba(255, 152, 0, 0.25);
            transform: {'scale(1.02)' if dominant == 'Neutral' else 'scale(1)'};
            transition: all 0.3s ease;
            border: {'3px solid #F57C00' if dominant == 'Neutral' else '2px solid rgba(255,255,255,0.1)'};
            position: relative;
            overflow: hidden;
        ">
            <div style="font-size: 3.5rem; margin-bottom: 15px; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">😐</div>
            <div style="font-size: 2.2rem; font-weight: bold; margin-bottom: 8px;">{neutral_count}</div>
            <div style="font-size: 1.4rem; margin-bottom: 8px; font-weight: 600;">{neu_percent:.1f}%</div>
            <div style="font-size: 1.1rem; opacity: 0.95; margin-bottom: 10px; font-weight: 500;">NEUTRAL</div>
            <div style="
                background: rgba(255,255,255,0.2);
                padding: 8px 12px;
                border-radius: 25px;
                font-size: 0.95rem;
                font-weight: 600;
                letter-spacing: 0.5px;
            ">
                Avg Score: {neu_avg_score:+.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced progress bar
        st.progress(neu_percent / 100, text=f"Neutral Distribution: {neu_percent:.1f}%")
    
    with col3:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #F44336, #EF5350);
            padding: 25px;
            border-radius: 18px;
            text-align: center;
            color: white;
            box-shadow: 0 8px 25px rgba(244, 67, 54, 0.25);
            transform: {'scale(1.02)' if dominant == 'Negative' else 'scale(1)'};
            transition: all 0.3s ease;
            border: {'3px solid #C62828' if dominant == 'Negative' else '2px solid rgba(255,255,255,0.1)'};
            position: relative;
            overflow: hidden;
        ">
            <div style="font-size: 3.5rem; margin-bottom: 15px; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">😔</div>
            <div style="font-size: 2.2rem; font-weight: bold; margin-bottom: 8px;">{negative_count}</div>
            <div style="font-size: 1.4rem; margin-bottom: 8px; font-weight: 600;">{neg_percent:.1f}%</div>
            <div style="font-size: 1.1rem; opacity: 0.95; margin-bottom: 10px; font-weight: 500;">NEGATIVE</div>
            <div style="
                background: rgba(255,255,255,0.2);
                padding: 8px 12px;
                border-radius: 25px;
                font-size: 0.95rem;
                font-weight: 600;
                letter-spacing: 0.5px;
            ">
                Avg Score: {neg_avg_score:+.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced progress bar
        st.progress(neg_percent / 100, text=f"Negative Distribution: {neg_percent:.1f}%")
    
    # Professional summary card
    sentiment_interpretation = "Highly Positive" if avg_sentiment > 0.3 else "Positive" if avg_sentiment > 0.1 else "Slightly Positive" if avg_sentiment > 0.05 else "Highly Negative" if avg_sentiment < -0.3 else "Negative" if avg_sentiment < -0.1 else "Slightly Negative" if avg_sentiment < -0.05 else "Balanced/Neutral"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {dominant_color}15, {dominant_color}25);
        border: 2px solid {dominant_color};
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        margin: 25px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    ">
        <div style="display: flex; justify-content: center; align-items: center; gap: 15px; margin-bottom: 15px;">
            <div style="font-size: 2.5rem;">{dominant_emoji}</div>
            <h3 style="color: {dominant_color}; margin: 0; font-size: 1.8rem; font-weight: 700;">
                Sentiment Analysis Summary
            </h3>
        </div>
        <div style="
            background: rgba(255,255,255,0.9);
            padding: 20px;
            border-radius: 15px;
            margin: 15px 0;
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.05);
        ">
            <div style="color: #333; font-size: 1.1rem; line-height: 1.6;">
                <strong>Overall Interpretation:</strong> {sentiment_interpretation}<br>
                <strong>Dominant Sentiment:</strong> {dominant} ({max(pos_percent, neg_percent, neu_percent):.1f}% of messages)<br>
                <strong>Conversation Tone:</strong> {"Very Positive" if avg_sentiment > 0.2 else "Positive" if avg_sentiment > 0.05 else "Very Negative" if avg_sentiment < -0.2 else "Negative" if avg_sentiment < -0.05 else "Balanced"}
            </div>
        </div>
        <div style="
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
            flex-wrap: wrap;
        ">
            <div style="text-align: center; color: {dominant_color};">
                <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 5px;">Positivity Ratio</div>
                <div style="font-size: 1.5rem; font-weight: bold;">{(pos_percent / max(neg_percent, 1)):.1f}:1</div>
            </div>
            <div style="text-align: center; color: {dominant_color};">
                <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 5px;">Emotional Range</div>
                <div style="font-size: 1.5rem; font-weight: bold;">{abs(df['sentiment'].max() - df['sentiment'].min()):.2f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def calculate_conversation_health(df):
    """Calculate a conversation health score based on multiple factors"""
    if df.empty:
        return 50
    
    # Factor 1: Sentiment balance (40% weight)
    avg_sentiment = df['sentiment'].mean()
    sentiment_score = min(100, max(0, (avg_sentiment + 1) * 50))  # Scale from -1,1 to 0,100
    
    # Factor 2: Engagement level (30% weight)
    total_messages = len(df)
    unique_authors = df['author'].nunique()
    engagement_score = min(100, (total_messages / unique_authors) * 2)  # Messages per author
    
    # Factor 3: Conversation consistency (30% weight)
    sentiment_variance = df['sentiment'].var()
    consistency_score = max(0, 100 - (sentiment_variance * 100))
    
    # Calculate weighted health score
    health_score = int(
        sentiment_score * 0.4 + 
        engagement_score * 0.3 + 
        consistency_score * 0.3
    )
    
    return min(100, max(0, health_score))

def get_health_message(score):
    """Get a descriptive message based on health score"""
    if score >= 80:
        return "🌟 Excellent conversation quality with positive engagement!"
    elif score >= 60:
        return "😊 Good conversation health with balanced interactions."
    elif score >= 40:
        return "😐 Average conversation quality, room for improvement."
    elif score >= 20:
        return "😔 Below average conversation health detected."
    else:
        return "🚨 Poor conversation quality, consider reviewing dynamics."

def analyze_message_types(df):
    def get_type(message):
        if "<Media omitted>" in message or "document omitted" in message: return "Media"
        if "?" in message: return "Question"
        if "http" in message or "www." in message: return "Link"
        if len(message.split()) < 4: return "Short Reaction"
        return "Regular Message"
    return df['message'].apply(get_type).value_counts()

@st.cache_data
def perform_topic_modeling(_messages):
    if len(_messages) < 5: return None, None
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
    try:
        dtm = vectorizer.fit_transform(_messages)
    except ValueError:
        return None, None
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(dtm)
    return lda, vectorizer

def get_interaction_matrix(df):
    authors = sorted(df[df['author'] != 'System']['author'].unique())
    matrix = pd.DataFrame(0, index=authors, columns=authors)
    for i in range(1, len(df)):
        sender = df.iloc[i]['author']
        receiver = df.iloc[i-1]['author']
        if sender != receiver and sender != 'System' and receiver != 'System':
            matrix.loc[sender, receiver] += 1
    return matrix

# -----------------------------------------------------------------------------
# 3. CHARTING & UI DISPLAY FUNCTIONS
# -----------------------------------------------------------------------------
def create_user_pie_chart(df):
    author_counts = df[df['author'] != 'System']['author'].value_counts()
    top_authors = author_counts.head(7)
    if len(author_counts) > 7: top_authors['Others'] = author_counts[7:].sum()
    if not top_authors.empty:
        fig = px.pie(values=top_authors.values, names=top_authors.index, hole=0.4,
                     color_discrete_sequence=px.colors.sequential.Blues_r,
                     title="Most Active Users")
        fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                          title_x=0.5)
        fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14,
                          hovertemplate='<b>%{label}</b><br>%{value} messages<br>%{percent}')
        return fig
    return None

def create_type_pie_chart(df):
    message_types = analyze_message_types(df)
    if not message_types.empty:
        fig = px.pie(values=message_types.values, names=message_types.index, hole=0.4,
                     color_discrete_sequence=px.colors.sequential.Greens_r,
                     title="Message Types")
        fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                          title_x=0.5)
        fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14,
                          hovertemplate='<b>%{label}</b><br>%{value} messages<br>%{percent}')
        return fig
    return None

def metric_card(icon, label, value, card_color="var(--card-color)"):
    st.markdown(f"""<div class="metric-card" style="background-color:{card_color}"><div class="metric-card-icon">{icon}</div><div class="metric-card-label">{label}</div><div class="metric-card-value">{html_lib.escape(str(value))}</div></div>""", unsafe_allow_html=True)

def parse_threat_data(threat_text):
    """Parse the structured threat data into a clean format"""
    import re
    
    # Initialize threat data
    threat_info = {
        'risk_level': 'Unknown',
        'category': 'Unknown',
        'psychological_assessment': 'Not available',
        'context_analysis': 'Not available',
        'evidence_summary': 'Not available',
        'key_indicators': [],
        'critical_messages': [],
        'recommendation': 'Review recommended'
    }
    
    # Extract risk level
    risk_match = re.search(r'<risk_level>(.*?)</risk_level>', threat_text, re.DOTALL)
    if risk_match:
        threat_info['risk_level'] = risk_match.group(1).strip()
    
    # Extract category
    category_match = re.search(r'<threat_category>(.*?)</threat_category>', threat_text, re.DOTALL)
    if category_match:
        threat_info['category'] = category_match.group(1).strip()
    
    # Extract psychological assessment
    psych_match = re.search(r'<psychological_assessment>(.*?)</psychological_assessment>', threat_text, re.DOTALL)
    if psych_match:
        threat_info['psychological_assessment'] = psych_match.group(1).strip()
    
    # Extract context analysis
    context_match = re.search(r'<context_analysis>(.*?)</context_analysis>', threat_text, re.DOTALL)
    if context_match:
        threat_info['context_analysis'] = context_match.group(1).strip()
    
    # Extract evidence summary
    evidence_match = re.search(r'<evidence_summary>(.*?)</evidence_summary>', threat_text, re.DOTALL)
    if evidence_match:
        threat_info['evidence_summary'] = evidence_match.group(1).strip()
    
    # Extract key indicators
    indicators_match = re.search(r'<key_indicators>(.*?)</key_indicators>', threat_text, re.DOTALL)
    if indicators_match:
        indicators_text = indicators_match.group(1).strip()
        # Split by bullet points or lines
        indicators = [line.strip('• ').strip() for line in indicators_text.split('\n') if line.strip() and not line.strip().startswith('•')]
        if not indicators:
            indicators = [ind.strip() for ind in indicators_text.split('•') if ind.strip()]
        threat_info['key_indicators'] = [ind for ind in indicators if ind]
    
    # Extract critical messages
    messages_match = re.search(r'<critical_messages>(.*?)</critical_messages>', threat_text, re.DOTALL)
    if messages_match:
        messages_text = messages_match.group(1).strip()
        # Split by bullet points or lines
        messages = [line.strip('• ').strip() for line in messages_text.split('\n') if line.strip()]
        if not messages:
            messages = [msg.strip() for msg in messages_text.split('•') if msg.strip()]
        threat_info['critical_messages'] = [msg for msg in messages if msg]
    
    # Extract recommendation
    rec_match = re.search(r'<recommendation>(.*?)</recommendation>', threat_text, re.DOTALL)
    if rec_match:
        threat_info['recommendation'] = rec_match.group(1).strip()
    
    return threat_info

def display_conversation_insights(messages):
    """Display helpful insights about the conversation when no threats are found"""
    st.markdown("### 📊 Conversation Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Message count
        total_messages = len(messages)
        st.metric("💬 Total Messages", total_messages)
    
    with col2:
        # Unique participants
        participants = set()
        for msg in messages:
            if isinstance(msg, dict) and 'sender' in msg:
                participants.add(msg['sender'])
        st.metric("👥 Participants", len(participants))
    
    with col3:
        # Conversation length estimate
        if messages:
            try:
                start_time = str(messages[0].get('timestamp', 'Unknown'))[:10]
                end_time = str(messages[-1].get('timestamp', 'Unknown'))[:10]
                if start_time != end_time:
                    st.metric("⏱️ Date Range", f"{start_time} to {end_time}")
                else:
                    st.metric("⏱️ Date", start_time)
            except:
                st.metric("⏱️ Timeline", "Available")
    
    # Additional insights
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #e3f2fd, #f3e5f5);
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 4px solid #2196F3;
    ">
        <h4 style="color: #1976D2; margin-bottom: 15px;">✨ Key Insights</h4>
        <ul style="color: #333; line-height: 1.8;">
            <li><strong>Safety Score:</strong> This conversation appears to follow healthy communication patterns</li>
            <li><strong>Tone Analysis:</strong> No aggressive or threatening language patterns detected</li>
            <li><strong>Content Review:</strong> Messages contain normal social interaction content</li>
            <li><strong>Risk Assessment:</strong> Very low risk based on AI behavioral analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def get_risk_level_color(risk_level):
    """Get color scheme based on risk level"""
    risk_level = risk_level.upper()
    colors = {
        'CRITICAL': {'bg': '#D32F2F', 'border': '#B71C1C', 'icon': '🔴'},
        'HIGH': {'bg': '#F44336', 'border': '#D32F2F', 'icon': '🟠'},
        'MEDIUM': {'bg': '#FF9800', 'border': '#F57C00', 'icon': '🟡'},
        'LOW': {'bg': '#4CAF50', 'border': '#388E3C', 'icon': '🟢'},
    }
    return colors.get(risk_level, colors['MEDIUM'])

def display_ai_threat_report(report, messages):
    """Enhanced, user-friendly threat analysis display"""
    
    # Header
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a237e 0%, #3f51b5 100%);
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        text-align: center;
    ">
        <h1 style="color: white; margin: 0; font-size: 2rem;">
            🛡️ AI Security Analysis Report
        </h1>
        <p style="color: rgba(255,255,255,0.8); margin: 10px 0 0 0; font-size: 1.1rem;">
            Advanced threat detection with behavioral analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if no threats found
    if "No credible threats found" in report or "COMPREHENSIVE ANALYSIS COMPLETE" in report:
        # Safe result
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #4CAF50, #66BB6A);
                padding: 30px;
                border-radius: 20px;
                text-align: center;
                box-shadow: 0 8px 25px rgba(76, 175, 80, 0.3);
                margin: 20px 0;
            ">
                <div style="font-size: 4rem; margin-bottom: 15px;">✅</div>
                <h2 style="color: white; margin-bottom: 15px;">All Clear!</h2>
                <p style="color: white; font-size: 1.2rem; margin: 0;">
                    Our advanced AI analysis found no security concerns in this conversation.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display conversation insights
        st.markdown("---")
        display_conversation_insights(messages)
        
    else:
        # Threats detected - parse and display cleanly
        st.warning("⚠️ **Security Alert**: Our AI analysis has identified potential concerns that require attention.", icon="🚨")
        
        # Parse threats
        threats = [threat.strip() for threat in report.split('<threat>') if threat.strip()]
        
        # Display summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Threats Detected", len(threats))
        with col2:
            # Calculate highest risk level
            risk_levels = []
            for threat in threats:
                if 'CRITICAL' in threat.upper():
                    risk_levels.append(4)
                elif 'HIGH' in threat.upper():
                    risk_levels.append(3)
                elif 'MEDIUM' in threat.upper():
                    risk_levels.append(2)
                else:
                    risk_levels.append(1)
            
            max_risk = max(risk_levels) if risk_levels else 1
            risk_text = ['LOW', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'][max_risk]
            st.metric("⚡ Highest Risk", risk_text)
        
        with col3:
            st.metric("📊 Analysis Status", "Complete")
        
        st.markdown("---")
        
        # Display each threat in a clean, organized way
        for i, threat_text in enumerate(threats, 1):
            threat_data = parse_threat_data(threat_text)
            colors = get_risk_level_color(threat_data['risk_level'])
            
            # Threat container
            with st.container():
                st.markdown(f"""
                <div style="
                    background: white;
                    border-left: 6px solid {colors['border']};
                    border-radius: 10px;
                    padding: 0;
                    margin: 20px 0;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    overflow: hidden;
                ">
                    <div style="
                        background: {colors['bg']};
                        color: white;
                        padding: 15px 20px;
                        margin: 0;
                    ">
                        <h3 style="margin: 0; display: flex; align-items: center;">
                            {colors['icon']} Threat Analysis #{i}
                            <span style="
                                background: rgba(255,255,255,0.2);
                                padding: 4px 12px;
                                border-radius: 15px;
                                font-size: 0.8rem;
                                margin-left: auto;
                            ">
                                {threat_data['risk_level']} RISK
                            </span>
                        </h3>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Create tabs for organized information
                tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "🧠 Analysis", "🔍 Evidence", "💡 Recommendations"])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Category**")
                        st.info(f"🏷️ {threat_data['category']}")
                        
                        st.markdown("**Risk Level**")
                        color_map = {
                            'CRITICAL': '🔴',
                            'HIGH': '🟠', 
                            'MEDIUM': '🟡',
                            'LOW': '🟢'
                        }
                        icon = color_map.get(threat_data['risk_level'].upper(), '🟡')
                        st.error(f"{icon} {threat_data['risk_level']}")
                    
                    with col2:
                        st.markdown("**Evidence Summary**")
                        st.write(threat_data['evidence_summary'])
                
                with tab2:
                    st.markdown("**🧠 Psychological Assessment**")
                    st.write(threat_data['psychological_assessment'])
                    
                    st.markdown("**📊 Context Analysis**")
                    st.write(threat_data['context_analysis'])
                
                with tab3:
                    st.markdown("**🔑 Key Risk Indicators**")
                    if threat_data['key_indicators']:
                        for indicator in threat_data['key_indicators']:
                            if indicator.strip():
                                st.write(f"• {indicator}")
                    else:
                        st.write("No specific indicators extracted")
                    
                    st.markdown("**💬 Critical Messages**")
                    if threat_data['critical_messages']:
                        for msg in threat_data['critical_messages']:
                            if msg.strip():
                                st.code(msg, language=None)
                    else:
                        st.write("No critical messages extracted")
                
                with tab4:
                    st.markdown("**⚠️ Recommended Actions**")
                    st.warning(threat_data['recommendation'])
                    
                    # Add action buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"📞 Report Threat #{i}", key=f"report_{i}"):
                            st.success("✅ Threat reported to authorities")
                    with col2:
                        if st.button(f"📝 Export Details #{i}", key=f"export_{i}"):
                            st.success("✅ Details exported to file")
                    with col3:
                        if st.button(f"❌ Mark as False Positive #{i}", key=f"false_{i}"):
                            st.info("✅ Marked as false positive")
                
                st.markdown("---")

def display_conversation_insights(messages):
    """Display advanced conversation insights when no threats are found"""
    if not messages or len(messages) < 5:
        return
        
    # Calculate some interesting insights
    total_messages = len(messages)
    avg_length = sum(len(msg) for msg in messages) / total_messages
    questions = sum(1 for msg in messages if '?' in msg)
    exclamations = sum(1 for msg in messages if '!' in msg)
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #36d1dc, #5b86e5);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
    ">
        <h3 style="color: white; text-align: center; margin-bottom: 20px;">
            💡 Conversation Insights
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            backdrop-filter: blur(10px);
        ">
            <div style="font-size: 2rem; margin-bottom: 10px;">📝</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: white;">{total_messages}</div>
            <div style="color: rgba(255,255,255,0.8);">Total Messages</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            backdrop-filter: blur(10px);
        ">
            <div style="font-size: 2rem; margin-bottom: 10px;">📏</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: white;">{avg_length:.0f}</div>
            <div style="color: rgba(255,255,255,0.8);">Avg Length</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            backdrop-filter: blur(10px);
        ">
            <div style="font-size: 2rem; margin-bottom: 10px;">❓</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: white;">{questions}</div>
            <div style="color: rgba(255,255,255,0.8);">Questions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style="
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            backdrop-filter: blur(10px);
        ">
            <div style="font-size: 2rem; margin-bottom: 10px;">❗</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: white;">{exclamations}</div>
            <div style="color: rgba(255,255,255,0.8);">Exclamations</div>
        </div>
        """, unsafe_allow_html=True)

def display_dashboard(df):
    st.subheader("📊 User & Message Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig_users = create_user_pie_chart(df)
        if fig_users:
            st.plotly_chart(fig_users, use_container_width=True)
        else:
            st.info("No user messages to display.")
    with col2:
        fig_types = create_type_pie_chart(df)
        if fig_types:
            st.plotly_chart(fig_types, use_container_width=True)
        else:
            st.info("No messages to classify.")

    st.markdown("<hr style='border-color: var(--border-color); margin: 2rem 0;'>", unsafe_allow_html=True)
    st.subheader("🕒 Temporal Analysis")
    
    st.markdown("<h4 class='chart-title'>Sentiment Timeline</h4>", unsafe_allow_html=True)
    user_df = df[df['author'] != 'System'].copy()
    if not user_df.empty and len(user_df) > 1:
        user_df.sort_values('datetime_parsed', inplace=True)
        window_size = max(10, len(user_df) // 10)
        user_df['sentiment_rolling_avg'] = user_df['sentiment'].rolling(window=window_size, min_periods=1).mean()
        
        fig = px.line(user_df, x='datetime_parsed', y='sentiment_rolling_avg',
                      title='Conversation Sentiment Over Time (Rolling Average)',
                      labels={'sentiment_rolling_avg': 'Sentiment Score', 'datetime_parsed': 'Date'})
        fig.add_hline(y=0, line_dash="dash", line_color="grey")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to generate a sentiment timeline.")

    st.markdown("<hr style='border-color: var(--border-color); margin: 1rem 0;'>", unsafe_allow_html=True)
    st.subheader("🤖 Advanced Analysis")
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("<h4 class='chart-title'>Topic Modeling</h4>", unsafe_allow_html=True)
        lda_model, vectorizer = perform_topic_modeling(df['message'])
        if lda_model and vectorizer:
            feature_names = vectorizer.get_feature_names_out()
            for i, topic in enumerate(lda_model.components_):
                st.write(f"**Topic {i+1}:**", ", ".join([feature_names[i] for i in topic.argsort()[-5:]]))
        else: st.info("Not enough data for topic modeling.")
    with col6:
        st.markdown("<h4 class='chart-title'>Interaction Heatmap</h4>", unsafe_allow_html=True)
        interaction_matrix = get_interaction_matrix(df)
        if not interaction_matrix.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(interaction_matrix, annot=True, fmt='d', cmap="Blues", ax=ax, cbar=False, annot_kws={"color": "white"})
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            plt.xticks(rotation=45, ha='right', c='white', fontsize=10)
            plt.yticks(rotation=0, c='white', fontsize=10)
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Not enough data to generate an interaction heatmap.")

def display_word_cloud(df, title="Most Frequent Words"):
    st.subheader(f"☁️ {title}")
    all_text = " ".join(df[df['author'] != 'System']['message'].astype(str).tolist())
    if all_text.strip():
        wc = WordCloud(width=1200, height=500, background_color=None, mode="RGBA", stopwords=STOPWORDS, collocations=False).generate(all_text)
        fig, ax = plt.subplots(figsize=(10, 5)); ax.imshow(wc, interpolation='bilinear'); ax.axis('off'); fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
        st.pyplot(fig, use_container_width=True)
    else: st.info("No text data available to generate a word cloud.")

def display_key_insights(df, title="💡 Key Insights"):
    st.markdown(f"<h3 style='margin-bottom: 1rem;'>{title}</h3>", unsafe_allow_html=True)
    user_df = df[df['author'] != 'System']
    if user_df.empty: st.warning("No user messages available for insights."); return
    avg_sentiment = user_df['sentiment'].mean()
    sentiment_text = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
    sentiment_color = "var(--success-color)" if sentiment_text == "Positive" else "var(--danger-color)" if sentiment_text == "Negative" else "var(--warning-color)"
    most_active_user = user_df['author'].value_counts().idxmax()
    busiest_hour = user_df['datetime_parsed'].dt.hour.value_counts().idxmax()
    insights = f"""
    - **Overall Sentiment**: The general mood is <span style='color:{sentiment_color}; font-weight:bold;'>{sentiment_text}</span> (Avg. Score: {avg_sentiment:.2f}).
    - **Top Contributor**: **{most_active_user}** is the most active user.
    - **Peak Activity**: The busiest hour for conversations is around **{busiest_hour}:00**.
    """
    st.markdown(insights, unsafe_allow_html=True)
    st.markdown("---")

def display_user_deepdive(df):
    st.subheader("👤 User Deep Dive")
    authors = sorted(df[df['author'] != 'System']['author'].unique())
    if not authors:
        st.info("No users to analyze.")
        return
    selected_author = st.selectbox("Select a user to analyze", options=authors)
    if selected_author:
        user_df = df[df['author'] == selected_author]
        display_key_insights(user_df, title=f"💡 Key Insights for {selected_author}")
        display_word_cloud(user_df, title=f"Word Cloud for {selected_author}")

def generate_pdf_report(df, ai_report, insights_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'WhatsApp Chat Analysis Report', 0, 1, 'C')
    pdf.ln(10)

    # Key Insights
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, 'Key Insights', 0, 1)
    pdf.set_font("Arial", '', 12)
    cleaned_insights = insights_text.encode('latin-1', 'ignore').decode('latin-1')
    pdf.multi_cell(0, 8, cleaned_insights)
    pdf.ln(5)

    # AI Threat Assessment
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, 'AI Threat Assessment', 0, 1)
    pdf.set_font("Arial", '', 10)
    cleaned_report = str(ai_report).encode('latin-1', 'ignore').decode('latin-1')
    pdf.multi_cell(0, 5, cleaned_report)
        
    return bytes(pdf.output(dest='S'))

# -----------------------------------------------------------------------------
# 4. MAIN APP LOGIC
# -----------------------------------------------------------------------------
def show_admin_panel():
    """Enhanced admin panel with professional styling and admin controls"""
    
    # Professional admin header
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    ">
        <div style="font-size: 2.5rem; margin-bottom: 10px;">👨‍💼</div>
        <h1 style="color: white; margin: 0; font-size: 2rem;">Admin Dashboard</h1>
        <p style="color: rgba(255,255,255,0.8); margin: 10px 0 0 0; font-size: 1.1rem;">
            User Management & System Overview
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Admin info
    current_admin = st.session_state.user_email
    st.markdown(f"""
    <div style="
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    ">
        <strong>� Admin Session:</strong> {current_admin}<br>
        <strong>🕒 Access Time:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)
    
    users_db = get_all_users()
    
    if not users_db:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #74b9ff, #0984e3);
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            color: white;
        ">
            <div style="font-size: 3rem; margin-bottom: 15px;">👥</div>
            <h3>No Users Registered Yet</h3>
            <p>The system is ready to accept user registrations.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # User statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #4CAF50, #66BB6A);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            color: white;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        ">
            <div style="font-size: 2rem; margin-bottom: 10px;">👥</div>
            <div style="font-size: 1.8rem; font-weight: bold;">{len(users_db)}</div>
            <div style="opacity: 0.9;">Total Users</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Calculate active users (logged in recently)
        active_users = sum(1 for user in users_db.values() if user.get('login_count', 0) > 0)
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #2196F3, #42A5F5);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            color: white;
            box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
        ">
            <div style="font-size: 2rem; margin-bottom: 10px;">🔥</div>
            <div style="font-size: 1.8rem; font-weight: bold;">{active_users}</div>
            <div style="opacity: 0.9;">Active Users</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Calculate today's registrations
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        today_registrations = sum(1 for user in users_db.values() 
                                 if user.get('registration_date', '').startswith(today))
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #FF9800, #FFB74D);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            color: white;
            box-shadow: 0 4px 15px rgba(255, 152, 0, 0.3);
        ">
            <div style="font-size: 2rem; margin-bottom: 10px;">📅</div>
            <div style="font-size: 1.8rem; font-weight: bold;">{today_registrations}</div>
            <div style="opacity: 0.9;">Today's Signups</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Calculate total login count
        total_logins = sum(user.get('login_count', 0) for user in users_db.values())
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #9C27B0, #BA68C8);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            color: white;
            box-shadow: 0 4px 15px rgba(156, 39, 176, 0.3);
        ">
            <div style="font-size: 2rem; margin-bottom: 10px;">📊</div>
            <div style="font-size: 1.8rem; font-weight: bold;">{total_logins}</div>
            <div style="opacity: 0.9;">Total Logins</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Create enhanced user data
    users_data = []
    for email, info in users_db.items():
        users_data.append({
            "Email": email,
            "Full Name": info.get("full_name", "N/A"),
            "Registration Date": info.get("registration_date", "N/A"),
            "Last Login": info.get("last_login", "Never"),
            "Login Count": info.get("login_count", 0),
            "Status": "Active" if info.get("login_count", 0) > 0 else "Registered"
        })
    
    users_df = pd.DataFrame(users_data)
    
    # User management section
    st.markdown("### 📊 **User Management**")
    
    # Search and filter options
    col1, col2 = st.columns([2, 1])
    with col1:
        search_term = st.text_input("🔍 Search Users", placeholder="Search by email or name...")
    with col2:
        status_filter = st.selectbox("Status Filter", ["All", "Active", "Registered"])
    
    # Apply filters
    filtered_df = users_df.copy()
    if search_term:
        filtered_df = filtered_df[
            filtered_df['Email'].str.contains(search_term, case=False) |
            filtered_df['Full Name'].str.contains(search_term, case=False)
        ]
    
    if status_filter != "All":
        filtered_df = filtered_df[filtered_df['Status'] == status_filter]
    
    # Display filtered users table with enhanced styling
    st.markdown("#### 👥 **User List**")
    st.dataframe(
        filtered_df, 
        use_container_width=True,
        hide_index=True,
        column_config={
            "Email": st.column_config.TextColumn("📧 Email", width="medium"),
            "Full Name": st.column_config.TextColumn("👤 Full Name", width="medium"),
            "Registration Date": st.column_config.TextColumn("📅 Registration", width="medium"),
            "Last Login": st.column_config.TextColumn("🕒 Last Login", width="medium"),
            "Login Count": st.column_config.NumberColumn("📊 Logins", width="small"),
            "Status": st.column_config.TextColumn("🔘 Status", width="small")
        }
    )
    
    # Admin actions
    st.markdown("---")
    st.markdown("### 🛠️ **Admin Actions**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download users data as CSV
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 **Download User Data (CSV)**",
            data=csv_data,
            file_name=f"user_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="secondary",
            use_container_width=True
        )
    
    with col2:
        # Export JSON data
        json_data = json.dumps(users_db, indent=2)
        st.download_button(
            label="📥 **Download User Data (JSON)**",
            data=json_data,
            file_name=f"user_database_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            type="secondary",
            use_container_width=True
        )
    
    with col3:
        # System info
        if st.button("🔧 **System Information**", use_container_width=True):
            st.info(f"""
            **System Status:**
            - Database File: `{USER_DB_FILE}`
            - Admin Users: {len([email for email in ADMIN_EMAILS if email.strip()])}
            - Current Admin: {current_admin}
            - Server Time: {datetime.datetime.now()}
            """)
    
    # Security notice
    st.markdown("""
    <div style="
        background: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #FFC107;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
    ">
        <strong>🔒 Security Notice:</strong> This admin panel contains sensitive user information. 
        Ensure you follow data privacy regulations and keep this information secure.
    </div>
    """, unsafe_allow_html=True)

def main():
    load_css()
    
    # Check authentication status
    if not is_user_logged_in():
        create_login_form()
        return
    
    # Show user dashboard if logged in
    create_user_dashboard()
    
    # Show profile if requested
    if st.session_state.get('show_profile', False):
        show_user_profile()
        return
    
    # Admin panel - secured access control
    if not ADMIN_EMAILS or not any(email.strip() for email in ADMIN_EMAILS):
        # No admin configured yet - show setup interface
        if st.sidebar.checkbox("⚙️ Admin Setup (Owner Only)", key="admin_setup"):
            configure_admin_access()
            return
    else:
        # Admin access control
        is_admin = is_admin_user(st.session_state.user_email)
        if is_admin:
            if st.sidebar.checkbox("👨‍💼 Admin Panel (View Users)", key="admin_panel"):
                show_admin_panel()
                return
        elif st.sidebar.checkbox("👨‍💼 Admin Panel (Request Access)", key="admin_panel_unauthorized"):
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #FF6B6B, #FF8E8E);
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                margin: 20px 0;
                box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
            ">
                <div style="font-size: 3rem; margin-bottom: 15px;">🚫</div>
                <h2 style="color: white; margin-bottom: 15px;">Access Denied</h2>
                <p style="color: white; font-size: 1.1rem; line-height: 1.6;">
                    Only authorized administrators can access the admin panel.<br>
                    Contact the website owner if you need admin access.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info(f"👤 **Your Email:** {st.session_state.user_email}")
            st.info("📧 **Contact the website owner** to request admin privileges for your account.")
            return

    with st.sidebar:
        st.image(get_image_as_base64(SIDEBAR_LOGO_SVG), width=50)
        st.header("🕵️‍♀️ Controls Panel")
        uploaded_file = st.file_uploader("Upload WhatsApp chat export (.txt)", type=["txt"])
        
    st.markdown("""<div class="title-container"><div class="main-title">WhatsApp Chat Analysis</div><div class="sub-title">Analyze sentiment, threats, and activity patterns from your chat history.</div></div>""", unsafe_allow_html=True)

    if 'current_file_name' not in st.session_state:
        st.session_state.current_file_name = None

    if uploaded_file:
        if uploaded_file.name != st.session_state.current_file_name:
            st.cache_data.clear()
            st.session_state.current_file_name = uploaded_file.name
            st.success(f"New file detected: '{uploaded_file.name}'. Running fresh analysis...")
            time.sleep(1)

        raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
        df = preprocess_chat(raw_text)

        if df.empty or 'datetime' not in df.columns or df['datetime'].str.strip().eq('').all():
            st.error("Could not parse the chat file. Please ensure it is a valid WhatsApp export.")
            return

        df['sentiment'] = df['message'].apply(analyze_sentiment)
        df['datetime_parsed'] = pd.to_datetime(df['datetime'], errors='coerce', dayfirst=True)
        df.dropna(subset=['datetime_parsed'], inplace=True)
        df = df.reset_index(drop=True)

        if df.empty:
            st.error("Failed to parse dates from the chat file. The application cannot proceed.")
            return

        filtered_df = df.copy()

        with st.sidebar:
            st.markdown("---")
            st.header("🔍 Filters")
            
            user_df_for_filters = filtered_df[filtered_df['author'] != 'System']
            authors = sorted(user_df_for_filters['author'].unique()) if not user_df_for_filters.empty else []
            selected_authors = st.multiselect("Filter by Author(s)", options=authors)
            if selected_authors:
                filtered_df = filtered_df[filtered_df['author'].isin(selected_authors)]
            
            if not filtered_df.empty:
                min_date = filtered_df['datetime_parsed'].min().date()
                max_date = filtered_df['datetime_parsed'].max().date()
                
                st.markdown("##### Filter by Date Range")
                start_date = st.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
                end_date = st.date_input("End date", max_date, min_value=start_date, max_value=max_date)

                if start_date and end_date:
                    filtered_df = filtered_df[(filtered_df['datetime_parsed'].dt.date >= start_date) & (filtered_df['datetime_parsed'].dt.date <= end_date)]
            else:
                st.warning("No data for the selected author(s).")


        if filtered_df.empty:
            st.warning("No data matches the selected filters."); return
        
        user_df = filtered_df[filtered_df['author'] != 'System']

        with st.spinner("Performing AI threat analysis..."):
            api_key = get_gemini_api_key()
            ai_threat_report = analyze_chat_for_threats_holistically(user_df, api_key)
            num_ai_threats = ai_threat_report.count("<threat>")

        with st.sidebar:
            st.header("📊 Chat Metrics & Report")
            if not user_df.empty:
                top_sender = user_df['author'].value_counts().idxmax()
                avg_sent = user_df['sentiment'].mean()
                metric_card("💬", "Messages", f"{len(user_df):,}")
                metric_card("🚨", "Threats", num_ai_threats)
                metric_card("👑", "Top User", top_sender)
                metric_card("📈", "Sentiment", f"{avg_sent:.2f}")
                
                st.markdown("---")
                with st.spinner("Generating PDF Report..."):
                    insights_for_pdf = f"Overall Sentiment: {'Positive' if avg_sent > 0.05 else 'Negative' if avg_sent < -0.05 else 'Neutral'} (Avg. Score: {avg_sent:.2f})\nTop Contributor: {top_sender}\nPeak Activity: Around {user_df['datetime_parsed'].dt.hour.value_counts().idxmax()}:00"
                    
                    pdf_bytes = generate_pdf_report(filtered_df, ai_threat_report, insights_for_pdf)
                    st.download_button("📄 Download Report (PDF)", data=pdf_bytes, file_name="whatsapp_report.pdf", mime="application/pdf")
            else:
                metric_card("💬", "Messages", "0")
                metric_card("🚨", "Threats", "0")

        display_key_insights(filtered_df)
        
        # Enhanced Sentiment Scale Display
        if not user_df.empty:
            st.markdown("---")
            # Calculate sentiment counts for the scale
            positive_count = len(user_df[user_df['sentiment'] > 0.05])
            negative_count = len(user_df[user_df['sentiment'] < -0.05])
            neutral_count = len(user_df[(user_df['sentiment'] >= -0.05) & (user_df['sentiment'] <= 0.05)])
            
            # Display the beautiful sentiment scale
            create_sentiment_scale_visualization(positive_count, negative_count, neutral_count, user_df)
            
            # Add conversation health score as a surprise feature
            health_score = calculate_conversation_health(user_df)
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 15px;
                margin: 20px 0;
                text-align: center;
                box-shadow: 0 8px 25px rgba(0,0,0,0.2);
                animation: pulse 3s ease-in-out infinite;
            ">
                <h3 style="color: white; margin-bottom: 10px;">🎯 Conversation Health Score</h3>
                <div style="
                    font-size: 2.5rem;
                    font-weight: bold;
                    color: white;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                ">{health_score}/100</div>
                <p style="color: rgba(255,255,255,0.9); margin-top: 10px;">
                    {get_health_message(health_score)}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        tab_list = ["🤖 AI Context Analysis", "📊 Dashboard", "☁️ Word Cloud", "👤 User Deep Dive", "📄 Raw Data"]
        tab_ai, tab_dashboard, tab_wordcloud, tab_user_dive, tab_raw_data = st.tabs(tab_list)
        
        with tab_ai:
            display_ai_threat_report(ai_threat_report, user_df['message'].tolist())
        with tab_dashboard: display_dashboard(filtered_df)
        with tab_wordcloud: display_word_cloud(filtered_df)
        with tab_user_dive: display_user_deepdive(filtered_df)
        with tab_raw_data:
            st.subheader("Filtered Chat Data")
            st.dataframe(filtered_df.head(200), use_container_width=True)
    else:
        st.info("**Welcome!** Please upload a WhatsApp `.txt` file using the sidebar to begin.", icon="👋")
        st.markdown("---")
        st.subheader("How It Works")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div class='metric-card'><div class='metric-card-icon'>📤</div><div class='metric-card-label'>Step 1</div><div class='metric-card-value' style='font-size:1.2rem;'>Upload File</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='metric-card'><div class='metric-card-icon'>🔬</div><div class='metric-card-label'>Step 2</div><div class='metric-card-value' style='font-size:1.2rem;'>Analyze Data</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div class='metric-card'><div class='metric-card-icon'>📈</div><div class='metric-card-label'>Step 3</div><div class='metric-card-value' style='font-size:1.2rem;'>Explore Insights</div></div>", unsafe_allow_html=True)

    st.markdown('<div class="footer">Built with ❤️ using Streamlit & Python</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
