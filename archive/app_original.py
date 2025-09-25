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
    if not api_key:
        return "API Key is required for this feature."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        chat_text = "\n".join([f"{row['datetime']} - {row['author']}: {row['message']}" for _, row in _df.iterrows()])
        max_chars = 20000
        if len(chat_text) > max_chars:
            chat_text = chat_text[-max_chars:]
        prompt = f"""
        You are a security analyst. Review this entire chat log for any conversations that discuss planning or carrying out harmful, violent, or illegal acts like terrorism, violence, or criminal activity.
        Analyze the full context. Do not flag jokes, sarcasm, or metaphorical language.

        If you find one or more credible, suspicious conversations, format EACH conversation inside <threat> tags like this:
        <threat>
        <summary>A brief, one-sentence summary of the threat.</summary>
        <severity>High, Medium, or Low</severity>
        <messages>
        Quote the 2-4 most critical messages from that conversation, including who said them. **Highlight the suspicious keywords within the quoted messages using the format `**<keyword>**`**.
        </messages>
        </threat>

        Do not add any introductory or concluding text outside of the <threat> tags.
        If you find no credible threats, respond with the single phrase: "No credible threats found."

        Chat Log:
        ---
        {chat_text}
        ---
        """
        response = model.generate_content(prompt, request_options={'timeout': 120})
        return response.text
    except Exception as e:
        if "quota" in str(e):
            return "An error occurred during AI analysis: You have exceeded your daily quota for the Google AI API."
        return f"An error occurred during AI analysis: {str(e)}"

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

def display_ai_threat_report(report, messages):
    st.markdown('<div style="animation: fadeIn 0.5s;">', unsafe_allow_html=True)
    st.subheader("🤖 AI Context Analysis")
    if "No credible threats found" in report:
        st.success("✅ AI analysis complete. No credible threats were found in the conversation.", icon="🎉")
    else:
        st.error("🚨 AI has identified potentially serious conversations in this chat. Please review carefully.", icon="🔥")
        threats = [threat.strip() for threat in report.split('<threat>') if threat.strip()]
        for threat in threats:
            st.markdown(
                f"""
                <div style="
                    background-color: white;
                    color: black;
                    padding: 1rem;
                    border-radius: 8px;
                    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                    margin-bottom: 1rem;
                    animation: slideIn 0.5s ease-out;
                ">
                    {html_lib.escape(threat)}
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)

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
        
    return pdf.output(dest='S').encode('latin-1')

# -----------------------------------------------------------------------------
# 4. MAIN APP LOGIC
# -----------------------------------------------------------------------------
def main():
    load_css()

    with st.sidebar:
        st.image(get_image_as_base64(SIDEBAR_LOGO_SVG), width=50)
        st.header("🕵️‍♀️ Controls Panel")
        uploaded_file = st.file_uploader("Upload WhatsApp chat export (.txt)", type=["txt"])
        
        with st.expander("🔑 API Configuration"):
            if 'api_key' not in st.session_state: st.session_state.api_key = ''
            api_key_input = st.text_input("Enter Google AI API Key", type="password", value=st.session_state.api_key)
            if api_key_input: st.session_state.api_key = api_key_input
        
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
            ai_threat_report = analyze_chat_for_threats_holistically(user_df, st.session_state.api_key)
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