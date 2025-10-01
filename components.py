"""
Streamlit UI components and styling.
"""
import base64
import streamlit as st
from typing import Dict
from ..config import config


class UIComponents:
    """Handles UI styling and components for Streamlit."""
    
    def __init__(self, theme: str = "dark"):
        self.theme = theme
        self.config = config
    
    def load_css(self):
        """Load custom CSS for the application."""
        theme_vars = (config.theme.DARK_THEME if self.theme == "dark" 
                     else config.theme.LIGHT_THEME)
        
        css_vars = "\n".join([f"{k}: {v};" for k, v in theme_vars.items()])
        
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
                color: var(--text-color);
            }}
            
            .metric-card-value {{
                font-size: 1.2rem;
                font-weight: bold;
                color: var(--text-color);
            }}
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render the main application header."""
        st.markdown("""
        <div class="title-container">
            <div class="main-title">WhatsApp Chat Analysis</div>
            <div class="sub-title">Analyze sentiment, threats, and activity patterns from your chat history.</div>
        </div>
        """, unsafe_allow_html=True)
    
    def metric_card(self, icon: str, label: str, value: str, card_color: str = "var(--card-color)"):
        """Render a metric card component."""
        import html
        st.markdown(f"""
        <div class="metric-card" style="background-color:{card_color}">
            <div class="metric-card-icon">{icon}</div>
            <div class="metric-card-label">{label}</div>
            <div class="metric-card-value">{html.escape(str(value))}</div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_welcome_screen(self):
        """Render the welcome screen when no file is uploaded."""
        st.info("**Welcome!** Please upload a WhatsApp `.txt` file using the sidebar to begin.", icon="👋")
        
        st.markdown("---")
        st.subheader("How It Works")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self.metric_card("📤", "Step 1", "Upload File")
        
        with col2:
            self.metric_card("🔬", "Step 2", "Analyze Data")
        
        with col3:
            self.metric_card("📈", "Step 3", "Explore Insights")
    
    def render_loading_animation(self, message: str = "Processing..."):
        """Render a loading animation with message."""
        st.markdown(f"""
        <div class="loading-animation">
            <div class="pulsate"></div>
            <p>{message}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def get_image_as_base64(self, svg_string: str) -> str:
        """Convert SVG string to base64 data URL."""
        svg_bytes = svg_string.encode('utf-8')
        b64 = base64.b64encode(svg_bytes).decode()
        return f"data:image/svg+xml;base64,{b64}"
    
    def render_sidebar_logo(self):
        """Render sidebar logo."""
        sidebar_logo_svg = '''
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" 
             viewBox="0 0 24 24" fill="none" stroke="currentColor" 
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
            <path d="m9 12 2 2 4-4"></path>
        </svg>
        '''
        st.image(self.get_image_as_base64(sidebar_logo_svg), width=50)
