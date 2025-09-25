"""
Main Streamlit application for WhatsApp Chat Analysis.

This is the entry point for the refactored, industrial-grade chat analyzer.
"""
import streamlit as st
import pandas as pd
import time
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from whatsapp_analyzer import ChatAnalyzer, SentimentAnalyzer, ThreatDetector
from whatsapp_analyzer.config import config
from whatsapp_analyzer.ui.components import UIComponents
from whatsapp_analyzer.ui.charts import ChartGenerator
from whatsapp_analyzer.utils.report_generator import ReportGenerator
from whatsapp_analyzer.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging("INFO")
logger = get_logger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title=config.app.PAGE_TITLE,
    page_icon=config.app.PAGE_ICON,
    layout=config.app.LAYOUT
)


class WhatsAppAnalyzerApp:
    """Main application class for WhatsApp Chat Analysis."""
    
    def __init__(self):
        self.ui = UIComponents()
        self.chart_generator = ChartGenerator()
        self.report_generator = ReportGenerator()
        self.analyzer = None
        
        # Initialize session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'current_file_name' not in st.session_state:
            st.session_state.current_file_name = None
        
        if 'api_key' not in st.session_state:
            st.session_state.api_key = ''
        
        if 'analysis_result' not in st.session_state:
            st.session_state.analysis_result = None
        
        if 'threat_result' not in st.session_state:
            st.session_state.threat_result = None
    
    def run(self):
        """Run the main application."""
        try:
            # Load UI styling
            self.ui.load_css()
            
            # Render sidebar
            self._render_sidebar()
            
            # Render main header
            self.ui.render_header()
            
            # Handle file upload and analysis
            uploaded_file = st.session_state.get('uploaded_file')
            
            if uploaded_file:
                self._handle_file_upload(uploaded_file)
            else:
                self.ui.render_welcome_screen()
            
            # Footer
            st.markdown('<div class="footer">Built with ❤️ using Streamlit & Python</div>', 
                       unsafe_allow_html=True)
                       
        except Exception as e:
            logger.error(f"Application error: {str(e)}")
            st.error(f"An unexpected error occurred: {str(e)}")
    
    def _render_sidebar(self):
        """Render the sidebar with controls."""
        with st.sidebar:
            # Logo
            self.ui.render_sidebar_logo()
            st.header("🕵️‍♀️ Controls Panel")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Upload WhatsApp chat export (.txt)", 
                type=["txt"]
            )
            st.session_state.uploaded_file = uploaded_file
            
            # API Configuration
            with st.expander("🔑 API Configuration"):
                api_key_input = st.text_input(
                    "Enter Google AI API Key", 
                    type="password", 
                    value=st.session_state.api_key,
                    help="Required for AI-powered threat detection"
                )
                if api_key_input:
                    st.session_state.api_key = api_key_input
            
            # Analysis filters (shown only when file is uploaded)
            if uploaded_file and st.session_state.analysis_result:
                self._render_analysis_filters()
            
            # Metrics and report download (shown when analysis is complete)
            if st.session_state.analysis_result:
                self._render_sidebar_metrics()
    
    def _render_analysis_filters(self):
        """Render analysis filters in sidebar."""
        st.markdown("---")
        st.header("🔍 Filters")
        
        # User filter
        analysis_result = st.session_state.analysis_result
        user_names = [user.username for user in analysis_result.user_metrics]
        
        selected_users = st.multiselect(
            "Filter by Author(s)", 
            options=user_names,
            help="Select specific users to analyze"
        )
        
        # Date range filter
        if analysis_result.date_range:
            min_date = analysis_result.date_range[0].date()
            max_date = analysis_result.date_range[1].date()
            
            st.markdown("##### Filter by Date Range")
            start_date = st.date_input(
                "Start date", 
                min_date, 
                min_value=min_date, 
                max_value=max_date
            )
            end_date = st.date_input(
                "End date", 
                max_date, 
                min_value=start_date, 
                max_value=max_date
            )
            
            # Store filter values in session state
            st.session_state.selected_users = selected_users
            st.session_state.date_filter = (start_date, end_date)
    
    def _render_sidebar_metrics(self):
        """Render key metrics in sidebar."""
        st.markdown("---")
        st.header("📊 Chat Metrics & Report")
        
        analysis_result = st.session_state.analysis_result
        threat_result = st.session_state.threat_result
        
        if analysis_result:
            # Key metrics
            self.ui.metric_card("💬", "Messages", f"{analysis_result.total_messages:,}")
            
            threat_count = threat_result.threats_found if threat_result else 0
            self.ui.metric_card("🚨", "Threats", threat_count)
            
            self.ui.metric_card("👑", "Top User", analysis_result.most_active_user)
            
            sentiment_score = f"{analysis_result.overall_sentiment:.2f}"
            self.ui.metric_card("📈", "Sentiment", sentiment_score)
            
            # PDF Report download
            st.markdown("---")
            if st.button("📄 Generate PDF Report"):
                self._generate_pdf_report()
    
    def _handle_file_upload(self, uploaded_file):
        """Handle file upload and analysis."""
        # Check if this is a new file
        if uploaded_file.name != st.session_state.current_file_name:
            st.cache_data.clear()
            st.session_state.current_file_name = uploaded_file.name
            st.session_state.analysis_result = None
            st.session_state.threat_result = None
            
            st.success(f"New file detected: '{uploaded_file.name}'. Running fresh analysis...")
            time.sleep(1)
        
        # Perform analysis if not already done
        if not st.session_state.analysis_result:
            self._perform_analysis(uploaded_file)
        
        # Display results
        if st.session_state.analysis_result:
            self._display_analysis_results()
    
    def _perform_analysis(self, uploaded_file):
        """Perform the chat analysis."""
        try:
            with st.spinner("Analyzing chat data..."):
                # Read file content
                raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
                logger.info(f"Processing file: {uploaded_file.name}")
                
                # Initialize analyzer
                self.analyzer = ChatAnalyzer(api_key=st.session_state.api_key)
                
                # Perform main analysis
                analysis_result = self.analyzer.analyze_chat(raw_text)
                st.session_state.analysis_result = analysis_result
                
                logger.info("Main analysis completed successfully")
            
            # Perform threat analysis
            if st.session_state.api_key:
                with st.spinner("Performing AI threat analysis..."):
                    # Parse the data again for threat analysis
                    df = self.analyzer.parser.parse_chat_export(raw_text)
                    df = self.analyzer._preprocess_dataframe(df)
                    
                    threat_result = self.analyzer.analyze_threats(df)
                    st.session_state.threat_result = threat_result
                    
                    logger.info("Threat analysis completed successfully")
            else:
                st.session_state.threat_result = None
                st.warning("API key not provided. Threat analysis will be skipped.")
                
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            st.error(f"Analysis failed: {str(e)}")
            st.session_state.analysis_result = None
            st.session_state.threat_result = None
    
    def _display_analysis_results(self):
        """Display the analysis results in tabs."""
        analysis_result = st.session_state.analysis_result
        threat_result = st.session_state.threat_result
        
        if not analysis_result:
            return
        
        # Display key insights
        self._display_key_insights(analysis_result)
        
        # Create tabs for different views
        tab_names = [
            "🤖 AI Threat Analysis" if threat_result else "🤖 Threat Analysis (No API)",
            "📊 Dashboard", 
            "☁️ Word Cloud", 
            "👤 User Deep Dive", 
            "📈 Advanced Analytics"
        ]
        
        tabs = st.tabs(tab_names)
        
        with tabs[0]:
            self._display_threat_analysis(threat_result)
        
        with tabs[1]:
            self._display_dashboard(analysis_result)
        
        with tabs[2]:
            self._display_word_cloud(analysis_result)
        
        with tabs[3]:
            self._display_user_deep_dive(analysis_result)
        
        with tabs[4]:
            self._display_advanced_analytics(analysis_result)
    
    def _display_key_insights(self, analysis_result):
        """Display key insights from the analysis."""
        st.markdown("### 💡 Key Insights")
        
        sentiment_text = analysis_result.sentiment_label.value
        sentiment_color = {
            "Positive": "var(--success-color)",
            "Negative": "var(--danger-color)",
            "Neutral": "var(--warning-color)"
        }.get(sentiment_text, "var(--text-color)")
        
        insights = f"""
        - **Overall Sentiment**: The general mood is <span style='color:{sentiment_color}; font-weight:bold;'>{sentiment_text}</span> (Score: {analysis_result.overall_sentiment:.3f})
        - **Most Active User**: **{analysis_result.most_active_user}** leads the conversation
        - **Peak Activity**: Most messages sent around **{analysis_result.peak_activity_hour}:00**
        - **Duration**: Chat spans **{(analysis_result.date_range[1] - analysis_result.date_range[0]).days}** days
        """
        
        st.markdown(insights, unsafe_allow_html=True)
        st.markdown("---")
    
    def _display_threat_analysis(self, threat_result):
        """Display threat analysis results."""
        if not threat_result:
            st.warning("⚠️ Threat analysis requires a Google AI API key. Please add your key in the sidebar.")
            return
        
        st.subheader("🤖 AI-Powered Threat Assessment")
        
        if threat_result.threats_found == 0:
            st.success("✅ No credible threats detected in the conversation.", icon="🎉")
        elif threat_result.threats_found == -1:
            st.error(f"❌ {threat_result.analysis_summary}")
        else:
            st.error(f"🚨 {threat_result.threats_found} potential threat(s) detected.")
            
            # Display threat details
            for i, threat in enumerate(threat_result.threat_details, 1):
                with st.expander(f"Threat {i}: {threat.get('severity', 'Unknown')} Severity"):
                    st.write(f"**Summary:** {threat.get('summary', 'No summary')}")
                    st.write(f"**Evidence:**")
                    st.code(threat.get('messages', 'No messages'), language="text")
    
    def _display_dashboard(self, analysis_result):
        """Display dashboard with charts."""
        st.subheader("📊 User & Message Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create dummy DataFrame for chart generation
            user_data = [(user.username, user.message_count) for user in analysis_result.user_metrics]
            user_df = pd.DataFrame(user_data, columns=['author', 'count'])
            user_df = pd.concat([user_df] * user_df['count'].iloc[0])  # Expand for pie chart
            
            fig_users = self.chart_generator.create_user_pie_chart(user_df)
            if fig_users:
                st.plotly_chart(fig_users, use_container_width=True)
        
        with col2:
            fig_types = self.chart_generator.create_message_type_pie_chart(
                analysis_result.message_types_distribution
            )
            if fig_types:
                st.plotly_chart(fig_types, use_container_width=True)
        
        # Additional charts
        st.markdown("---")
        col3, col4 = st.columns(2)
        
        with col3:
            # User sentiment comparison
            fig_sentiment = self.chart_generator.create_user_sentiment_comparison(
                analysis_result.user_metrics
            )
            if fig_sentiment:
                st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col4:
            # Interaction heatmap
            if analysis_result.interaction_matrix is not None:
                fig_heatmap = self.chart_generator.create_interaction_heatmap(
                    analysis_result.interaction_matrix
                )
                if fig_heatmap:
                    st.pyplot(fig_heatmap, use_container_width=True)
    
    def _display_word_cloud(self, analysis_result):
        """Display word cloud visualization."""
        st.subheader("☁️ Most Frequent Words")
        
        # Combine all messages for word cloud
        all_messages = []
        for user in analysis_result.user_metrics:
            # This is a simplified approach - in real implementation,
            # we'd need access to the original messages
            all_messages.extend([user.username] * user.message_count)
        
        all_text = " ".join(all_messages)
        
        if all_text.strip():
            fig_wordcloud = self.chart_generator.create_word_cloud(all_text)
            if fig_wordcloud:
                st.pyplot(fig_wordcloud, use_container_width=True)
        else:
            st.info("No text data available for word cloud generation.")
    
    def _display_user_deep_dive(self, analysis_result):
        """Display detailed user analysis."""
        st.subheader("👤 User Deep Dive")
        
        # User selection
        usernames = [user.username for user in analysis_result.user_metrics]
        selected_user = st.selectbox("Select a user to analyze", options=usernames)
        
        if selected_user:
            # Find user metrics
            user_metrics = next(
                (user for user in analysis_result.user_metrics if user.username == selected_user), 
                None
            )
            
            if user_metrics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    self.ui.metric_card("💬", "Messages", user_metrics.message_count)
                
                with col2:
                    sentiment_label = SentimentAnalyzer().get_sentiment_label(user_metrics.avg_sentiment)
                    self.ui.metric_card("😊", "Sentiment", sentiment_label.value)
                
                with col3:
                    self.ui.metric_card("⏰", "Active Hour", f"{user_metrics.most_active_hour}:00")
                
                with col4:
                    emoji_count = len(user_metrics.emoji_usage)
                    self.ui.metric_card("😀", "Emojis Used", emoji_count)
                
                # Detailed metrics
                st.markdown("#### Message Types")
                for msg_type, count in user_metrics.message_types.items():
                    st.write(f"• {msg_type.value}: {count}")
                
                if user_metrics.emoji_usage:
                    st.markdown("#### Top Emojis")
                    top_emojis = sorted(
                        user_metrics.emoji_usage.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:10]
                    
                    for emoji, count in top_emojis:
                        st.write(f"{emoji} × {count}")
    
    def _display_advanced_analytics(self, analysis_result):
        """Display advanced analytics."""
        st.subheader("📈 Advanced Analytics")
        
        # Topics
        if analysis_result.topics:
            st.markdown("#### 🎯 Key Topics Discussed")
            for topic in analysis_result.topics:
                st.write(f"• {topic}")
        else:
            st.info("Not enough data for topic modeling (minimum 10 messages required)")
        
        # Statistics
        st.markdown("#### 📊 Conversation Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Users", analysis_result.total_users)
            st.metric("Total Messages", analysis_result.total_messages)
            
        with col2:
            duration = (analysis_result.date_range[1] - analysis_result.date_range[0]).days
            st.metric("Duration (Days)", duration)
            
            if duration > 0:
                avg_messages_per_day = analysis_result.total_messages / duration
                st.metric("Avg Messages/Day", f"{avg_messages_per_day:.1f}")
    
    def _generate_pdf_report(self):
        """Generate and provide PDF report download."""
        try:
            with st.spinner("Generating PDF report..."):
                analysis_result = st.session_state.analysis_result
                threat_result = st.session_state.threat_result or None
                
                pdf_bytes = self.report_generator.generate_comprehensive_report(
                    analysis_result, threat_result
                )
                
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"whatsapp_analysis_{int(time.time())}.pdf",
                    mime="application/pdf"
                )
                
                st.success("Report generated successfully!")
                
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            st.error(f"Failed to generate report: {str(e)}")


def main():
    """Main entry point."""
    app = WhatsAppAnalyzerApp()
    app.run()


if __name__ == "__main__":
    main()
