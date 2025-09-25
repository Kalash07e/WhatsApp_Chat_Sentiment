"""
PDF report generation utilities.
"""
from fpdf import FPDF
from typing import Dict, Any
import pandas as pd
from datetime import datetime
from ..models import ChatAnalysisResult, ThreatAnalysisResult


class ReportGenerator:
    """Generates PDF reports for chat analysis."""
    
    def __init__(self):
        self.pdf = None
    
    def generate_comprehensive_report(
        self, 
        analysis_result: ChatAnalysisResult,
        threat_result: ThreatAnalysisResult,
        filename: str = "whatsapp_analysis_report.pdf"
    ) -> bytes:
        """
        Generate a comprehensive PDF report.
        
        Args:
            analysis_result: Complete chat analysis results
            threat_result: Threat analysis results
            filename: Output filename
            
        Returns:
            PDF content as bytes
        """
        self.pdf = FPDF()
        
        # Add title page
        self._add_title_page()
        
        # Add executive summary
        self._add_executive_summary(analysis_result, threat_result)
        
        # Add detailed analysis
        self._add_detailed_analysis(analysis_result)
        
        # Add threat assessment
        self._add_threat_assessment(threat_result)
        
        # Add user profiles
        self._add_user_profiles(analysis_result.user_metrics)
        
        # Add appendices
        self._add_appendices(analysis_result)
        
        return self.pdf.output(dest='S').encode('latin-1')
    
    def _add_title_page(self):
        """Add title page to the report."""
        self.pdf.add_page()
        self.pdf.set_font("Arial", 'B', 24)
        
        # Center the title
        self.pdf.cell(0, 60, '', 0, 1)  # Spacer
        self.pdf.cell(0, 15, 'WhatsApp Chat Analysis Report', 0, 1, 'C')
        
        # Add generation date
        self.pdf.set_font("Arial", '', 12)
        self.pdf.cell(0, 20, '', 0, 1)  # Spacer
        self.pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%B %d, %Y")}', 0, 1, 'C')
        
        # Add disclaimer
        self.pdf.cell(0, 40, '', 0, 1)  # Spacer
        self.pdf.set_font("Arial", 'I', 10)
        disclaimer = ("This report contains analysis of WhatsApp chat data. "
                     "All analysis is automated and should be reviewed by qualified personnel.")
        self.pdf.multi_cell(0, 5, disclaimer, 0, 'C')
    
    def _add_executive_summary(self, analysis_result: ChatAnalysisResult, threat_result: ThreatAnalysisResult):
        """Add executive summary section."""
        self.pdf.add_page()
        self.pdf.set_font("Arial", 'B', 18)
        self.pdf.cell(0, 15, 'Executive Summary', 0, 1)
        self.pdf.ln(5)
        
        self.pdf.set_font("Arial", '', 12)
        
        # Key metrics
        summary_text = f"""
Chat Overview:
• Total Messages: {analysis_result.total_messages:,}
• Unique Users: {analysis_result.total_users}
• Date Range: {analysis_result.date_range[0].strftime('%B %d, %Y')} to {analysis_result.date_range[1].strftime('%B %d, %Y')}
• Duration: {(analysis_result.date_range[1] - analysis_result.date_range[0]).days} days

Sentiment Analysis:
• Overall Sentiment: {analysis_result.sentiment_label.value}
• Sentiment Score: {analysis_result.overall_sentiment:.3f} (Range: -1 to +1)
• Most Active User: {analysis_result.most_active_user}
• Peak Activity Hour: {analysis_result.peak_activity_hour}:00

Security Assessment:
• Threats Detected: {threat_result.threats_found}
• Analysis Status: {threat_result.analysis_summary}
        """.strip()
        
        self.pdf.multi_cell(0, 6, self._clean_text(summary_text))
    
    def _add_detailed_analysis(self, analysis_result: ChatAnalysisResult):
        """Add detailed analysis section."""
        self.pdf.add_page()
        self.pdf.set_font("Arial", 'B', 16)
        self.pdf.cell(0, 12, 'Detailed Analysis', 0, 1)
        self.pdf.ln(3)
        
        # Message type distribution
        self.pdf.set_font("Arial", 'B', 14)
        self.pdf.cell(0, 10, 'Message Type Distribution', 0, 1)
        self.pdf.set_font("Arial", '', 11)
        
        for msg_type, count in analysis_result.message_types_distribution.items():
            percentage = (count / analysis_result.total_messages) * 100
            self.pdf.cell(0, 6, f'• {msg_type.value}: {count} ({percentage:.1f}%)', 0, 1)
        
        self.pdf.ln(5)
        
        # Topics (if available)
        if analysis_result.topics:
            self.pdf.set_font("Arial", 'B', 14)
            self.pdf.cell(0, 10, 'Key Topics Discussed', 0, 1)
            self.pdf.set_font("Arial", '', 11)
            
            for topic in analysis_result.topics:
                self.pdf.multi_cell(0, 6, f'• {self._clean_text(topic)}')
            
            self.pdf.ln(3)
    
    def _add_threat_assessment(self, threat_result: ThreatAnalysisResult):
        """Add threat assessment section."""
        self.pdf.add_page()
        self.pdf.set_font("Arial", 'B', 16)
        self.pdf.cell(0, 12, 'Security Threat Assessment', 0, 1)
        self.pdf.ln(3)
        
        if threat_result.threats_found == 0:
            self.pdf.set_font("Arial", '', 12)
            self.pdf.cell(0, 8, '✓ No credible threats detected in the conversation.', 0, 1)
        else:
            # Severity distribution
            if threat_result.severity_distribution:
                self.pdf.set_font("Arial", 'B', 14)
                self.pdf.cell(0, 10, 'Threat Severity Distribution', 0, 1)
                self.pdf.set_font("Arial", '', 11)
                
                for severity, count in threat_result.severity_distribution.items():
                    if count > 0:
                        self.pdf.cell(0, 6, f'• {severity.value}: {count}', 0, 1)
                
                self.pdf.ln(5)
            
            # Individual threats
            self.pdf.set_font("Arial", 'B', 14)
            self.pdf.cell(0, 10, 'Detected Threats', 0, 1)
            
            for i, threat in enumerate(threat_result.threat_details, 1):
                self.pdf.set_font("Arial", 'B', 12)
                self.pdf.cell(0, 8, f'Threat {i}: {threat.get("severity", "Unknown")} Severity', 0, 1)
                
                self.pdf.set_font("Arial", '', 10)
                summary = threat.get('summary', 'No summary available')
                self.pdf.multi_cell(0, 5, f'Summary: {self._clean_text(summary)}')
                
                messages = threat.get('messages', 'No messages available')
                self.pdf.multi_cell(0, 5, f'Evidence: {self._clean_text(messages)}')
                self.pdf.ln(3)
    
    def _add_user_profiles(self, user_metrics: list):
        """Add user profile section."""
        self.pdf.add_page()
        self.pdf.set_font("Arial", 'B', 16)
        self.pdf.cell(0, 12, 'User Profiles', 0, 1)
        self.pdf.ln(3)
        
        # Top 10 users
        top_users = user_metrics[:10]
        
        for user in top_users:
            self.pdf.set_font("Arial", 'B', 14)
            self.pdf.cell(0, 10, f'{user.username}', 0, 1)
            
            self.pdf.set_font("Arial", '', 11)
            
            user_info = f"""
• Messages: {user.message_count}
• Average Sentiment: {user.avg_sentiment:.3f}
• Most Active Hour: {user.most_active_hour}:00
• Top Emojis: {', '.join(list(user.emoji_usage.keys())[:5]) if user.emoji_usage else 'None'}
            """.strip()
            
            self.pdf.multi_cell(0, 5, self._clean_text(user_info))
            self.pdf.ln(3)
    
    def _add_appendices(self, analysis_result: ChatAnalysisResult):
        """Add appendices with technical details."""
        self.pdf.add_page()
        self.pdf.set_font("Arial", 'B', 16)
        self.pdf.cell(0, 12, 'Appendices', 0, 1)
        self.pdf.ln(3)
        
        # Technical specifications
        self.pdf.set_font("Arial", 'B', 14)
        self.pdf.cell(0, 10, 'A. Technical Specifications', 0, 1)
        self.pdf.set_font("Arial", '', 10)
        
        tech_info = """
Analysis Methods:
• Sentiment Analysis: VADER + TextBlob hybrid approach
• Threat Detection: AI-powered analysis using Google Gemini
• Topic Modeling: Latent Dirichlet Allocation (LDA)
• Interaction Analysis: Temporal pattern recognition

Data Processing:
• Date Range Parsing: Automatic format detection
• Language Support: English and Hinglish
• Emoji Analysis: Unicode pattern matching
• Message Classification: Rule-based categorization
        """.strip()
        
        self.pdf.multi_cell(0, 4, self._clean_text(tech_info))
    
    def _clean_text(self, text: str) -> str:
        """Clean text for PDF compatibility."""
        # Remove problematic characters and encode for latin-1
        try:
            return text.encode('latin-1', 'ignore').decode('latin-1')
        except:
            # Fallback: remove non-ASCII characters
            return ''.join(char for char in text if ord(char) < 128)
