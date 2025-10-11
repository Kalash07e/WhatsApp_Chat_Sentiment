"""
AI-powered threat detection for chat analysis.
"""
import re
from typing import List, Dict, Any, Optional
from ..models import ThreatSeverity, ThreatAnalysisResult
from ..config import config

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False


class ThreatDetector:
    """Handles AI-powered threat detection in chat conversations."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.GOOGLE_AI_API_KEY
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the AI model if API key is available."""
        if not GENAI_AVAILABLE:
            raise ImportError("google.generativeai is not available. Install it with: pip install google-generativeai")
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash')
            except Exception as e:
                raise RuntimeError(f"Failed to initialize AI model: {str(e)}")
    
    def analyze_threats(self, messages_df) -> ThreatAnalysisResult:
        """
        Analyze chat messages for potential threats using AI.
        
        Args:
            messages_df: DataFrame containing chat messages
            
        Returns:
            ThreatAnalysisResult object
        """
        if not self.model:
            return self._create_empty_result("AI model not available or API key not provided")
        
        try:
            # Prepare chat text for analysis
            chat_text = self._prepare_chat_text(messages_df)
            
            # Generate AI analysis
            analysis_text = self._generate_threat_analysis(chat_text)
            
            # Parse the analysis results
            return self._parse_analysis_results(analysis_text)
            
        except Exception as e:
            return self._create_error_result(f"Analysis failed: {str(e)}")
    
    def _prepare_chat_text(self, messages_df) -> str:
        """Prepare chat text for AI analysis."""
        chat_text = "\n".join([
            f"{row['datetime']} - {row['author']}: {row['message']}" 
            for _, row in messages_df.iterrows()
        ])
        
        # Truncate if too long
        max_chars = config.app.AI_MAX_CHARS
        if len(chat_text) > max_chars:
            chat_text = chat_text[-max_chars:]
        
        return chat_text
    
    def _generate_threat_analysis(self, chat_text: str) -> str:
        """Generate threat analysis using AI model."""
        prompt = self._create_analysis_prompt(chat_text)
        
        try:
            response = self.model.generate_content(
                prompt, 
                request_options={'timeout': config.app.AI_REQUEST_TIMEOUT}
            )
            return response.text
        except Exception as e:
            if "quota" in str(e).lower():
                raise RuntimeError("API quota exceeded. Please check your Google AI API limits.")
            raise RuntimeError(f"AI analysis failed: {str(e)}")
    
    def _create_analysis_prompt(self, chat_text: str) -> str:
        """Create the analysis prompt for the AI model."""
        return f"""
        You are a security analyst. Review this entire chat log for any conversations that discuss 
        planning or carrying out harmful, violent, or illegal acts like terrorism, violence, or criminal activity.
        
        Analyze the full context. Do not flag jokes, sarcasm, or metaphorical language.
        
        If you find one or more credible, suspicious conversations, format EACH conversation inside <threat> tags like this:
        <threat>
        <summary>A brief, one-sentence summary of the threat.</summary>
        <severity>High, Medium, or Low</severity>
        <messages>
        Quote the 2-4 most critical messages from that conversation, including who said them. 
        **Highlight the suspicious keywords within the quoted messages using the format `**<keyword>**`**.
        </messages>
        </threat>

        Do not add any introductory or concluding text outside of the <threat> tags.
        If you find no credible threats, respond with the single phrase: "No credible threats found."

        Chat Log:
        ---
        {chat_text}
        ---
        """
    
    def _parse_analysis_results(self, analysis_text: str) -> ThreatAnalysisResult:
        """Parse AI analysis results into structured format."""
        if "No credible threats found" in analysis_text:
            return ThreatAnalysisResult(
                threats_found=0,
                severity_distribution={},
                threat_details=[],
                analysis_summary="No credible threats detected in the conversation."
            )
        
        # Parse threat blocks
        threat_blocks = [
            block.strip() 
            for block in analysis_text.split('<threat>') 
            if block.strip()
        ]
        
        threat_details = []
        severity_counts = {severity: 0 for severity in ThreatSeverity}
        
        for block in threat_blocks:
            threat_detail = self._parse_threat_block(block)
            if threat_detail:
                threat_details.append(threat_detail)
                severity = ThreatSeverity(threat_detail.get('severity', 'Low'))
                severity_counts[severity] += 1
        
        return ThreatAnalysisResult(
            threats_found=len(threat_details),
            severity_distribution=severity_counts,
            threat_details=threat_details,
            analysis_summary=f"Found {len(threat_details)} potential threat(s) in the conversation."
        )
    
    def _parse_threat_block(self, block: str) -> Optional[Dict[str, Any]]:
        """Parse individual threat block."""
        try:
            # Extract summary
            summary_match = re.search(r'<summary>(.*?)</summary>', block, re.DOTALL)
            summary = summary_match.group(1).strip() if summary_match else "Unknown threat"
            
            # Extract severity
            severity_match = re.search(r'<severity>(.*?)</severity>', block, re.DOTALL)
            severity = severity_match.group(1).strip() if severity_match else "Low"
            
            # Extract messages
            messages_match = re.search(r'<messages>(.*?)</messages>', block, re.DOTALL)
            messages = messages_match.group(1).strip() if messages_match else "No messages found"
            
            return {
                'summary': summary,
                'severity': severity,
                'messages': messages,
                'raw_block': block
            }
        except Exception:
            return None
    
    def _create_empty_result(self, message: str) -> ThreatAnalysisResult:
        """Create empty result with message."""
        return ThreatAnalysisResult(
            threats_found=0,
            severity_distribution={},
            threat_details=[],
            analysis_summary=message
        )
    
    def _create_error_result(self, error_message: str) -> ThreatAnalysisResult:
        """Create error result."""
        return ThreatAnalysisResult(
            threats_found=-1,
            severity_distribution={},
            threat_details=[],
            analysis_summary=f"Error during analysis: {error_message}"
        )
