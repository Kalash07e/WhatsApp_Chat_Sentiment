# �️ WhatsApp Chat Sentiment Analysis | AI-Powered Analytics

> **Advanced AI-powered sentiment analysis and threat detection for WhatsApp conversations**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-brightgreen)](https://rvjr7jwntmp4t.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Source%20Code-blue)](https://github.com/Kalash07e/WhatsApp_Chat_Sentiment)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)](https://streamlit.io)

## 🌟 Live Application

**🚀 [Try the App Now](https://rvjr7jwntmp4t.streamlit.app)** - Analyze your WhatsApp chats instantly!

## 📋 Overview

Transform your WhatsApp conversations into powerful insights with our AI-driven analysis platform. Get detailed sentiment analysis, emotion detection, threat assessment, and comprehensive reports from your chat history.

### 🎯 Use Cases

- **Personal Insights** - Understand communication patterns in your relationships
- **Group Dynamics** - Analyze team or family group conversations
- **Content Moderation** - Detect inappropriate content or threats
- **Research & Analysis** - Academic or professional communication studies
- **Relationship Counseling** - Therapists analyzing client communication patterns
- **Security Assessment** - Organizations monitoring internal communications

## ✨ Features

### ✨ Key Features

- **🤖 AI-Powered Sentiment Analysis** - Advanced emotion detection using VADER and TextBlob
- **🛡️ Intelligent Threat Detection** - Google Gemini AI identifies potential security risks
- **😊 Emoji Pattern Analysis** - Comprehensive emoji usage and sentiment correlation
- **👥 User Comparison Dashboard** - Side-by-side participant analysis
- **📊 Interactive Visualizations** - Beautiful charts and graphs powered by Plotly
- **📱 Mobile-Friendly** - Optimized for mobile upload and analysis
- **📋 PDF Report Generation** - Professional, customizable reports
- **🔐 Secure Authentication** - User accounts with admin panel

## 🚀 Quick Start

### 1. Access the App
Visit [https://rvjr7jwntmp4t.streamlit.app](https://rvjr7jwntmp4t.streamlit.app)

### 2. Export Your WhatsApp Chat
- Open WhatsApp → Select Chat → Menu (⋮) → Export Chat → Without Media
- Save as `.txt` file

### 3. Upload & Analyze
- Create account or login
- Upload your `.txt` file
- Get instant AI-powered insights!

## 📊 Analysis Features

### Sentiment Analysis Dashboard
- **Message Sentiment Trends** - Track emotional patterns over time
- **User Sentiment Profiles** - Individual emotional characteristics
- **Sentiment Distribution** - Overall conversation mood analysis
- **Peak Activity Times** - When conversations are most active

### Emoji Analysis Dashboard
- **Emoji Frequency Rankings** - Most used emojis and their meanings
- **Sentiment-Emoji Correlation** - How emojis relate to emotions
- **User Emoji Patterns** - Individual emoji usage styles
- **Emoji Evolution** - How emoji usage changes over time

### AI Threat Detection
- **Context-Aware Analysis** - Understanding conversation nuances
- **Psychological Profiling** - Behavioral pattern recognition
- **Risk Assessment** - Credible threat identification
- **Safety Recommendations** - Actionable security advice

### User Comparison Tools
- **Side-by-Side Analysis** - Compare participant behavior
- **Activity Patterns** - Message frequency and timing
- **Communication Styles** - Language and tone analysis
- **Relationship Dynamics** - Interaction pattern insights

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Python web framework)
- **AI/ML**: Google Gemini API, VADER Sentiment, TextBlob
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy, scikit-learn
- **Authentication**: Custom secure user system
- **PDF Generation**: FPDF with custom styling
- **Deployment**: Streamlit Community Cloud

## 🔒 Privacy & Security

- **Local Processing**: Most analysis happens on secure servers
- **No Data Storage**: Chat content is not permanently stored
- **User Authentication**: Secure account system
- **API Security**: Encrypted AI service communications
- **GDPR Compliant**: Privacy-first design

## 📈 SEO Keywords

WhatsApp chat analysis, sentiment analysis tool, AI emotion detection, conversation analytics, chat sentiment analyzer, WhatsApp data analysis, social media analytics, text sentiment analysis, conversation insights, chat emotion tracker, WhatsApp sentiment tool, AI chat analysis, conversation sentiment analysis, chat analytics platform, emotion detection software, social conversation analysis

## 🌐 Links & Resources

- **🚀 [Live Application](https://rvjr7jwntmp4t.streamlit.app)**
- **📂 [Source Code](https://github.com/Kalash07e/WhatsApp_Chat_Sentiment)**
- **📖 [Documentation](https://github.com/Kalash07e/WhatsApp_Chat_Sentiment/blob/main/docs/)**
- **🐛 [Report Issues](https://github.com/Kalash07e/WhatsApp_Chat_Sentiment/issues)**

## 👨‍� Author

**Kalash Bhargava**
- 🌐 [Portfolio](https://github.com/Kalash07e)
- 📧 [Contact](mailto:kalashbhargava017@gmail.com)
- 💼 [LinkedIn](https://linkedin.com/in/kalash-bhargava)

### 🔐 Authentication System
- **🔐 Authentication System**

### 📊 **Chat Analysis**
- **Sentiment Analysis**: VADER + TextBlob hybrid approach
- **Threat Detection**: AI-powered security analysis with Google Gemini
- **User Activity**: Message frequency and participation metrics
- **Emoji Analysis**: Emoji usage patterns and sentiment correlation
- **Topic Modeling**: Automatic topic discovery with LDA
- **Temporal Analysis**: Time-based conversation patterns

### 📈 **Visualizations**
- Interactive sentiment timeline charts
- User activity pie charts
- Message type distribution
- Conversation health metrics
- Word clouds with sentiment coloring
- Interaction heatmaps

### 📄 **Export & Reporting**
- PDF report generation
- CSV data export
- Professional sentiment dashboards
- Detailed threat assessment reports

## 🛠️ Installation & Setup

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/WhatsApp_Chat_Sentiment.git
   cd WhatsApp_Chat_Sentiment
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure secrets** (Optional)
   ```bash
   cp .streamlit/secrets_template.toml .streamlit/secrets.toml
   # Edit .streamlit/secrets.toml with your API keys
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Access the app**
   - Open your browser and navigate to `http://localhost:8501`

## 🌐 Deployment to Streamlit Community Cloud

### Prerequisites
- GitHub account
- Streamlit Community Cloud account
- Your repository pushed to GitHub

### Deployment Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository: `WhatsApp_Chat_Sentiment`
   - Main file path: `app.py`
   - Click "Deploy!"

3. **Configure Secrets** (Optional)
   - In your deployed app, go to "Settings" → "Secrets"
   - Add your secrets in TOML format:
     ```toml
     GEMINI_API_KEY = "your_api_key_here"
     ```

## 📱 How to Use

### 1. **Export WhatsApp Chat**
- Open WhatsApp on your phone
- Go to the chat you want to analyze
- Tap on chat name → More → Export Chat
- Choose "Without Media" for faster processing
- Save the exported `.txt` file

### 2. **Create Account**
- Visit the deployed application
- Click "Register" tab
- Fill in your details with a valid email
- Accept terms and create account

### 3. **Analyze Your Chat**
- Sign in to your account
- Upload your WhatsApp export file
- Wait for processing (may take a few moments)
- Explore the interactive analysis results

### 4. **View Results**
- **Sentiment Dashboard**: Overall conversation mood and trends
- **User Activity**: Who's most active in the chat
- **Threat Analysis**: AI-powered security assessment (requires API key)
- **Export Options**: Download reports and data

## 🔧 Configuration

### Environment Variables
Create `.streamlit/secrets.toml` for local development:

```toml
# Google Gemini AI API Key (optional)
GEMINI_API_KEY = "your_gemini_api_key_here"

# Admin Email Addresses (optional)
ADMIN_EMAILS = ["your-email@example.com"]
```

### Admin Panel Access
- Configure admin emails in `auth.py` or via environment variables
- Admin users can view all registered users and system statistics
- Access admin panel through the sidebar (admin users only)

## 🔒 Security & Privacy

- **Password Security**: All passwords are hashed using SHA-256
- **Data Privacy**: Chat data is processed locally and not stored permanently
- **Email Validation**: Prevents fake email registrations
- **Session Management**: Secure user session handling
- **Admin Controls**: Restricted admin panel access

## 📊 Technical Details

### Architecture
- **Frontend**: Streamlit with custom CSS styling
- **Backend**: Python with pandas for data processing
- **Authentication**: Custom auth system with JSON storage
- **AI Integration**: Google Gemini API for threat detection
- **Visualization**: Plotly, Matplotlib, Seaborn

### Key Dependencies
- `streamlit` - Web framework
- `pandas` - Data manipulation
- `plotly` - Interactive visualizations
- `textblob` - Sentiment analysis
- `vaderSentiment` - Emotion detection
- `google-generativeai` - AI threat analysis
- `fpdf2` - PDF report generation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Common Issues

1. **Large Files**: WhatsApp exports with media can be very large. Use "Without Media" option.

2. **API Limits**: Google Gemini API has usage limits. Monitor your usage in Google Cloud Console.

3. **Memory Issues**: For very large chats (>10,000 messages), processing may be slow.

### Getting Help

- **Issues**: Report bugs or request features via GitHub Issues
- **Documentation**: Check this README for setup and usage instructions
- **Community**: Join discussions in GitHub Discussions

## 🙏 Acknowledgments

- **Streamlit** for the amazing web framework
- **Google Gemini** for AI-powered threat detection
- **VADER Sentiment** for emotion analysis
- **TextBlob** for natural language processing
- **Plotly** for interactive visualizations

## 📊 Project Statistics

- **Language**: Python 🐍
- **Framework**: Streamlit 🚀
- **Features**: 15+ analysis tools
- **Visualizations**: 10+ interactive charts
- **Security**: Enterprise-grade authentication

---

**⭐ Star this repository if you find it useful!**

*Built with ❤️ using Python, Streamlit, and AI*

[![Deploy on Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)