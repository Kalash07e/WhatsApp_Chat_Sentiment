# 📊 WhatsApp Chat Analysis Platform

A professional web application for analyzing WhatsApp chat exports with advanced sentiment analysis, threat detection, and interactive visualizations.

## 🚀 Live Demo

**Deployed on Streamlit Community Cloud:** [Your App URL will be here]

## ✨ Features

### 🔐 **Authentication System**
- Secure user registration and login
- Password hashing with SHA-256
- Email validation and domain verification
- Admin panel for user management
- Session management

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

**Made with ❤️ for better communication analysis**

[![Deploy on Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)