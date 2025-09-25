# 🕵️ WhatsApp Chat Sentiment Analysis

A comprehensive, professional-grade web application for analyzing WhatsApp chat exports with advanced sentiment analysis, AI-powered threat detection, and detailed conversation analytics.

![WhatsApp Chat Analysis](https://img.shields.io/badge/WhatsApp-Chat%20Analysis-25D366?style=for-the-badge&logo=whatsapp&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![AI Powered](https://img.shields.io/badge/AI-Powered-FF6B6B?style=for-the-badge&logo=robot&logoColor=white)

## ✨ Features

### 🎯 **Professional Sentiment Analysis Dashboard**
- **Actual Sentiment Scores**: Displays numerical sentiment values (+0.245, -0.182, etc.)
- **Beautiful Visual Scale**: Professional gradient cards with progress bars
- **Dynamic Highlighting**: Dominant sentiment categories are emphasized
- **Advanced Metrics**: Positivity ratio, emotional range, and comprehensive interpretations

### 🤖 **AI-Powered Threat Detection**
- **Enhanced Context Analysis**: Advanced psychological profiling using Google Gemini AI
- **Security Assessment**: Real-time threat detection with risk stratification
- **Professional Reports**: Clear, actionable threat analysis with evidence summaries
- **Background Processing**: Secure API key management (no user input required)

### 📊 **Comprehensive Analytics**
- **Interactive Visualizations**: Plotly charts, word clouds, and interaction heatmaps
- **User Deep Dive**: Individual participant analysis and behavior patterns
- **Conversation Health Score**: Surprise feature to assess overall chat quality
- **PDF Report Generation**: Professional reports with insights and metrics

### 🎨 **Modern UI/UX**
- **Professional Design**: Dark theme with animated gradients and shadows
- **Responsive Layout**: Optimized for desktop and mobile viewing
- **Loading Animations**: Smooth transitions and professional feedback
- **Enhanced Typography**: Poppins font with carefully crafted visual hierarchy

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Google Gemini API key (for AI threat analysis)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/WhatsApp_Chat_Sentiment.git
   cd WhatsApp_Chat_Sentiment
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API key** (for AI threat analysis)
   ```bash
   echo "your_gemini_api_key_here" > .gemini_api_key
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

## 📱 How to Export WhatsApp Chats

1. Open WhatsApp on your phone
2. Go to the chat you want to analyze
3. Tap the chat name → Export Chat
4. Choose "Without Media" for faster processing
5. Save the `.txt` file and upload it to the application

## 🛠️ Technology Stack

- **Frontend**: Streamlit with custom CSS/HTML
- **Backend**: Python with advanced NLP libraries
- **AI Integration**: Google Gemini API for threat analysis
- **Data Processing**: Pandas, NumPy for efficient data handling
- **Visualizations**: Plotly, Matplotlib, Seaborn
- **NLP Libraries**: VADER Sentiment, TextBlob, scikit-learn
- **Report Generation**: FPDF2 for professional PDF reports

## 📂 Project Structure

```
WhatsApp_Chat_Sentiment/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .gemini_api_key       # API key file (create this)
├── .gitignore           # Git ignore rules
├── README.md            # Project documentation
├── src/                 # Source code modules
├── tests/              # Unit tests
├── docs/               # Additional documentation
└── archive/            # Backup files
```

## 🎯 Key Features Breakdown

### Sentiment Analysis
- **Hybrid Approach**: Combines VADER and TextBlob for accuracy
- **Hinglish Support**: Recognizes common Hindi-English mixed words
- **Emoji Analysis**: Extracts and analyzes emoji usage patterns
- **Real-time Processing**: Fast analysis of large chat files

### AI Threat Detection
- **Context-Aware Analysis**: Understands conversation flow and participant dynamics
- **Psychological Profiling**: Advanced behavioral pattern recognition
- **Risk Stratification**: Clear threat levels (Critical, High, Medium, Low)
- **False Positive Reduction**: Smart filtering to avoid misclassification

### Professional UI
- **Dashboard Metrics**: Key insights at a glance
- **Interactive Charts**: Click and explore your data
- **Responsive Design**: Works on all screen sizes
- **Export Options**: PDF reports and data downloads

## 🔒 Security & Privacy

- **Local Processing**: All analysis happens on your machine
- **Secure API Management**: API keys stored locally, never transmitted in UI
- **No Data Storage**: Chat data is processed in memory only
- **Privacy First**: Your conversations never leave your device

## 📈 Use Cases

- **Personal**: Understand your conversation patterns and relationships
- **Research**: Academic studies on digital communication
- **Security**: Corporate chat monitoring and threat assessment
- **Mental Health**: Analyze conversation sentiment trends over time
- **Relationship Analysis**: Understand communication dynamics

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit Team** for the amazing framework
- **Google AI** for Gemini API access
- **Open Source Community** for the incredible NLP libraries
- **WhatsApp** for making chat exports possible

## 📞 Support

If you encounter any issues or have questions:
1. Check the existing issues on GitHub
2. Create a new issue with detailed information
3. Include error messages and screenshots if applicable

---

**Made with ❤️ and Python** | **Star ⭐ this repo if you found it helpful!**