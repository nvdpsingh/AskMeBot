# 🦇 AskMeBot - AI-Powered Chat Assistant

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen)](https://nvdpsingh.github.io/AskMeBot/)
[![Python](https://img.shields.io/badge/Python-3.13+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A modern, AI-powered chat assistant built with FastAPI and Groq's LLM models. Features a sleek Batman-themed UI, real-time chat functionality, and intelligent conversation management.

## ✨ Features

### 🤖 **AI Chat Capabilities**
- **Multiple LLM Models**: Support for OpenAI GPT OSS models via Groq
- **Real-time Responses**: Fast, streaming chat responses
- **Markdown Support**: Rich text formatting with **bold**, *italic*, `code`, and headings
- **Smart Title Generation**: Automatic, user-friendly chat titles

### 💬 **Chat Management**
- **Persistent Chat History**: All conversations saved locally
- **Chat Organization**: Sidebar with chat history and management
- **Rename Chats**: Change titles via commands or inline editing
- **Delete Chats**: Individual or bulk chat deletion with confirmation
- **Export/Import**: Download chat history as JSON

### 🎨 **Modern UI/UX**
- **Batman Theme**: Dark, professional design with golden accents
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Smooth Animations**: Floating background elements and transitions
- **Keyboard Shortcuts**: Quick actions with keyboard combinations
- **Toast Notifications**: Real-time feedback for user actions

### ⚙️ **Advanced Features**
- **Environment Configuration**: Secure API key management
- **Health Monitoring**: Built-in health check endpoints
- **Error Handling**: Graceful error management and user feedback
- **Mobile Responsive**: Optimized for all screen sizes

## 🚀 Live Demo

**Try AskMeBot now**: [https://nvdpsingh.github.io/AskMeBot/](https://nvdpsingh.github.io/AskMeBot/)

## 📋 Prerequisites

- Python 3.13+
- Groq API Key
- Modern web browser

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/nvdpsingh/AskMeBot.git
cd AskMeBot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Setup
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the Application
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000` to access the application.

## 🏗️ Project Structure

```
AskMeBot/
├── app/                    # Backend application
│   ├── main.py            # FastAPI application and endpoints
│   ├── groq_router.py     # LLM integration and routing
│   ├── chat_parser.py     # Markdown parsing and response formatting
│   ├── context_manager.py # Context management utilities
│   ├── doc_parser.py      # Document parsing utilities
│   ├── memory_builder.py  # Memory and context building
│   └── utils.py           # General utilities
├── static/                # Frontend assets
│   └── index.html         # Main HTML file with embedded CSS/JS
├── .github/               # GitHub Actions workflows
│   └── workflows/
│       └── deploy.yml     # CI/CD pipeline for GitHub Pages
├── requirements.txt       # Python dependencies
├── pyproject.toml        # Project configuration
├── README.md             # This file
└── .env.example          # Environment variables template
```

## 🔧 API Endpoints

### Core Endpoints
- `GET /` - Serve the main application
- `GET /health` - Health check endpoint
- `POST /chat` - Send chat messages and receive AI responses
- `POST /generate-title` - Generate smart chat titles
- `POST /change-title` - Update chat titles

### Request/Response Examples

#### Chat Request
```json
{
  "prompt": "Hello, how are you?",
  "model": "openai/gpt-oss-20b"
}
```

#### Chat Response
```json
{
  "model": "openai/gpt-oss-20b",
  "response": "Hello! I'm doing well, thank you for asking. How can I assist you today?",
  "error": false
}
```

## 🎯 Usage Guide

### Basic Chat
1. Open the application in your browser
2. Type your message in the input field
3. Press Enter or click Send
4. View the AI response with markdown formatting

### Chat Management
- **New Chat**: Click "New Chat" in the sidebar
- **Rename Chat**: 
  - Use commands: "change my heading to [new name]"
  - Click the edit icon next to any chat title
- **Delete Chat**: 
  - Click the trash icon next to any chat
  - Use `Ctrl+Delete` for the current chat
- **Export Chats**: Go to Settings → Export Chats

### Keyboard Shortcuts
- `Enter` - Send message
- `Shift+Enter` - New line in message
- `Ctrl+Delete` - Delete current chat

## 🔐 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Your Groq API key for LLM access | Yes |

## 🚀 Deployment

### GitHub Pages (Automatic)
The application is automatically deployed to GitHub Pages using GitHub Actions:

1. **Push to main branch** triggers the deployment
2. **GitHub Actions** builds and deploys the application
3. **Live site** available at `https://nvdpsingh.github.io/FastAPITut/`

### Manual Deployment
For other platforms:

```bash
# Build the application
pip install -r requirements.txt

# Run with production settings
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 🧪 Testing

### Health Check
```bash
curl http://localhost:8000/health
```

### Chat API Test
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "model": "openai/gpt-oss-20b"}'
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Development

### Adding New Features
1. Update the backend in `app/` directory
2. Modify the frontend in `static/index.html`
3. Test locally with `uvicorn app.main:app --reload`
4. Update documentation as needed

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable names
- Add comments for complex logic
- Keep functions focused and small

## 🐛 Troubleshooting

### Common Issues

#### API Key Not Working
- Ensure `.env` file exists with `GROQ_API_KEY`
- Check API key is valid and has sufficient credits
- Restart the server after adding environment variables

#### Chat Not Loading
- Check browser console for JavaScript errors
- Verify server is running on correct port
- Clear browser cache and localStorage

#### Styling Issues
- Ensure all CSS is properly loaded
- Check for conflicting styles
- Verify Tailwind CSS classes are correct

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Groq](https://groq.com/) - Fast LLM inference
- [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS framework
- [Font Awesome](https://fontawesome.com/) - Icons

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/nvdpsingh/AskMeBot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nvdpsingh/AskMeBot/discussions)
- **Email**: [Contact Developer](mailto:your-email@example.com)

---

<div align="center">
  <p>Made with ❤️ and ☕ by <a href="https://github.com/nvdpsingh">Navdeep Singh</a></p>
  <p>
    <a href="https://nvdpsingh.github.io/AskMeBot/">🌐 Live Demo</a> •
    <a href="https://github.com/nvdpsingh/AskMeBot/issues">🐛 Report Bug</a> •
    <a href="https://github.com/nvdpsingh/AskMeBot/discussions">💬 Request Feature</a>
  </p>
</div>