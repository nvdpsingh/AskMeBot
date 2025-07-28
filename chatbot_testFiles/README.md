# AskMeBot - MongoDB Migration

This project has been migrated from Supabase to MongoDB for better performance and flexibility.

## 🚀 Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up MongoDB:
```bash
# macOS
brew install mongodb-community
brew services start mongodb-community

# Or use MongoDB Atlas (cloud)
```

3. Create `.env` file:
```env
GROQ_API_KEY=your_groq_api_key
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=askmebot
JWT_SECRET_KEY=your_secret_key
```

4. Run the application:
```bash
python main.py
```

## 🔄 Migration Benefits

- **Better Performance**: MongoDB's document-based storage is perfect for chat data
- **Flexible Schema**: Easy to add new fields without migrations
- **Scalability**: MongoDB scales horizontally better than PostgreSQL
- **JSON Native**: Natural fit for JavaScript/JSON data structures

## 📊 Database Collections

- `users` - User accounts and authentication
- `sessions` - Chat sessions per user
- `chat_messages` - Individual chat messages with metadata

## 🧪 Test Endpoints

- `GET /health` - Health check with MongoDB status
- `GET /api/test` - Test MongoDB connection and operations

## 🔧 Development

The application now uses:
- **Motor** - Async MongoDB driver for Python
- **FastAPI** - Modern web framework
- **MongoDB** - Document database
- **JWT** - Authentication tokens 