from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.groq_router import query_llm
from app.chat_parser import parse_llm_output
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Health check endpoint
@app.get("/health")
def health_check():
    import os
    has_api_key = bool(os.getenv("GROQ_API_KEY"))
    return {
        "status": "healthy", 
        "message": "AskMeBot is running!",
        "api_key_available": has_api_key
    }

# Serve the main HTML file
@app.get("/")
def read_index():
    return FileResponse("static/index.html")

class ChatInput(BaseModel):
    prompt: str
    model: str

class ChatTitleInput(BaseModel):
    messages: list
    model: str

class ChangeTitleInput(BaseModel):
    new_title: str
    chat_id: str

@app.post("/chat")
def chat(chat_input: ChatInput):
    return query_llm(chat_input.prompt, chat_input.model)

@app.post("/generate-title")
def generate_chat_title(title_input: ChatTitleInput):
    """Generate a chat title based on the conversation messages"""
    try:
        if not os.getenv("GROQ_API_KEY"):
            return {"error": "GROQ_API_KEY not found", "title": "Untitled Chat"}
        
        from langchain_groq import ChatGroq
        from langchain.prompts import ChatPromptTemplate
        
        # Create a prompt for title generation
        title_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that generates user-friendly, recognizable titles for chat conversations. 
            
            Guidelines for creating titles:
            - Make titles user-oriented and easy to recognize at a glance
            - Focus on what the user is trying to accomplish or learn
            - Use action-oriented language when appropriate
            - Keep titles concise (maximum 50 characters)
            - Make them memorable and descriptive
            
            Examples of good titles:
            - "Testing My App" (not "User seeking guidance to test the app")
            - "Python Learning Help" (not "Programming assistance request")
            - "Weather App Development" (not "Building weather application")
            - "Database Connection Issues" (not "Technical troubleshooting")
            - "Recipe Recommendations" (not "Cooking advice request")
            
            Return only the title, nothing else."""),
            ("user", "Here are the conversation messages:\n{messages}")
        ])
        
        # Format the messages for the prompt
        formatted_messages = "\n".join([f"{msg['sender']}: {msg['text']}" for msg in title_input.messages])
        
        # Create the LLM instance
        llm = ChatGroq(model=title_input.model)
        
        # Generate the title
        formatted_prompt = title_prompt.format_messages(messages=formatted_messages)
        response = llm.invoke(formatted_prompt)
        
        title = response.content.strip()
        
        # Ensure title is not too long
        if len(title) > 50:
            title = title[:47] + "..."
        
        return {"title": title, "success": True}
        
    except Exception as e:
        print(f"Title generation error: {e}")
        return {"error": str(e), "title": "Untitled Chat", "success": False}

@app.post("/change-title")
def change_chat_title(title_input: ChangeTitleInput):
    """Change the title of a specific chat"""
    try:
        # This endpoint is mainly for API consistency
        # The actual title change is handled on the frontend
        return {
            "success": True, 
            "message": "Title change request received",
            "new_title": title_input.new_title
        }
    except Exception as e:
        return {"error": str(e), "success": False}

@app.post("/save")
def save():
    return {"message": "Save endpoint"}