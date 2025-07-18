from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.groq_router import query_llm

from app.chat_parser import parse_llm_output


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
    return {"status": "healthy", "message": "AskMeBot is running!"}

# Serve the main HTML file
@app.get("/")
def read_index():
    return FileResponse("static/index.html")

class ChatInput(BaseModel):
    prompt: str
    model: str

@app.post("/chat")
def chat(chat_input: ChatInput):
    return query_llm(chat_input.prompt, chat_input.model)

parsed = parse_llm_output(raw_llm_output)
@app.post("/save")
def save():
    return