from fastapi import FastAPI
from pydantic import BaseModel
from .groq_router import query_llm

app = FastAPI()

class ChatInput(BaseModel):
    prompt: str
    model: str

@app.post("/chat")
def chat(chat_input: ChatInput):
    return query_llm(chat_input.prompt, chat_input.model)