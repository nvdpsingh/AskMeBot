from fastapi import APIRouter
from models.message import Message

router = APIRouter()

@router.post("/send-message")
def send_message(msg:Message):
    return {"response" : f"User {msg.username} said: {msg.message}"}

@router.get("/test")
def test():
    return {"msg":"All good, keep going Navdeep!"}

@router.get("/username/{name}")
def greet_user(name : str):
    return {"message":f"Welcome to the chatbot, {name}"}

@router.get("/greet/{name}")
def greet_user1(name : str):
    return {f"Hello Again, {name.capitalize()}"}

@router.get("/search")
def search(query : str = "", limit:int = 5):
    return{"result":f"Searching for '{query}',showing {limit} results"}