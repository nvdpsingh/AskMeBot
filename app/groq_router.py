from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from dotenv import load_dotenv
from app.chat_parser import parse_llm_output
import os

# Load environment variables from .env file
load_dotenv()

chat_history=[]
def query_llm(prompt:str,model:str):
    global chat_history
    try:
        if not os.getenv("GROQ_API_KEY"):
            return {
                "model": model,
                "response": "🤖 Hi! I'm AskMeBot, your AI assistant. GROQ_API_KEY not found in environment variables. Please Configure that to interact with the bot.",
                "error": False,
                "demo_mode": True
            }
        
        llm = ChatGroq(model=model)
        prompt_template = ChatPromptTemplate.from_messages([
            ("system","you are a helpful assistant reply to this {prompt}"),
            MessagesPlaceholder("chat_history"),
            ("user","{prompt}")
        ])
        formatted_prompt = prompt_template.format_messages(prompt=prompt, chat_history=chat_history)
        response = llm.invoke(formatted_prompt)
        
        # Update chat history with the new interaction
        chat_history.append(("user", prompt))
        chat_history.append(("assistant", str(response)))
        
        # Parse the response to convert markdown to HTML
        parsed_response = parse_llm_output(str(response.content))
        
        return {
            "model": model,
            "response": parsed_response["parsed_output"],
            "original_response": parsed_response["original_output"],
            "formatted": parsed_response["formatted"]
        }

    except Exception as e:
        print("LLM Error: ",e)
        error_str = str(e)
        return {
            "model": model,
            "response": f"Error occurred: {error_str}",
            "error": True
        }