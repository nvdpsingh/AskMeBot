from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
import os

chat_history=[]
def query_llm(prompt:str,model:str):
    global chat_history
    try:
        if not os.getenv("GROQ_API_KEY"):
            return {
                "model": model,
                "response": "🤖 Hi! I'm AskMeBot, your AI assistant. I'm currently running in demo mode because no Groq API key is configured.\n\nTo enable full functionality:\n1. Get a free API key from https://console.groq.com/\n2. Set the environment variable: export GROQ_API_KEY=your-key-here\n3. Restart the application\n\nFor now, I can help you with basic responses! What would you like to know?",
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
        
        return {
            "model": model,
            "response":str(response.content)
        }

    except Exception as e:
        print("LLM Error: ",e)
        error_str = str(e)
        return {
            "model": model,
            "response": f"Error occurred: {error_str}",
            "error": True
        }