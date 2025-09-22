from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from dotenv import load_dotenv
from app.chat_parser import parse_llm_output
import os
import logging
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Create logger for groq router
logger = logging.getLogger(__name__)

chat_history=[]
def query_llm(prompt:str,model:str):
    global chat_history
    logger.info("💬 GROQ ROUTER - REGULAR CHAT MODE")
    logger.info("=" * 50)
    logger.info(f"📝 Prompt: {prompt}")
    logger.info(f"🤖 Model: {model}")
    logger.info(f"⏰ Start Time: {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        if not os.getenv("GROQ_API_KEY"):
            logger.error("❌ GROQ_API_KEY not found")
            return {
                "model": model,
                "response": "🤖 Hi! I'm AskMeBot, your AI assistant. GROQ_API_KEY not found in environment variables. Please Configure that to interact with the bot.",
                "error": False,
                "demo_mode": True
            }
        
        logger.info("🔧 Initializing ChatGroq...")
        llm = ChatGroq(model=model)
        
        logger.info("📝 Creating prompt template...")
        prompt_template = ChatPromptTemplate.from_messages([
            ("system","you are a helpful assistant reply to this {prompt}"),
            MessagesPlaceholder("chat_history"),
            ("user","{prompt}")
        ])
        
        logger.info(f"📊 Chat history length: {len(chat_history)} messages")
        logger.info("📤 Formatting prompt...")
        formatted_prompt = prompt_template.format_messages(prompt=prompt, chat_history=chat_history)
        
        logger.info("🚀 Sending request to Groq API...")
        response = llm.invoke(formatted_prompt)
        
        logger.info("✅ Response received from Groq")
        logger.info(f"📏 Response length: {len(response.content)} characters")
        
        # Update chat history with the new interaction
        chat_history.append(("user", prompt))
        chat_history.append(("assistant", str(response)))
        logger.info(f"📚 Chat history updated: {len(chat_history)} messages")
        
        # Parse the response to convert markdown to HTML
        logger.info("🔍 Parsing LLM output for markdown...")
        logger.info(f"📝 Raw response type: {type(response)}")
        logger.info(f"📝 Response content type: {type(response.content)}")
        logger.info(f"📝 Response content preview: {str(response.content)[:200]}...")
        parsed_response = parse_llm_output(str(response.content))
        logger.info(f"✅ Parsing completed - Formatted: {parsed_response['formatted']}")
        
        logger.info("🎉 GROQ ROUTER COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
        
        return {
            "model": model,
            "response": parsed_response["parsed_output"],
            "original_response": parsed_response["original_output"],
            "formatted": parsed_response["formatted"]
        }

    except Exception as e:
        logger.error("❌ GROQ ROUTER ERROR")
        logger.error("=" * 50)
        logger.error(f"❌ Error: {e}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        logger.error("=" * 50)
        
        error_str = str(e)
        return {
            "model": model,
            "response": f"Error occurred: {error_str}",
            "error": True
        }