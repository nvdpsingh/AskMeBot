from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.groq_router import query_llm
from app.chat_parser import parse_llm_output
from app.mirix_memory import mirix_manager, MemoryType
from dotenv import load_dotenv
import os
import logging
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('askmebot.log')  # File output
    ]
)

# Create logger
logger = logging.getLogger(__name__)


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
    logger.info("Health check requested")
    has_api_key = bool(os.getenv("GROQ_API_KEY"))
    logger.info(f"API key available: {has_api_key}")
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
def chat(chat_input: dict):
    """Main chat endpoint with deep research mode support"""
    logger.info("=" * 80)
    logger.info("NEW CHAT REQUEST RECEIVED")
    logger.info("=" * 80)
    
    try:
        if not os.getenv("GROQ_API_KEY"):
            logger.error("GROQ_API_KEY not found in environment")
            return {"error": "GROQ_API_KEY not found"}
        
        prompt = chat_input.get("prompt", "")
        model = chat_input.get("model", "openai/gpt-oss-20b")
        deep_research_mode = chat_input.get("deepResearchMode", False)
        chat_history = chat_input.get("chatHistory", [])
        user_id = chat_input.get("userId", "default")
        
        logger.info(f"📝 User Prompt: {prompt}")
        logger.info(f"🤖 Selected Model: {model}")
        logger.info(f"🧠 Deep Research Mode: {deep_research_mode}")
        logger.info(f"👤 User ID: {user_id}")
        logger.info(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if not prompt:
            logger.warning("Empty prompt provided")
            return {"error": "No prompt provided"}
        
        # 🧠 MIRIX Memory Integration
        logger.info("🔍 MIRIX: Retrieving relevant memories...")
        memory_context = mirix_manager.get_memory_context(prompt, {
            "user_id": user_id,
            "model": model,
            "deep_research_mode": deep_research_mode,
            "chat_history": chat_history
        })
        
        if memory_context:
            logger.info(f"🧠 MIRIX: Found relevant memories ({len(memory_context)} chars)")
            # Enhance prompt with memory context
            enhanced_prompt = f"{memory_context}\n\nUser Query: {prompt}"
        else:
            logger.info("🧠 MIRIX: No relevant memories found")
            enhanced_prompt = prompt
        
        if deep_research_mode:
            logger.info("🚀 Starting LangGraph Deep Research Mode")
            logger.info("🔄 Initializing multi-agent collaboration...")
            # Use LangGraph deep research mode with ReAct agents
            from app.langgraph_research import deep_research_analysis
            result = deep_research_analysis(enhanced_prompt, model)
            logger.info("✅ LangGraph Deep Research completed")
        else:
            logger.info("💬 Starting regular chat mode")
            # Use the groq_router to get the response
            result = query_llm(enhanced_prompt, model)
            logger.info("✅ Regular chat completed")
        
        logger.info(f"📊 Response Model: {result.get('model', 'Unknown')}")
        logger.info(f"📏 Response Length: {len(str(result.get('response', '')))} characters")
        
        # 🧠 MIRIX Memory Storage
        logger.info("💾 MIRIX: Storing conversation in memory...")
        try:
            # Store conversation in episodic memory
            conversation_data = {
                "user_prompt": prompt,
                "response": result.get('response', ''),
                "model": model,
                "deep_research_mode": deep_research_mode,
                "memory_context_used": bool(memory_context),
                "response_length": len(str(result.get('response', ''))),
                "priority": 5
            }
            mirix_manager.update_conversation_memory(user_id, conversation_data)
            
            # Store key facts in semantic memory if response contains factual information
            response_text = str(result.get('response', ''))
            if any(keyword in response_text.lower() for keyword in ['fact', 'information', 'data', 'research', 'study']):
                # Extract and store key facts
                from app.mirix_memory import SemanticMemoryEntry
                fact_entry = SemanticMemoryEntry(
                    content=f"User asked: {prompt[:100]}... | Response: {response_text[:200]}...",
                    timestamp=datetime.now(),
                    priority=6,
                    concept="user_interaction",
                    source="chat_conversation",
                    tags=["conversation", "factual", user_id]
                )
                mirix_manager.add_memory(MemoryType.SEMANTIC, fact_entry)
            
            logger.info("✅ MIRIX: Memory storage completed")
        except Exception as e:
            logger.error(f"❌ MIRIX Memory storage error: {e}")
        
        logger.info("=" * 80)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        logger.error("=" * 80)
        return {"error": str(e)}

@app.post("/generate-title")
def generate_chat_title(title_input: ChatTitleInput):
    """Generate a chat title based on the conversation messages"""
    logger.info("📝 Title generation requested")
    logger.info(f"📊 Messages count: {len(title_input.messages)}")
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

@app.post("/enhance-prompt")
def enhance_prompt(enhance_input: dict):
    """Enhance a prompt using the COSTAR principle with chat history context"""
    logger.info("✨ Prompt enhancement requested")
    try:
        if not os.getenv("GROQ_API_KEY"):
            logger.error("GROQ_API_KEY not found for prompt enhancement")
            return {"error": "GROQ_API_KEY not found", "success": False}
        
        from langchain_groq import ChatGroq
        from langchain.prompts import ChatPromptTemplate
        
        prompt = enhance_input.get("prompt", "")
        model = enhance_input.get("model", "openai/gpt-oss-20b")
        chat_history = enhance_input.get("chatHistory", [])
        
        if not prompt:
            return {"error": "No prompt provided", "success": False}
        
        # Build context from chat history
        context = ""
        if chat_history:
            context = "Previous conversation context:\n"
            for msg in chat_history[-3:]:  # Last 3 messages for context
                context += f"{msg.get('sender', 'user')}: {msg.get('text', '')}\n"
            context += "\n"
        
        # Enhanced prompt generation
        enhancement_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an expert prompt engineer. Your task is to enhance user prompts to make them more effective and specific.

{context}

Instructions:
1. Analyze the user's prompt and identify what they're trying to achieve
2. Enhance it by adding necessary context, clarity, and structure
3. Make the prompt more specific and actionable
4. Keep the enhanced prompt concise but comprehensive
5. Maintain the user's original intent while improving effectiveness
6. Add relevant background information if needed
7. Specify the desired output format or style when appropriate
8. If the prompt is already well-structured, make minor improvements for clarity

Return only the enhanced prompt, nothing else."""),
            ("user", "Please enhance this prompt: {prompt}")
        ])
        
        llm = ChatGroq(model=model)
        formatted_prompt = enhancement_prompt.format_messages(prompt=prompt)
        response = llm.invoke(formatted_prompt)
        
        enhanced_prompt = response.content.strip()
        
        return {
            "success": True,
            "enhanced_prompt": enhanced_prompt,
            "original_prompt": prompt
        }
        
    except Exception as e:
        print(f"Prompt enhancement error: {e}")
        return {"error": str(e), "success": False}

@app.post("/save")
def save():
    return {"message": "Save endpoint"}

# 🧠 MIRIX Memory Management Endpoints

@app.get("/memory/search")
def search_memory(query: str, memory_type: str = None, limit: int = 10):
    """Search across MIRIX memory system"""
    logger.info(f"🔍 Memory search requested: {query}")
    try:
        if memory_type and memory_type in [mt.value for mt in MemoryType]:
            memory_types = [MemoryType(memory_type)]
        else:
            memory_types = None
        
        results = mirix_manager.search_memory(query, memory_types, limit)
        
        # Format results for frontend
        formatted_results = {}
        for mt, entries in results.items():
            formatted_results[mt.value] = [
                {
                    "id": entry.id,
                    "content": entry.content,
                    "timestamp": entry.timestamp.isoformat(),
                    "priority": entry.priority,
                    "tags": entry.tags,
                    "metadata": entry.metadata
                }
                for entry in entries
            ]
        
        return {
            "success": True,
            "results": formatted_results,
            "total_found": sum(len(entries) for entries in results.values())
        }
    except Exception as e:
        logger.error(f"Memory search error: {e}")
        return {"error": str(e), "success": False}

@app.get("/memory/stats")
def get_memory_stats():
    """Get memory system statistics"""
    logger.info("📊 Memory stats requested")
    try:
        stats = {}
        for memory_type, agent in mirix_manager.agents.items():
            stats[memory_type.value] = {
                "total_entries": len(agent.entries),
                "storage_path": str(agent.storage_path)
            }
        
        return {
            "success": True,
            "stats": stats,
            "total_memories": sum(len(agent.entries) for agent in mirix_manager.agents.values())
        }
    except Exception as e:
        logger.error(f"Memory stats error: {e}")
        return {"error": str(e), "success": False}

@app.post("/memory/add")
def add_memory(memory_input: dict):
    """Add memory entry to specific memory type"""
    logger.info("➕ Add memory requested")
    try:
        memory_type_str = memory_input.get("memory_type", "episodic")
        content = memory_input.get("content", "")
        priority = memory_input.get("priority", 5)
        tags = memory_input.get("tags", [])
        metadata = memory_input.get("metadata", {})
        
        if not content:
            return {"error": "No content provided", "success": False}
        
        # Create appropriate memory entry
        if memory_type_str == "core":
            from app.mirix_memory import CoreMemoryEntry
            entry = CoreMemoryEntry(
                content=content,
                timestamp=datetime.now(),
                priority=priority,
                tags=tags,
                metadata=metadata
            )
        elif memory_type_str == "episodic":
            from app.mirix_memory import EpisodicMemoryEntry
            entry = EpisodicMemoryEntry(
                content=content,
                timestamp=datetime.now(),
                priority=priority,
                tags=tags,
                metadata=metadata
            )
        elif memory_type_str == "semantic":
            from app.mirix_memory import SemanticMemoryEntry
            entry = SemanticMemoryEntry(
                content=content,
                timestamp=datetime.now(),
                priority=priority,
                tags=tags,
                metadata=metadata
            )
        elif memory_type_str == "procedural":
            from app.mirix_memory import ProceduralMemoryEntry
            entry = ProceduralMemoryEntry(
                content=content,
                timestamp=datetime.now(),
                priority=priority,
                tags=tags,
                metadata=metadata
            )
        elif memory_type_str == "resource":
            from app.mirix_memory import ResourceMemoryEntry
            entry = ResourceMemoryEntry(
                content=content,
                timestamp=datetime.now(),
                priority=priority,
                tags=tags,
                metadata=metadata
            )
        elif memory_type_str == "knowledge_vault":
            from app.mirix_memory import KnowledgeVaultEntry
            entry = KnowledgeVaultEntry(
                content=content,
                timestamp=datetime.now(),
                priority=priority,
                tags=tags,
                metadata=metadata
            )
        else:
            return {"error": f"Invalid memory type: {memory_type_str}", "success": False}
        
        # Add to memory system
        memory_type_enum = MemoryType(memory_type_str)
        entry_id = mirix_manager.add_memory(memory_type_enum, entry)
        
        return {
            "success": True,
            "entry_id": entry_id,
            "memory_type": memory_type_str,
            "message": f"Memory added to {memory_type_str} memory"
        }
    except Exception as e:
        logger.error(f"Add memory error: {e}")
        return {"error": str(e), "success": False}

@app.delete("/memory/{memory_type}/{entry_id}")
def delete_memory(memory_type: str, entry_id: str):
    """Delete specific memory entry"""
    logger.info(f"🗑️ Delete memory requested: {memory_type}/{entry_id}")
    try:
        if memory_type not in [mt.value for mt in MemoryType]:
            return {"error": f"Invalid memory type: {memory_type}", "success": False}
        
        memory_type_enum = MemoryType(memory_type)
        success = mirix_manager.agents[memory_type_enum].delete_entry(entry_id)
        
        if success:
            return {
                "success": True,
                "message": f"Memory entry {entry_id} deleted from {memory_type} memory"
            }
        else:
            return {
                "success": False,
                "error": f"Memory entry {entry_id} not found in {memory_type} memory"
            }
    except Exception as e:
        logger.error(f"Delete memory error: {e}")
        return {"error": str(e), "success": False}