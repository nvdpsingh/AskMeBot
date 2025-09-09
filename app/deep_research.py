"""
Deep Research Mode - Multiple open-source models collaborate to provide the best answer
"""

import os
import asyncio
from typing import Dict, List, Any
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

# Available open-source models for collaboration
RESEARCH_MODELS = [
    "meta-llama/llama-3.1-70b-versatile",
    "mistralai/mixtral-8x7b-32768", 
    "qwen/qwen-2.5-72b-instruct",
    "deepseek-ai/deepseek-coder-33b-instruct",
    "google/gemma-2-27b-it"
]

FINAL_SYNTHESIS_MODEL = "openai/gpt-oss-120b"

def deep_research_analysis(prompt: str, primary_model: str) -> Dict[str, Any]:
    """
    Conduct deep research using multiple open-source models collaborating
    """
    try:
        # Step 1: Get initial analysis from different models
        model_insights = get_model_insights(prompt)
        
        # Step 2: Create debate/discussion between models
        model_debate = create_model_debate(prompt, model_insights)
        
        # Step 3: Final synthesis by the 120B model
        final_response = synthesize_final_response(prompt, model_insights, model_debate)
        
        return {
            "model": "Deep Research Mode",
            "response": final_response,
            "original_response": final_response,
            "formatted": True,
            "research_process": {
                "model_insights": model_insights,
                "model_debate": model_debate,
                "synthesis_model": FINAL_SYNTHESIS_MODEL
            }
        }
        
    except Exception as e:
        print(f"Deep research error: {e}")
        # Fallback to regular response
        from app.groq_router import query_llm
        return query_llm(prompt, primary_model)

def get_model_insights(prompt: str) -> Dict[str, str]:
    """
    Get initial insights from different open-source models
    """
    insights = {}
    
    for model in RESEARCH_MODELS:
        try:
            llm = ChatGroq(model=model)
            
            # Create specialized prompts for each model
            if "llama" in model.lower():
                system_prompt = "You are Llama 3.1, a large language model. Provide a comprehensive analysis focusing on factual accuracy and logical reasoning."
            elif "mixtral" in model.lower():
                system_prompt = "You are Mixtral, a mixture of experts model. Provide insights from multiple perspectives and consider various approaches."
            elif "qwen" in model.lower():
                system_prompt = "You are Qwen 2.5, a multilingual model. Provide detailed analysis with cross-cultural and technical perspectives."
            elif "deepseek" in model.lower():
                system_prompt = "You are DeepSeek Coder, specialized in technical analysis. Focus on technical accuracy, code examples, and implementation details."
            elif "gemma" in model.lower():
                system_prompt = "You are Gemma 2, a Google model. Provide balanced, well-researched insights with consideration for different viewpoints."
            else:
                system_prompt = "You are an AI assistant. Provide your best analysis of the given prompt."
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", "Please analyze this question and provide your insights: {prompt}")
            ])
            
            formatted_prompt = prompt_template.format_messages(prompt=prompt)
            response = llm.invoke(formatted_prompt)
            
            insights[model] = response.content.strip()
            
        except Exception as e:
            print(f"Error getting insights from {model}: {e}")
            insights[model] = f"Error analyzing with {model}: {str(e)}"
    
    return insights

def create_model_debate(prompt: str, insights: Dict[str, str]) -> str:
    """
    Create a debate/discussion between the models about their different perspectives
    """
    try:
        # Use a model to facilitate the debate
        debate_llm = ChatGroq(model="meta-llama/llama-3.1-70b-versatile")
        
        # Create debate prompt
        debate_prompt = f"""
You are facilitating a debate between multiple AI models about this question: {prompt}

Here are the different perspectives from each model:

{chr(10).join([f"**{model}**: {insight}" for model, insight in insights.items()])}

Please create a structured debate where these models discuss their different viewpoints, challenge each other's assumptions, and build upon each other's ideas. Show the collaborative thinking process and highlight areas of agreement and disagreement.

Format it as a conversation between the models, showing their reasoning process.
"""
        
        response = debate_llm.invoke(debate_prompt)
        return response.content.strip()
        
    except Exception as e:
        print(f"Error creating model debate: {e}")
        return f"Debate creation failed: {str(e)}"

def synthesize_final_response(prompt: str, insights: Dict[str, str], debate: str) -> str:
    """
    Use the 120B model to synthesize the final response without adding bias
    """
    try:
        synthesis_llm = ChatGroq(model=FINAL_SYNTHESIS_MODEL)
        
        synthesis_prompt = f"""
You are the final synthesizer in a collaborative research process. Your role is to create the best possible response by synthesizing insights from multiple AI models without adding your own bias.

Original Question: {prompt}

Individual Model Insights:
{chr(10).join([f"**{model}**: {insight}" for model, insight in insights.items()])}

Model Debate/Discussion:
{debate}

Your task:
1. Synthesize the best insights from all models
2. Resolve any contradictions by finding common ground
3. Create a comprehensive, well-structured response
4. Do NOT add your own opinions or bias
5. Present the most accurate and complete answer based on the collaborative analysis

Provide a clear, comprehensive response that represents the collective wisdom of all the models.
"""
        
        response = synthesis_llm.invoke(synthesis_prompt)
        return response.content.strip()
        
    except Exception as e:
        print(f"Error in final synthesis: {e}")
        # Fallback to a simple combination
        combined_insights = "\n\n".join(insights.values())
        return f"Based on collaborative analysis:\n\n{combined_insights}\n\nNote: Synthesis error occurred, showing combined insights instead."
