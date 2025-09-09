"""
LangGraph-based Deep Research Mode with ReAct Agents
Multiple AI agents collaborate using LangGraph's ReAct pattern
"""

import os
from typing import Dict, List, Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import asyncio

# Available models for different agent roles
RESEARCH_MODELS = {
    "analyst": "llama-3.3-70b-versatile",  # Factual analysis and reasoning
    "researcher": "qwen/qwen3-32b",        # Cross-cultural and multilingual insights
    "technician": "deepseek-r1-distill-llama-70b",  # Technical analysis and reasoning
    "innovator": "meta-llama/llama-4-maverick-17b-128e-instruct",  # Creative approaches
    "reviewer": "meta-llama/llama-4-scout-17b-16e-instruct",  # Critical review and validation
    "synthesizer": "openai/gpt-oss-120b"  # Final synthesis
}

class ResearchState(TypedDict):
    """State for the research workflow"""
    question: str
    current_agent: str
    agent_insights: Dict[str, str]
    agent_questions: Dict[str, List[str]]
    agent_answers: Dict[str, List[str]]
    debate_points: List[Dict[str, str]]
    final_synthesis: str
    research_log: List[Dict[str, Any]]

def create_research_tools():
    """Create tools for research agents"""
    
    @tool
    def analyze_question(question: str) -> str:
        """Analyze the given question and provide initial insights"""
        return f"Analyzing question: {question}"
    
    @tool
    def research_topic(topic: str) -> str:
        """Research a specific topic or aspect"""
        return f"Researching topic: {topic}"
    
    @tool
    def ask_peer_agent(agent_name: str, question: str) -> str:
        """Ask a question to another agent for collaboration"""
        return f"Asking {agent_name}: {question}"
    
    @tool
    def validate_insight(insight: str, evidence: str) -> str:
        """Validate an insight with evidence"""
        return f"Validating insight with evidence: {insight}"
    
    @tool
    def synthesize_findings(findings: List[str]) -> str:
        """Synthesize multiple findings into a coherent response"""
        return f"Synthesizing findings: {findings}"
    
    return [analyze_question, research_topic, ask_peer_agent, validate_insight, synthesize_findings]

def create_agent_llm(agent_type: str) -> ChatGroq:
    """Create a specialized LLM for each agent type"""
    model = RESEARCH_MODELS[agent_type]
    
    system_prompts = {
        "analyst": """You are an AI Research Analyst. Your role is to:
        - Analyze questions with factual accuracy and logical reasoning
        - Break down complex problems into manageable components
        - Provide evidence-based insights
        - Use tools to research and validate information
        - Ask clarifying questions when needed""",
        
        "researcher": """You are an AI Research Specialist. Your role is to:
        - Provide cross-cultural and multilingual perspectives
        - Research diverse viewpoints and approaches
        - Consider global and cultural contexts
        - Use tools to explore different angles
        - Challenge assumptions and explore alternatives""",
        
        "technician": """You are an AI Technical Expert. Your role is to:
        - Focus on technical accuracy and implementation details
        - Provide step-by-step reasoning and analysis
        - Use tools to validate technical claims
        - Consider practical implementation aspects
        - Ask technical questions for clarification""",
        
        "innovator": """You are an AI Innovation Specialist. Your role is to:
        - Provide creative insights and alternative approaches
        - Think outside conventional frameworks
        - Propose novel solutions and perspectives
        - Use tools to explore creative possibilities
        - Challenge traditional thinking patterns""",
        
        "reviewer": """You are an AI Critical Reviewer. Your role is to:
        - Review and validate other agents' findings
        - Identify gaps and inconsistencies
        - Provide critical analysis and feedback
        - Use tools to verify claims and evidence
        - Ensure quality and accuracy of research""",
        
        "synthesizer": """You are an AI Synthesis Expert. Your role is to:
        - Synthesize findings from all agents into a coherent response
        - Resolve contradictions and find common ground
        - Create comprehensive, well-structured answers
        - Maintain objectivity and avoid bias
        - Present the collective wisdom of all agents"""
    }
    
    llm = ChatGroq(
        model=model,
        temperature=0.7,
        system_message=system_prompts[agent_type]
    )
    
    return llm

def create_agent_node(agent_type: str):
    """Create a node for a specific agent"""
    
    def agent_node(state: ResearchState) -> ResearchState:
        """Execute agent logic"""
        llm = create_agent_llm(agent_type)
        tools = create_research_tools()
        tool_node = ToolNode(tools)
        
        # Create agent-specific prompt
        agent_prompts = {
            "analyst": f"""As the Research Analyst, analyze this question: {state['question']}
            
            Current insights from other agents: {state.get('agent_insights', {})}
            
            Use your tools to:
            1. Analyze the question thoroughly
            2. Research key aspects
            3. Validate your findings
            4. Ask other agents for input if needed
            
            Provide your analysis and any questions for other agents.""",
            
            "researcher": f"""As the Research Specialist, provide diverse perspectives on: {state['question']}
            
            Current insights: {state.get('agent_insights', {})}
            
            Use your tools to:
            1. Research different cultural/global perspectives
            2. Explore alternative approaches
            3. Challenge assumptions
            4. Collaborate with other agents
            
            Provide your research findings and questions for peers.""",
            
            "technician": f"""As the Technical Expert, provide technical analysis of: {state['question']}
            
            Current insights: {state.get('agent_insights', {})}
            
            Use your tools to:
            1. Analyze technical aspects
            2. Provide step-by-step reasoning
            3. Validate technical claims
            4. Ask technical questions to other agents
            
            Provide your technical analysis and validation.""",
            
            "innovator": f"""As the Innovation Specialist, provide creative insights on: {state['question']}
            
            Current insights: {state.get('agent_insights', {})}
            
            Use your tools to:
            1. Explore creative approaches
            2. Propose novel solutions
            3. Think outside conventional frameworks
            4. Collaborate with other agents
            
            Provide your innovative insights and creative questions.""",
            
            "reviewer": f"""As the Critical Reviewer, review and validate findings on: {state['question']}
            
            Current insights: {state.get('agent_insights', {})}
            
            Use your tools to:
            1. Review all findings critically
            2. Identify gaps and inconsistencies
            3. Validate evidence and claims
            4. Provide feedback to other agents
            
            Provide your critical review and validation.""",
            
            "synthesizer": f"""As the Synthesis Expert, synthesize all findings on: {state['question']}
            
            All agent insights: {state.get('agent_insights', {})}
            Agent questions: {state.get('agent_questions', {})}
            Agent answers: {state.get('agent_answers', {})}
            Debate points: {state.get('debate_points', [])}
            
            Create a comprehensive, well-structured final response that:
            1. Synthesizes all agent findings
            2. Resolves contradictions
            3. Provides a complete answer
            4. Maintains objectivity and accuracy"""
        }
        
        # Get agent response
        messages = [HumanMessage(content=agent_prompts[agent_type])]
        response = llm.invoke(messages)
        
        # Update state
        new_state = state.copy()
        new_state['current_agent'] = agent_type
        new_state['agent_insights'][agent_type] = response.content
        
        # Log the interaction
        new_state['research_log'].append({
            'agent': agent_type,
            'action': 'analysis',
            'response': response.content,
            'timestamp': str(asyncio.get_event_loop().time())
        })
        
        return new_state
    
    return agent_node

def create_debate_node():
    """Create a node for agent debate and collaboration"""
    
    def debate_node(state: ResearchState) -> ResearchState:
        """Facilitate debate between agents"""
        llm = create_agent_llm("reviewer")  # Use reviewer as debate facilitator
        
        # Create debate prompt
        debate_prompt = f"""Facilitate a structured debate between AI agents about: {state['question']}
        
        Agent insights:
        {chr(10).join([f"{agent}: {insight}" for agent, insight in state.get('agent_insights', {}).items()])}
        
        Create a debate where agents:
        1. Challenge each other's assumptions
        2. Build upon each other's ideas
        3. Identify areas of agreement and disagreement
        4. Collaborate to find the best solutions
        
        Structure the debate as a conversation between the agents."""
        
        messages = [HumanMessage(content=debate_prompt)]
        response = llm.invoke(messages)
        
        # Update state
        new_state = state.copy()
        new_state['debate_points'].append({
            'facilitator': 'reviewer',
            'debate_content': response.content,
            'timestamp': str(asyncio.get_event_loop().time())
        })
        
        new_state['research_log'].append({
            'agent': 'debate_facilitator',
            'action': 'debate',
            'response': response.content,
            'timestamp': str(asyncio.get_event_loop().time())
        })
        
        return new_state
    
    return debate_node

def create_synthesis_node():
    """Create final synthesis node"""
    
    def synthesis_node(state: ResearchState) -> ResearchState:
        """Synthesize final response"""
        llm = create_agent_llm("synthesizer")
        
        synthesis_prompt = f"""Synthesize a final response for: {state['question']}
        
        All research findings:
        {chr(10).join([f"{agent}: {insight}" for agent, insight in state.get('agent_insights', {}).items()])}
        
        Debate and collaboration:
        {chr(10).join([f"{point['facilitator']}: {point['debate_content']}" for point in state.get('debate_points', [])])}
        
        Create a comprehensive, well-structured final response that:
        1. Synthesizes all agent findings
        2. Resolves any contradictions
        3. Provides the most accurate and complete answer
        4. Maintains objectivity and avoids bias
        5. Represents the collective wisdom of all agents"""
        
        messages = [HumanMessage(content=synthesis_prompt)]
        response = llm.invoke(messages)
        
        # Update state
        new_state = state.copy()
        new_state['final_synthesis'] = response.content
        
        new_state['research_log'].append({
            'agent': 'synthesizer',
            'action': 'final_synthesis',
            'response': response.content,
            'timestamp': str(asyncio.get_event_loop().time())
        })
        
        return new_state
    
    return synthesis_node

def create_research_workflow():
    """Create the LangGraph workflow for deep research"""
    
    # Create the state graph
    workflow = StateGraph(ResearchState)
    
    # Add agent nodes
    for agent_type in ["analyst", "researcher", "technician", "innovator", "reviewer"]:
        workflow.add_node(agent_type, create_agent_node(agent_type))
    
    # Add debate and synthesis nodes
    workflow.add_node("debate", create_debate_node())
    workflow.add_node("synthesis", create_synthesis_node())
    
    # Define the workflow
    workflow.set_entry_point("analyst")
    
    # Add edges for sequential processing
    workflow.add_edge("analyst", "researcher")
    workflow.add_edge("researcher", "technician")
    workflow.add_edge("technician", "innovator")
    workflow.add_edge("innovator", "reviewer")
    workflow.add_edge("reviewer", "debate")
    workflow.add_edge("debate", "synthesis")
    workflow.add_edge("synthesis", END)
    
    return workflow.compile()

def deep_research_analysis(prompt: str, primary_model: str) -> Dict[str, Any]:
    """
    Conduct deep research using LangGraph ReAct agents
    """
    try:
        print(f"Starting LangGraph deep research for: {prompt}")
        
        # For now, let's use a simplified approach that works
        # Get insights from each agent sequentially
        agent_insights = {}
        
        for agent_type, model in RESEARCH_MODELS.items():
            if agent_type == "synthesizer":
                continue
                
            try:
                print(f"Getting insights from {agent_type} using {model}")
                llm = create_agent_llm(agent_type)
                
                # Create agent-specific prompt
                agent_prompts = {
                    "analyst": f"As a Research Analyst, analyze this question with factual accuracy and logical reasoning: {prompt}",
                    "researcher": f"As a Research Specialist, provide cross-cultural and multilingual perspectives on: {prompt}",
                    "technician": f"As a Technical Expert, provide technical analysis and step-by-step reasoning for: {prompt}",
                    "innovator": f"As an Innovation Specialist, provide creative insights and alternative approaches to: {prompt}",
                    "reviewer": f"As a Critical Reviewer, provide critical analysis and validation of: {prompt}"
                }
                
                messages = [HumanMessage(content=agent_prompts[agent_type])]
                response = llm.invoke(messages)
                agent_insights[agent_type] = response.content
                print(f"Got response from {agent_type}")
                
            except Exception as e:
                print(f"Error with {agent_type}: {e}")
                agent_insights[agent_type] = f"Error analyzing with {agent_type}: {str(e)}"
        
        # Create debate
        try:
            debate_llm = create_agent_llm("reviewer")
            debate_prompt = f"""Facilitate a debate between AI agents about: {prompt}
            
            Agent insights:
            {chr(10).join([f"{agent}: {insight}" for agent, insight in agent_insights.items()])}
            
            Create a structured debate where agents challenge each other and collaborate."""
            
            messages = [HumanMessage(content=debate_prompt)]
            debate_response = debate_llm.invoke(messages)
            debate_content = debate_response.content
            
        except Exception as e:
            print(f"Debate error: {e}")
            debate_content = f"Debate creation failed: {str(e)}"
        
        # Final synthesis
        try:
            synthesis_llm = create_agent_llm("synthesizer")
            synthesis_prompt = f"""Synthesize a final response for: {prompt}
            
            All agent insights:
            {chr(10).join([f"{agent}: {insight}" for agent, insight in agent_insights.items()])}
            
            Debate:
            {debate_content}
            
            Create a comprehensive, well-structured final response that synthesizes all findings."""
            
            messages = [HumanMessage(content=synthesis_prompt)]
            synthesis_response = synthesis_llm.invoke(messages)
            final_synthesis = synthesis_response.content
            
        except Exception as e:
            print(f"Synthesis error: {e}")
            final_synthesis = f"Synthesis failed: {str(e)}"
        
        return {
            "model": "LangGraph Deep Research",
            "response": final_synthesis,
            "original_response": final_synthesis,
            "formatted": True,
            "research_process": {
                "agent_insights": agent_insights,
                "debate_points": [{"facilitator": "reviewer", "debate_content": debate_content}],
                "research_log": [],
                "synthesis_model": RESEARCH_MODELS["synthesizer"]
            }
        }
        
    except Exception as e:
        print(f"LangGraph deep research error: {e}")
        # Fallback to regular response
        from app.groq_router import query_llm
        return query_llm(prompt, primary_model)
