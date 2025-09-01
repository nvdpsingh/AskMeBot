from langchain_groq import ChatGroq
import os

# Only initialize if API key is available
llm = None
if os.getenv("GROQ_API_KEY"):
    try:
        llm = ChatGroq(model="openai/gpt-oss-20b")
    except Exception as e:
        print(f"Failed to initialize Groq client: {e}")
        llm = None
def parse_llm_output(llm_output:str):
    if llm is None:
        return {"error": "Groq API key not configured", "parsed_output": llm_output}
    
    system_prompt = """
    You are a helpful assistant that can parse the output of a LLM.
    You will be given a string of text that is the output of a LLM.
    You will need to parse the text and return a structured output.
    """
    user_prompt = f"""
    Here is the output of a LLM:
    {llm_output}
    """
    return llm.invoke(system_prompt, user_prompt).content