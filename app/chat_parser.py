from langchain_groq import ChatGroq


llm = ChatGroq(model="gemma2-9b-it")
def parse_llm_output(llm_output:str):
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