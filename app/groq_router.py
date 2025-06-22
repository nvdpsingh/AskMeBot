from langchain_groq import ChatGroq


def query_llm(prompt:str,model:str):
    try:
        llm = ChatGroq(model=model)

        response = llm.invoke([
            {"role":"system","content":"you are a helpful assistant reply to this {prompt}"},
            {"role": "user", "content": prompt}
        ])
        return {
            "model": model,
            "response":str(response)
        }

    except Exception as e:
        print("LLM Error: ",e)
        return{
            "error": str(e)
        }