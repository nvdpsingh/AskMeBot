from langchain_groq import ChatGroq


def query_llm(prompt:str,model:str):
    llm = ChatGroq(model=model)

    llm.invoke([
        {"system":"you are a helpful assistant reply to this {prompt}"},
        {"user": prompt}
    ])
