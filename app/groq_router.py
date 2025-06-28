from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder

chat_history=[]
def query_llm(prompt:str,model:str):
    global chat_history
    try:
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
        return{
            "error": str(e)
        }