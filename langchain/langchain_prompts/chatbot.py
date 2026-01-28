from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv  
load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs=dict()
)   
chat = ChatHuggingFace(llm=llm)
chat_history = [SystemMessage(content="You are a helpful assistant.")]

while True:
    user_input = input("User: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat. Goodbye!")
        break

    response = chat.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("Bot:", response.content) 

print("Chat History:", chat_history)