from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs=dict()
)   
chat = ChatHuggingFace(llm=llm)
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="tell me about langchain."),
    
]
result = chat.invoke(messages)
messages.append(AIMessage(content=result.content))
print("AI Response:", result.content)
# print("Conversation History:", messages)