from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs=dict(temperature=0.7)
)

chat = ChatHuggingFace(llm=llm)
res = chat.invoke("What is the capital of karnataka?")
print(res.content)