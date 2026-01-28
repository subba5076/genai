from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

model = OpenAI(model="",temperature=0.7,max_tokens=150)
res=model.invoke("What is the capital of France?")
