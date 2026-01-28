from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

#chat prompt template with variables

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful {domain} expert."),
    ("human", "explain in simple terms what is {topic}")
])

prompt = chat_template.invoke({"domain": "medical", "topic": "diabetes"})
print(prompt)