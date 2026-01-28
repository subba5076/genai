from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
from dotenv import load_dotenv
load_dotenv()

#chat prompt template with message placeholder
chat_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}"),
])

chat_history = []
#load chat history
with open("chat_history.txt") as f:
   chat_history.extend(f.readlines())
#print(chat_history)

#create prompt with chat history
prompt = chat_template.invoke({"chat_history": chat_history, "query": "preliminary diagnosis"})
print(prompt)