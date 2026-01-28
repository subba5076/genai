from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

response = client.chat_completion(
    model="meta-llama/Llama-3.2-1B-Instruct",
    messages=[{"role": "user", "content": "What is the capital of India?"}],
    max_tokens=50
)

print(response.choices[0].message["content"])