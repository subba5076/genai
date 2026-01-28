from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
load_dotenv()

embeding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
documents = ["trackjum is a music company found by subrahmanya",
             "subrahmanyas last name is nayak",
                "to a vector representation"]

query = "last name of the founder of trackjum"

doc_embeding = embeding.embed_documents(documents)
query_embeding = embeding.embed_query(query)

scores = cosine_similarity([query_embeding], doc_embeding)[0]

index,score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(f"Most similar document: {documents[index]} with score {score}")