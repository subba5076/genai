from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer

load_dotenv()

# sentences = ["This is an example sentence", "Each sentence is converted"]
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# embeddings = model.encode(sentences)
# print(embeddings)

# # Example for embed_query

# embeding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# text = "This is an example sentence"
# vector = embeding.embed_query(text)
# print(vector)

# Example for embed_documents
embeding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
document = ["This is an example sentence",
            "Each sentence is converted",
            "to a vector representation"]
vector = embeding.embed_documents(document)
print(str(vector))