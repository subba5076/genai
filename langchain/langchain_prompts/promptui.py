from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from langchain_core.prompts import PromptTemplate,load_prompt
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs=dict(temperature=0.7)
)
chat = ChatHuggingFace(llm=llm)

st.header("summarizer Using TinyLlama-1.1B-Chat-v1.0 Model")

paper_input = st.selectbox("Select Research Paper Name", ["Select...", "Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

# # Define the prompt template
# template = PromptTemplate(
#     template = """
#     You are an expert in summarizing research papers.

#     Summarize the paper titled "{paper_input}" in a {style_input} style with a {length_input} explanation.

#     Instructions:
#     1. Mathematical Details:
#     - Include relevant mathematical equations if present in the paper.
#     - Explain the mathematical concepts using simple, intuitive code snippets where applicable.

#     2. Analogies:
#     - Use relatable analogies to simplify complex ideas.
#     If any of the above information is not available in the paper, respond with "Insufficient information available" instead of guessing.
#     Ensure the summary is clear, accurate, and aligned with the selected style and length.
#     """, input_variables=["paper_input", "style_input", "length_input"]
# )   
template = load_prompt("template.json")




if st.button("Generate Response"):
    # res = chat.invoke(prompt)
    # st.write(res.content)
    chain = template | chat
    res = chain.invoke({
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input
    })
    st.write(res.content)
   