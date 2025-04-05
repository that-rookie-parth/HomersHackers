import os

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.data_ingestion import extract_text_from_pdf

load_dotenv()


llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.environ.get("GROQ_API_KEY"))
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


def chunk_and_embed(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
    docs = splitter.create_documents([text])
    texts = [doc.page_content for doc in docs]
    vectors = embeddings_model.embed_documents(texts)
    return texts, vectors


def find_similar_chunks(query: str, texts, vectors, top_k=5):
    query_vec = embeddings_model.embed_query(query)
    similarities = np.dot(vectors, query_vec) / (
        np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vec) + 1e-10
    )
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [texts[i] for i in top_indices]


def analyze_rfp(relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    query = "Years of Experience in Temporary staffing, W-9 Form, qualifications, certifications, licenses"
    messages = [
        (
            "system",
            "You are a helpful assistant. Given a context from an RFP document, check if the following information is available:\n"
            "- Is there any mention of 'Years of Experience in Temporary staffing'? If yes, is the experience **less than 7 years**?\n"
            "- Is there any mention of a 'W-9 Form'?\n"
            "- Is there any mention of Insurance Certificates?\n"
            "- Is there any mention of 'Company Length of Existence'?\n"
            "- Is there any mention of 'Licenses'?\n"
            "Look for the specific keywords and do not mix up"
            "Respond with clear yes/no answers after complete analysis, and mark missing requirements clearly in red using markdown.",
        ),
        ("developer", f"{query}\n\nContext:\n{context}"),
    ]
    return llm.invoke(messages)


st.set_page_config(page_title="RFP Analyzer", layout="centered")
st.title("📄 RFP PDF Analyzer")

uploaded_file = st.file_uploader("Upload your RFP PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting and analyzing..."):
        rfp_text = extract_text_from_pdf(uploaded_file)
        chunks, vectors = chunk_and_embed(rfp_text)
        top_chunks = find_similar_chunks(
            "Years of Experience, W-9 Form, qualifications, certifications, licenses",
            chunks,
            np.array(vectors),
        )
        response = analyze_rfp(top_chunks)

    st.success("Analysis complete!")
    st.subheader("✅ AI Assistant's Evaluation")
    st.markdown(response.content.strip())
