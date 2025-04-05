import os

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.data_ingestion import extract_text_from_pdf

load_dotenv()


llm = ChatGroq(
    model="qwen-2.5-32b", temperature=0.25, api_key=os.environ.get("GROQ_API_KEY")
)
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
    query = "Years of Experience in Temporary staffing, Company Length of Existence, W-9 Form, qualifications, certifications, licenses"
    messages = [
        (
            "system",
            """You are a highly precise assistant helping evaluate RFP documents. Your job is to locate and report on specific information ONLY if mentioned explicitly in the context.

            When analyzing, follow these rules:
            - Look **only** for exact phrases or close variations. Do NOT infer or generalize.
            - Do not confuse **Temporary Staffing** with **Company Length of Existence**.

            Criteria to check:
            1. Is there any mention of **Years of Experience in Temporary staffing**? If yes, is the experience required less than 7 years?
            2. Is there any mention of a **W-9 Form**?
            3. Is there any mention of **Insurance Certificates**?
            4. Is there any mention of **Company Length of Existence** (how long the company has existed)?
            5. Is there any mention of **Licenses, Certifications, or Registrations**?

            Your response should answer each question clearly in Yes or No, in markdown format.
            """,
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
            "Years of Experience in Temporary staffing, Company Length of Existence, W-9 Form, qualifications, certifications, licenses",
            chunks,
            np.array(vectors),
        )
        response = analyze_rfp(top_chunks)

    st.success("Analysis complete!")
    st.markdown(response.content.strip(), unsafe_allow_html=True)
