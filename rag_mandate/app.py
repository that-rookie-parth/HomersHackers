import os

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from data_ingestion import extract_text_from_pdf
from rag_mandate.utils.mmr import mmr_select

load_dotenv()


llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b", api_key=os.environ.get("GROQ_API_KEY")
)


def find_similar_chunks(query: str, texts, vectors, top_k=5, lambda_param=0.5):
    query_vec = embeddings_model.embed_query(query)
    return mmr_select(
        query_vec, np.array(vectors), texts, top_k=top_k, lambda_param=lambda_param
    )


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
            "Respond with clear yes/no answers after complete analysis, and mark missing requirements clearly in red using markdown.",
        ),
        ("developer", f"{query}\n\nContext:\n{context}"),
    ]
    return llm.invoke(messages)


# Streamlit UI
st.set_page_config(page_title="RFP Analyzer", layout="centered")
st.title("📄 RFP PDF Analyzer")
st.markdown("Upload an RFP PDF file to analyze if it meets key criteria.")

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
    st.markdown(response.content.strip())
