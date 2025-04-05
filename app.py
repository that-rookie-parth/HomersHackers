import os

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from data_ingestion import extract_text_from_pdf

load_dotenv()

VECTOR_DB_PATH = "./app/vector_store/faiss_index"


def build_vector_store(rfp_text: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=20,
    )
    text_chunks = text_splitter.create_documents([rfp_text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vector_store = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
    vector_store.save_local(VECTOR_DB_PATH)

    return vector_store


def analyze_rfp(vector_store):

    query = "Years of Experience in Temporary staffing, W-9 Form, Must have qualifications, certifications, licenses"

    retrieved_docs = vector_store.similarity_search(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    llm = ChatGroq(
        model="deepseek-r1-distill-llama-70b", api_key=os.environ.get("GROQ_API_KEY")
    )

    messages = [
        (
            "system",
            "You are a helpful assistant. Given a context from an RFP document, check if the following information is available:\n"
            "- Is there any mention of 'Years of Experience in Temporary staffing'? If yes, is the experience **less than 7 years**?\n"
            "- Is there any mention of a 'W-9 Form'?\n\n"
            "- Is there any mention of Insurance Certificates?\n"
            "- Is there any mention of 'Company Length of Existence'?\n"
            "- Is there any mention of 'Licenses'?\n"
            "Respond with clear yes/no answers after complete analysis, and give a warning if something is not available.",
        ),
        ("developer", f"{query}\n\nContext:\n{context}"),
    ]

    return llm.invoke(messages)


st.set_page_config(page_title="RFP Analyzer", layout="centered")

st.title("📄 RFP PDF Analyzer")
st.markdown("Upload an RFP PDF file to analyze if it meets key criteria.")

uploaded_file = st.file_uploader("Upload your RFP PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting and analyzing..."):
        rfp_text = extract_text_from_pdf(uploaded_file)
        vector_store = build_vector_store(rfp_text)
        response = analyze_rfp(vector_store)

    st.success("Analysis complete!")
    markdown_response = f"""
    {response.content.strip()}
    """

    st.markdown(markdown_response)
