import os

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from typing_extensions import List, TypedDict
from utils.data_ingestion import extract_text_from_pdf
from utils.prompts import COMPLIANCE_PROMPT, MANDATE_PROMPT


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    compliance_checker: str

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


def retrieve(state: State):
    question = state["question"]

    rfp_text = extract_text_from_pdf(uploaded_file)
    all_chunks, all_vectors = chunk_and_embed(rfp_text)

    retrieved_docs = find_similar_chunks(question, all_chunks, np.array(all_vectors))

    return {"context": retrieved_docs}


def compliance_check(state):
    messages = [
        (
            "system",
            f"You are a legal auditor with the knowledge of{state['compliance_checker']} ",
        ),
        ("developer", f"{COMPLIANCE_PROMPT}"),
    ]
    response = llm.invoke(messages)
    return {"compliance_checker": response}


def analyze_rfp(state: State):
    context = "\n\n".join(state["context"])
    query = state["question"]
    messages = [
        (
            "system",
            MANDATE_PROMPT,
        ),
        ("developer", f"{query}\n\nContext:\n{context}"),
    ]
    response = llm.invoke(messages)
    return {"answer": response}


builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("compliance_check", compliance_check)
builder.add_node("generate", analyze_rfp)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "compliance_check")
builder.add_edge("compliance_check", "generate")
builder.add_edge("generate", END)


graph = builder.compile()

st.set_page_config(page_title="RFP Analyzer", layout="centered")

uploaded_file = st.file_uploader("Upload your RFP PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting and analyzing..."):

        query = "Years of Experience in Temporary staffing, Company Length of Existence, W-9 Form, qualifications, certifications, licenses"

        result = graph.invoke(
            {"question": query, "context": [], "answer": "", "compliance_checker": ""}
        )

    st.success("Analysis complete!")
    st.markdown(result["compliance_checker"].content, unsafe_allow_html=True)
    st.markdown(result["answer"].content, unsafe_allow_html=True)
