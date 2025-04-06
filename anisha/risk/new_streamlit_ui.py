import os

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from rfp_agent import RFPAnalysisAgent
from typing_extensions import List, TypedDict
from langchain_groq import ChatGroq
from utils.data_ingestion import extract_text_from_pdf
from utils.prompts import COMPLIANCE_PROMPT, MANDATE_PROMPT
from utils_2 import analyze_clause_bias, analyze_rfp_document, suggest_balanced_clause

# Set wide mode and custom title
st.set_page_config(page_title="ConsultAdd RFP Analyzer", page_icon="📄", layout="wide")

st.markdown("""
    <style>
        .reportview-container {
            background-color: #F7F9FC;
        }
        .sidebar .sidebar-content {
            background-color: #EFF2F5;
        }
        h1, h2, h3 {
            color: #0D3B66;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📄 ConsultAdd RFP Risk Analyzer")
st.markdown("##### *AI-powered tool to ensure compliance, analyze risks, and assist in RFP submissions.*")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    risk_threshold = st.slider("Risk Sensitivity", 0.0, 1.0, 0.7, help="Lower values = more risk items detected")
    show_suggestions = st.checkbox("💡 Show Balanced Alternatives", True)
    show_entities = st.checkbox("📍 Highlight Named Entities", True)
    st.markdown("---")
    uploaded_file = st.file_uploader("📤 Upload RFP Document (PDF)", type=["pdf"])

# Init agent
agent = RFPAnalysisAgent()

# If file uploaded
if uploaded_file:
    temp_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("🔍 Analyzing your RFP..."):
        try:
            analysis = analyze_rfp_document(temp_path)

            if analysis:
                # Main UI with tabs
                tab1, tab2, tab3 = st.tabs(
                    [
                        "📋 Eligibility and Compliance Check",
                        "🧾 Submission Checklist",
                        "⚖️ Contract Risk Analysis",
                    ]
                )

                with tab1:

                    class State(TypedDict):
                        question: str
                        context: List[Document]
                        answer: str
                        compliance_checker: str

                    load_dotenv()

                    # llm = ChatOpenAI(
                    #     model="gpt-4o-mini",
                    #     temperature=0,
                    #     api_key=os.environ.get("OPENAI_API_KEY"),
                    # )
                    llm = ChatGroq(
                        model="qwen-2.5-32b",
                        temperature=0.25,
                        api_key=os.environ.get("GROQ_API_KEY"),
                    )
                    embeddings_model = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-mpnet-base-v2"
                    )

                    def chunk_and_embed(text: str):
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=512, chunk_overlap=20
                        )
                        docs = splitter.create_documents([text])
                        texts = [doc.page_content for doc in docs]
                        vectors = embeddings_model.embed_documents(texts)
                        return texts, vectors

                    def find_similar_chunks(query: str, texts, vectors, top_k=5):
                        query_vec = embeddings_model.embed_query(query)
                        similarities = np.dot(vectors, query_vec) / (
                            np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vec)
                            + 1e-10
                        )
                        top_indices = np.argsort(similarities)[::-1][:top_k]
                        return [texts[i] for i in top_indices]

                    def retrieve(state: State):
                        question = state["question"]

                        rfp_text = extract_text_from_pdf(uploaded_file)
                        all_chunks, all_vectors = chunk_and_embed(rfp_text)

                        retrieved_docs = find_similar_chunks(
                            question, all_chunks, np.array(all_vectors)
                        )

                        return {"context": retrieved_docs}

                    def compliance_check(state):
                        context = "\n\n".join(state["context"])
                        messages = [
                            (
                                "system",
                                f"You are a legal auditor with the knowledge of{state['compliance_checker']} ",
                            ),
                            (
                                "developer",
                                f"{COMPLIANCE_PROMPT}\n\nContext:\n{context}",
                            ),
                        ]
                        response = llm.invoke(messages)
                        return {"compliance_checker": response}

                    def analyze_rfp(state: State):
                        context = "\n\n".join(state["context"])
                        # query = state["question"]
                        messages = [
                            (
                                "system",
                                "You are an legal auditor answers yes or now if the things in mandate prompt are available in context or not",
                            ),
                            (
                                "developer",
                                f"{MANDATE_PROMPT}\n\nContext:\n{context}",
                            ),
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

                    if uploaded_file is not None:
                        with st.spinner("Extracting and analyzing..."):

                            query = "Years of Experience in Temporary staffing, Company Length of Existence, W-9 Form, qualifications, certifications, licenses"

                            result = graph.invoke(
                                {
                                    "question": query,
                                    "context": [],
                                    "answer": "",
                                    "compliance_checker": "",
                                }
                            )

                        st.success("Analysis complete!")
                        st.markdown(
                            result["compliance_checker"].content, unsafe_allow_html=True
                        )
                        st.markdown("-----------------------------")
                        st.markdown(result["answer"].content, unsafe_allow_html=True)

                with tab2:
                    st.subheader("🧾 Submission Requirements Checklist")
                    st.markdown("Auto-extracted checklist from the RFP:")
                    checklist_items = [
                        "Max 10 pages (Arial, size 11)",
                        "Include TOC and section headers",
                        "Attach Company Registration & Tax Forms"
                    ]
                    for item in checklist_items:
                        st.checkbox(item, value=False)
                
                with tab3:
                    st.subheader("⚖️ Risky Clauses & Suggestions")

                    # Initialize risk levels
                    risk_levels = {'High': 0, 'Medium': 0, 'Low': 0}
                    
                    # Analyze all requirements and count risk levels
                    for req in analysis['requirements']:
                        findings = analyze_clause_bias(req['text'])
                        for finding in findings:
                            risk_levels[finding['risk_level']] += 1

                    # Display Risk Summary
                    st.markdown("### 🔍 Risk Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🔴 High Risk Items", risk_levels["High"])
                    with col2:
                        st.metric("🟡 Medium Risk Items", risk_levels["Medium"])
                    with col3:
                        st.metric("🟢 Low Risk Items", risk_levels["Low"])

                    st.markdown("---")

                    # Loop through each requirement and show detailed risks
                    for req in analysis['requirements']:
                        biased_findings = analyze_clause_bias(req['text'])
                        
                        if biased_findings:
                            with st.expander(f"🚨 Risk in Requirement {req['id']}"):
                                st.markdown("**🔹 Original Clause:**")
                                st.write(req['text'])

                                for finding in biased_findings:
                                    color = {
                                        'High': '🔴',
                                        'Medium': '🟡',
                                        'Low': '🟢'
                                    }.get(finding['risk_level'], '⚪')

                                    st.markdown(f"**Risk Level:** {color} {finding['risk_level']}")
                                    st.markdown(f"**Issue Type:** `{finding['type'].replace('_', ' ').title()}`")

                                    # Show suggestion if enabled
                                    if show_suggestions:
                                        st.markdown("**💡 Suggested Balanced Alternative:**")
                                        st.code(suggest_balanced_clause(finding), language='markdown')

                                        st.button("📋 Copy", key=f"copy_{hash(req['text'])}_{hash(finding['type'])}")

                                    # Feedback option
                                    st.radio(
                                        "Was this helpful?",
                                        ["Yes", "No", "Partially"],
                                        key=f"feedback_{hash(req['text'])}_{hash(finding['type'])}"
                                    )


        except Exception as e:
            st.error(f"🚫 Error: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

# Footer
st.markdown("---")
footer1, footer2 = st.columns([1, 2])
with footer1:
    st.markdown("🚀 Powered by **ConsultAdd AI**")
with footer2:
    st.markdown("Crafted with ❤️ by **Homer’s Hackers**")
