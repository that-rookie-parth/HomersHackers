import os

import fitz
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools.render import render_text_description
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from rfp_agent import RFPAnalysisAgent
from sentence_transformers import SentenceTransformer, util
from typing_extensions import List, TypedDict
from langchain_groq import ChatGroq
from utils.data_ingestion import extract_text_from_pdf
from utils.prompts import COMPLIANCE_PROMPT, MANDATE_PROMPT
from utils.risky_clause import analyze_clause_risk
from utils_2 import analyze_clause_bias, analyze_rfp_document, suggest_balanced_clause

AUDIT_INFO = {
    "Legal and Regulatory Info": {
        "State of Incorporation": {
            "Available": True,
            "Details": "Delaware",
        },
        "Business Structure": {
            "Available": True,
            "Details": "LLC",
        },
        "State Registration Number": {
            "Available": True,
            "Details": "SRN-DE-0923847",
        },
        "DUNS Number": {
            "Available": True,
            "Details": "07-842-1490",
        },
        "CAGE Code": {"Available": True, "Details": "8J4T7"},
        "SAM.gov Registration": {
            "Available": True,
            "Details": "Registered on 03/01/2022",
        },
    },
    "Experience and Capabilities": {
        "Company Age": {
            "Available": True,
            "Details": "9 years",
        },
        "Staffing Experience": {
            "Available": True,
            "Details": "7 years",
        },
        "Services Offered": {
            "Available": True,
            "Details": "Administrative, IT, legal, and credentialing staffing",
        },
        "NAICS Codes": {
            "Available": True,
            "Details": "Used for federal procurement (e.g., Temporary Help Services); specific codes not listed",
        },
    },
    "Compliance and Documentation": {
        "Certificate of Insurance": {
            "Available": True,
            "Details": "Available",
        },
        "W-9 Form (with Tax ID)": {
            "Available": True,
            "Details": "Includes Tax ID",
        },
        "Licenses": {
            "Available": True,
            "Details": "Texas Employment Agency license",
        },
        "Bank Letter of Creditworthiness": {
            "Available": False,
            "Details": "Not available",
        },
        "MBE / DBE / HUB Certification": {
            "Available": False,
            "Details": "Not certified",
        },
    },
}


def highlight_text_in_pdf(pdf_path, output_path, highlights):
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        for text in highlights:
            text_instances = page.search_for(text)
            for inst in text_instances:
                page.add_highlight_annot(inst)
    doc.save(output_path)


# Set wide mode and custom title
st.set_page_config(page_title="ConsultAdd RFP Analyzer", page_icon="📄", layout="wide")

st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

st.title("📄 ConsultAdd RFP Risk Analyzer")
st.markdown(
    "##### *AI-powered tool to ensure compliance, analyze risks, and assist in RFP submissions.*"
)

# Sidebar settings
with st.sidebar:
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
                tab1, tab2, tab3, tab4 = st.tabs(
                    [
                        "Eligibility and Compliance Check",
                        "Submission Checklist",
                        "Contract Risk Analysis",
                        "Actionable Insights",
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

                    def format_audit_info(audit_info):
                        formatted = []
                        for section, items in audit_info.items():
                            formatted.append(f"## {section}")
                            for item, details in items.items():
                                available = (
                                    "✅ Yes" if details["Available"] else "❌ No"
                                )
                                formatted.append(f"- **{item}**: {available}")
                                formatted.append(f"  - Details: {details['Details']}")
                            formatted.append("")  # Add space between sections
                        return "\n".join(formatted)

                    def compliance_check(state):
                        context = "\n\n".join(state["context"])
                        audit_info = AUDIT_INFO
                        audit_info_str = format_audit_info(audit_info)

                        messages = [
                            (
                                "system",
                                f"You are a legal auditor with the knowledge of {state['compliance_checker']}",
                            ),
                            (
                                "developer",
                                f"{COMPLIANCE_PROMPT}\n\nContext:\n{context}\n\nAudit Information:\n{audit_info_str}",
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

                        st.markdown("### 📚 Sources Used")
                        for i, source in enumerate(result["context"], 1):
                            st.markdown(f"**Source {i}:**\n```text\n{source}\n```")

                with tab2:
                    st.subheader("🧾 Submission Requirements Checklist")
                    st.markdown("Auto-extracted checklist from the RFP:")
                    checklist_items = [
                        "Max 10 pages (Arial, size 11)",
                        "Include TOC and section headers",
                        "Attach Company Registration & Tax Forms",
                    ]
                    for item in checklist_items:
                        st.checkbox(item, value=False)
                
                with tab3:
                    st.subheader("⚖️ Risky Clauses & Suggestions")

                    rfp_text = extract_text_from_pdf(uploaded_file)
                    all_chunks, all_vectors = chunk_and_embed(rfp_text)

                    tools = [analyze_clause_risk]

                    template = """
                    You are a contract risk assessment agent. You have tools to analyze text chunks.

                    Use this format:

                    Question: a contract clause to evaluate
                    Thought: reason about risk level
                    Action: the action to take, should be one of [{tool_names}]
                    Action Input: the clause
                    Observation: the result
                    ... repeat Thought/Action until done ...
                    Thought: I now know the final answer
                    Final Answer: return all results in a list of JSON per clause

                    Begin!

                    Question: {input}
                    Thought: {agent_scratchpad}
                    """

                    prompt = PromptTemplate.from_template(template).partial(
                        tools=render_text_description(tools),
                        tool_names=", ".join([t.name for t in tools]),
                    )

                    agent_chain = (
                        {
                            "input": lambda x: x["input"],
                            "agent_scratchpad": lambda x: format_log_to_str(
                                x["agent_scratchpad"]
                            ),
                        }
                        | prompt
                        | llm
                    )

                    similar_clauses = find_similar_chunks(
                        query="biased or risky contract clauses",
                        texts=all_chunks,
                        vectors=np.array(all_vectors),
                        top_k=5,
                    )

                    intermediate_steps = []
                    final_results = []

                    for clause in similar_clauses:
                        agent_step = agent_chain.invoke(
                            {
                                "input": clause,
                                "agent_scratchpad": intermediate_steps,
                            }
                        )

                        st.success("✅ Final Analysis: ")
                        st.markdown(agent_step.content)
                with tab4:
                    with st.spinner("Analyzing clauses for suggestions..."):

                        rfp_text = extract_text_from_pdf(uploaded_file)

                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=512, chunk_overlap=20
                        )

                        chunks = [
                            chunk.page_content.strip()
                            for chunk in splitter.create_documents([rfp_text])
                        ]

                        model = SentenceTransformer("all-MiniLM-L6-v2")
                        chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

                        reference_clauses = [
                            "The Contractor may offer additional maintenance services if eligible.",
                            "Future support services may enhance bid eligibility.",
                            "Optional maintenance support may be provided to enhance bid competitiveness.",
                        ]
                        ref_embeddings = model.encode(
                            reference_clauses, convert_to_tensor=True
                        )

                        threshold = 0.4
                        matched_chunks = []
                        for chunk, embedding in zip(chunks, chunk_embeddings):
                            cosine_scores = util.cos_sim(embedding, ref_embeddings)
                            max_score = cosine_scores.max().item()
                            if max_score >= threshold:
                                matched_chunks.append((chunk, max_score))

                        improvement_prompt = PromptTemplate(
                            input_variables=["clause"],
                            template="""
                                You are a proposal expert. Here is a contract clause that was identified as beneficial:
            
                                "{clause}"
            
                                Summarize the clause in simple language and explain what actions I can take to improve my proposal based on this clause.
                                Provide clear, short, and actionable advice.
                                """,
                        )
                        chain = improvement_prompt | llm | StrOutputParser()

                        if matched_chunks:
                            for clause, score in matched_chunks:
                                advice = chain.invoke({"clause": clause})

                                st.markdown(f"> {clause}")
                                st.markdown("**💡 Proposal Advice:**")
                                st.success(advice.strip())
                                st.markdown("---")
                        else:
                            st.info(
                                "No clauses were similar enough to the reference beneficial clauses."
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
