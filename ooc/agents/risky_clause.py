import datetime
import os
from typing import List

import numpy as np
import pdfplumber
from callbacks import AgentCallbackHandler
from dotenv import load_dotenv
from langchain.agents import tool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.schema import AgentAction, AgentFinish
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool
from langchain.tools.render import render_text_description
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

load_dotenv()

llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.25,
    api_key=os.environ.get("GROQ_API_KEY"),
    callbacks=[AgentCallbackHandler()]
)

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


def generate_pdf_report(filepath, data):
    c = canvas.Canvas(filepath, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 50, "📄 RFP Risk Analysis Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 80, f"Generated on: {datetime.datetime.now()}")

    y = height - 120
    for key, value in data.items():
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"{key}:")
        y -= 20
        c.setFont("Helvetica", 10)
        lines = str(value).split("\n")
        for line in lines:
            c.drawString(70, y, line[:100])
            y -= 15
            if y < 50:
                c.showPage()
                y = height - 50
    c.save()

@tool
def analyze_risk_levels(text: str) -> list:
    """
    Analyze each paragraph of the RFP and assign a risk level: High, Medium, or Low.
    Return a list of tuples (paragraph, risk_level).
    """
    prompt = PromptTemplate.from_template("""
    Analyze the risk level of each paragraph in the RFP text provided.
    Classify each one as 'High', 'Medium', or 'Low' risk for vendors.
    Return a Python list of tuples: [(paragraph1, risk_level), (paragraph2, risk_level), ...]

    RFP Text:
    {text}
    Output:
    """)
    response = llm.invoke(prompt.format(text=text))
    return eval(response)

@tool
def compute_vendor_friendly_score(text: str) -> dict:
    """
    Analyze the vendor-friendliness of the RFP and assign a score out of 100.
    Return the score and a short explanation.
    """
    prompt = PromptTemplate.from_template("""
    Analyze the RFP text and rate its vendor-friendliness on a scale of 0 to 100.
    Consider clarity, fairness of terms, length, and required forms.
    
    Return as dictionary: {"score": int, "reason": str}

    RFP Text:
    {text}
    Output:
    """)
    response = llm.invoke(prompt.format(text=text))
    return eval(response)

@tool
def rewrite_high_risk_clauses(text: str) -> List[dict]:
    """
    Rewrites high-risk clauses into balanced ones.
    Returns list of {"original": str, "suggested": str}
    """
    prompt = PromptTemplate.from_template("""
    Given the following RFP paragraphs, rewrite each one to be more balanced and fair to vendors,
    while maintaining legal intent.

    Return list of {"original": str, "suggested": str}

    RFP Text:
    {text}
    Output:
    """)
    response = llm.invoke(prompt.format(text=text))
    return eval(response)


@tool
def analyze_clause_risk(text: str) -> dict:
    """Analyze if a clause contains biased or risky content."""
    prompt = PromptTemplate.from_template("""
    You are a legal analyst AI.

    Analyze the following clause for any potential biases or risks. Return a structured JSON with:
    - risk_level: LOW, MEDIUM, or HIGH
    - bias_reason: Explain why it's considered biased or not
    - recommendation: Suggest what a vendor should watch out for

    Clause:
    {text}

    Output:
    """)

    response = llm.predict(prompt.format(text=text))
    try:
        return eval(response)
    except:
        return {"error": "Failed to parse response", "raw": response}

def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

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

def find_tool_by_id(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with {tool_name} not found!")


# this tool decorator will take the function and create a custom langchain tool out of it
@tool
def extract_format_requirements(text: str) -> dict:
    """
    Extracts document format requirements from RFP text including page limit, font, 
    font size, line spacing, and TOC requirements.
    """
    prompt = PromptTemplate.from_template("""
    From the following RFP text, extract only document format requirements. Include:
    - Page Limit
    - Font
    - Font Size
    - Line Spacing
    - Whether Table of Contents (TOC) is required (True/False)

    Return as a Python dictionary.

    RFP Text:
    {text}

    Output:
    """)
    response = llm.predict(prompt.format(text=text))
    try:
        return eval(response)
    except:
        return {"error": "Failed to parse response", "raw": response}


if __name__ == "__main__":
    rfp_path = "ELIGIBLE_RFP_2.pdf"
    rfp_text = extract_text_from_pdf(rfp_path)
    all_chunks, all_vectors = chunk_and_embed(rfp_text)

    # Define risky clause finder agent
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
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
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
        agent_step = agent_chain.invoke({
            "input": clause,
            "agent_scratchpad": intermediate_steps,
        })

        print("✅ Final Analysis: ")
        print(agent_step.content)
