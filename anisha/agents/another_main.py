import os
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from dotenv import load_dotenv

import pandas as pd
import pdfplumber

load_dotenv()
from typing import List, Union
from langchain.agents import tool
from langchain.tools.render import render_text_description
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.agents.output_parsers.react_single_input import (
    ReActSingleInputOutputParser,
)
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool, tool

from callbacks import AgentCallbackHandler
from langchain.agents.format_scratchpad import format_log_to_str



# Step 1: Define LLM + Embedding Model
llm = ChatGroq(
    model="qwen-2.5-32b",
    temperature=0.25,
    api_key=os.environ.get("GROQ_API_KEY"),
)
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Step 2: Chunk + Embed Function
def chunk_and_embed(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
    docs = splitter.create_documents([text])
    texts = [doc.page_content for doc in docs]
    vectors = embeddings_model.embed_documents(texts)
    return texts, vectors

# Step 3: Retrieval Function
def find_similar_chunks(query: str, texts, vectors, top_k=5):
    query_vec = embeddings_model.embed_query(query)
    similarities = np.dot(vectors, query_vec) / (
        np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vec) + 1e-10
    )
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [texts[i] for i in top_indices]


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


@tool
def extract_attachments_and_forms(text: str) -> dict:
    """
    Extracts required attachments and submission forms from RFP text and returns them
    in a structured dictionary format.
    """
    prompt = PromptTemplate.from_template("""
    From the RFP text below, list all required attachments and submission forms.
    
    - Attachments: like resumes, technical proposals, letters
    - Forms: like Form A-1, Disclosure Form, W-9, etc.

    Return result as a dictionary with keys: "Attachments" and "Forms".

    RFP Text:
    {text}

    Output:
    """)
    response = llm.predict(prompt.format(text=text))
    try:
        return eval(response)
    except:
        return {"error": "Failed to parse response", "raw": response}

@tool
def extract_forms_to_submit(text: str) -> list:
    """Extracts a list of forms that need to be submitted with the proposal."""
    return ["Form A-1", "Disclosure Form", "W-9 Form"]



@tool
def extract_attachments_and_forms(text: str) -> dict:
    """
    Extracts required attachments and submission forms from RFP text and returns them
    in a structured dictionary format.
    """
    prompt = PromptTemplate.from_template("""
    From the RFP text below, list all required attachments and submission forms.
    
    - Attachments: like resumes, technical proposals, letters
    - Forms: like Form A-1, Disclosure Form, W-9, etc.

    Return result as a dictionary with keys: "Attachments" and "Forms".

    RFP Text:
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


def find_tool_by_id(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
        raise ValueError(f"Tool with {tool_name} not found!")


# Main RAG + ReAct
if __name__ == "__main__":
    rfp_path = "ELIGIBLE_RFP_2.pdf"
    rfp_text = extract_text_from_pdf(rfp_path)

    # Step 4: Chunk + Embed
    all_chunks, all_vectors = chunk_and_embed(rfp_text)

    # Step 5: Retrieve relevant context
    question = "What are the submission requirements including page limits, formatting, and required attachments?"
    retrieved_chunks = find_similar_chunks(question, all_chunks, np.array(all_vectors))
    context = "\n".join(retrieved_chunks)

    print("🔍 Retrieved Context for ReAct Agent:\n", context[:500])

    # Step 6: Run ReAct agent on retrieved context
    tools = [extract_format_requirements, extract_attachments_and_forms, extract_forms_to_submit]

    template = """ 
    You are an expert at reading RFP documents and producing structured submission checklists.

    You have access to the following tools:
    {tools}

    Use this format:

    Question: the user-provided RFP content or question
    Thought: think step-by-step
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result
    ... repeat Thought/Action until you're done ...
    Thought: I now know the final answer
    Final Answer: the final checklist as structured data

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    intermediate_steps = []

    agent_chain = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    while True:
        agent_step = agent_chain.invoke({
            "input": context,
            "agent_scratchpad": intermediate_steps,
        })

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_id(tools, tool_name)
            observation = tool_to_use.func(agent_step.tool_input)
            intermediate_steps.append((agent_step, str(observation)))
            print(f"\n🛠 Tool used: {tool_name}")
            print(f"📥 Input: {agent_step.tool_input}")
            print(f"📤 Output: {observation}")

        elif isinstance(agent_step, AgentFinish):
            print("\n✅ Final Checklist Output:")
            print(agent_step.return_values["output"])
            break
