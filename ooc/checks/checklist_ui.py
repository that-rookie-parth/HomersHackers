import streamlit as st
import os
from io import BytesIO
import numpy as np
import tempfile

from working_checklist import (
    extract_text_from_pdf,
    chunk_and_embed,
    find_similar_chunks,
    extract_format_requirements,
    extract_attachments_and_forms,
    extract_forms_to_submit,
    llm,
    suggest_eligibility_actions,
    format_log_to_str,
    render_text_description,
)
from langchain_core.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool
from langchain.agents.format_scratchpad import format_log_to_str

# Set Streamlit page config
st.set_page_config(page_title="📑 RFP Checklist Generator", layout="wide")
st.title("📑 AI-Powered RFP Submission Checklist Generator")

tabs = st.tabs(["📤 Upload & Process", "🧠 Retrieved Context", "✅ Checklist Output", "🔍 Agent Logs"])

rfp_text = ""

# Add this near the top of the file, after the imports
if 'intermediate_steps' not in st.session_state:
    st.session_state.intermediate_steps = []

with tabs[0]:
    st.subheader("📤 Upload RFP PDF")
    uploaded_file = st.file_uploader("Upload your RFP document (PDF)", type=["pdf"])
    
    if uploaded_file:
        with st.spinner("Extracting text from PDF..."):
            temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            rfp_text = extract_text_from_pdf(temp_path)

        if rfp_text:
            st.success("✅ Text extracted successfully!")
            st.text_area("Preview Extracted Text:", value=rfp_text[:1500], height=300)

            with st.spinner("🔍 Chunking + Embedding..."):
                chunks, vectors = chunk_and_embed(rfp_text)

            question = "What are the submission requirements including page limits, formatting, and required attachments?"
            retrieved_chunks = find_similar_chunks(question, chunks, np.array(vectors))
            context = "\n".join(retrieved_chunks)
            st.session_state["context"] = context

            st.success("✅ Document processed! Move to the next tab 👉")

with tabs[1]:
    st.subheader("🧠 Retrieved Context")
    if "context" in st.session_state:
        st.code(st.session_state["context"][:1000], language="markdown")
    else:
        st.warning("Upload a PDF in Tab 1 first.")

with tabs[2]:
    st.subheader("✅ Final Checklist Output")
    if "context" in st.session_state:
        context = st.session_state["context"]
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

        agent_chain = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
            }
            | prompt
            | llm
        )

        intermediate_steps = []

        while True:
            agent_step = agent_chain.invoke({
                "input": context,
                "agent_scratchpad": st.session_state.intermediate_steps,
            })

            if isinstance(agent_step, AgentAction):
                tool_name = agent_step.tool
                tool_to_use = next(t for t in tools if t.name == tool_name)
                observation = tool_to_use.func(agent_step.tool_input)
                st.session_state.intermediate_steps.append((agent_step, str(observation)))

            elif isinstance(agent_step, AgentFinish):
                output = agent_step.return_values["output"]
                st.success("🎯 Final Checklist:")
                try:
                    checklist_data = eval(output)
                    for key, val in checklist_data.items():
                        st.markdown(f"**{key}**")
                        if isinstance(val, list):
                            for item in val:
                                st.checkbox(item, value=False)
                        else:
                            st.write(val)
                except:
                    st.code(output)
                break
    else:
        st.warning("Upload and process an RFP first.")

    with st.expander("🚧 Not Eligible? See Actionable Fixes"):
        suggestions = suggest_eligibility_actions(rfp_text)
        if "error" in suggestions:
            st.error(suggestions["error"])
        else:
            for item in suggestions:
                st.markdown(f"**📌 Requirement:** {item['Requirement']}")
                st.markdown(f"**❓ Reason:** {item['Reason']}")
                st.markdown(f"**✅ Action:** {item['Action']}")
                st.markdown(f"[🔗 Link]({item['Link']})")
                st.divider()

    with tabs[3]:
        st.subheader("🔍 Agent Logs (ReAct Reasoning)")
        if "context" in st.session_state and st.session_state.intermediate_steps:
            for i, (action, obs) in enumerate(st.session_state.intermediate_steps):
                st.markdown(f"**Step {i+1}**")
                st.markdown(f"🔧 Tool: `{action.tool}`")
                st.markdown(f"📥 Input: `{action.tool_input}`")
                st.markdown(f"📤 Output: `{obs}`")
        else:
            st.info("Logs will show after processing the RFP.")