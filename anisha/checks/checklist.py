import streamlit as st
import fitz  # PyMuPDF
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
# Set Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

# System and user prompts
SYSTEM_PROMPT = """
You are a professional legal assistant AI that specializes in analyzing Requests for Proposals (RFPs), especially government RFPs. Your job is to extract and structure submission-related requirements clearly and concisely.
"""

USER_PROMPT_TEMPLATE = """
Analyze the following RFP document and generate a structured and actionable Submission Checklist. Carefully extract all submission requirements and organize them under these four sections:

1. **Document Formatting Requirements**
   - Page limit, font type/size, line spacing, margin settings
   - Table of contents or indexing requirements
   - File naming conventions (if any)

2. **Attachments and Mandatory Forms**
   - List all specific documents and forms required (e.g., W-9, resumes, insurance certificates, signed affidavits, etc.)

3. **Submission Packaging and Delivery**
   - Number of physical/digital copies
   - Flash drive or USB requirements
   - Envelope labeling instructions
   - Submission address and deadline

4. **Optional but Recommended Additions**
   - Any extra documents that could strengthen the proposal (testimonials, case studies, brochures)

Also include a final section titled **Notes & Ambiguities** to flag any unclear or optional parts of the instructions.

--- RFP Document Below ---

{}
"""

def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF using PyMuPDF"""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text.strip()

def generate_checklist_with_groq(text, model="llama3-8b-8192"):
    """Use Groq to generate checklist from text"""
    prompt = USER_PROMPT_TEMPLATE.format(text[:15000])  # truncate if needed
    chat_completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )
    return chat_completion.choices[0].message.content

# Streamlit App
st.set_page_config(page_title="RFP Checklist Generator (Groq)", layout="wide")
st.title("📋 RFP Submission Checklist Generator (Groq)")
st.markdown("Upload a government RFP PDF and get an auto-generated submission checklist using Groq's LLM.")

uploaded_file = st.file_uploader("Upload your RFP PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("📄 Extracting and analyzing RFP with Groq..."):
        try:
            text = extract_text_from_pdf(uploaded_file)
            checklist = generate_checklist_with_groq(text)
            st.success("✅ Checklist generated successfully!")
            st.markdown("### 📝 Submission Checklist")
            st.markdown(checklist)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
