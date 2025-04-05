import streamlit as st
from groq import Groq

# Initialize Groq LLM
groq_client = Groq(api_key="REMOVED_GROQ_API_KEY")

def analyze_clause(clause):
    prompt = f"""
You are a legal contract analyst working for an IT services company called ConsultAdd.

You are given a clause from a contract or RFP. Your job is to:
1. Determine whether the clause poses any legal or business risk to ConsultAdd.
2. If the clause seems biased or risky, suggest a more balanced version of the clause.
3. Highlight why the clause could be risky (e.g., unilateral termination, ambiguous liability, etc.)

Output your answer in the following format:
---
Clause: "{clause}"

Risk Level: [Low / Medium / High]

Reason: <explain why it's risky>

Suggested Rewrite: <more balanced and fair version of the clause>
---
"""
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# UI
st.title("📄 Contract Risk Analyzer")

clause_input = st.text_area("Paste a clause from the contract or RFP")

if st.button("🔍 Analyze Clause"):
    if clause_input.strip():
        result = analyze_clause(clause_input)
        st.markdown(result)
    else:
        st.warning("Please enter a clause to analyze.")
