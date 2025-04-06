import streamlit as st
import os
from utils import extract_text, semantic_chunks, classify_clause, detect_risks, extract_ner_entities, setup_ner_pipeline

st.title("⚖️ Legal Contract Clause Analyzer")

api_key = st.text_input("Groq API Key", type="password")
uploaded_file = st.file_uploader("Upload Contract PDF")

if uploaded_file and api_key:
    # Save uploaded file temporarily
    temp_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    with st.spinner("Analyzing..."):
        try:
            text = extract_text(temp_path)
            if text:
                chunks = semantic_chunks(text)
                ner = setup_ner_pipeline()

                for chunk in chunks:
                    clause_type = classify_clause(chunk, api_key)
                    if "Clause" in clause_type:
                        st.markdown(f"### 📜 Clause")
                        st.write(chunk)
                        st.markdown(f"**Classification:** `{clause_type.strip()}`")
                        st.markdown(f"**Risk Check:** `{detect_risks(chunk)}`")
                        entities = extract_ner_entities(chunk, ner)
                        if entities:
                            st.markdown("**Entities Found:**")
                            st.json(entities)
                        st.divider()
            else:
                st.error("Could not extract text from the uploaded file")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)