import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# --- Initialize ---
chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="company_db"))
collection = chroma_client.get_or_create_collection(name="companies")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Or any open-source embedding model

# --- Helper to embed and upsert ---
def upsert_company(company):
    text_data = f"{company['name']}, {company['location']}, {company['industry']}, {company['description']}"
    embedding = embedder.encode(text_data).tolist()

    collection.upsert(
        documents=[text_data],
        embeddings=[embedding],
        ids=[company["id"]],
        metadatas=[company]
    )

# --- UI ---
st.set_page_config(page_title="Company Info Manager", layout="centered")
st.title("🏢 Company Info Editor")

# Get existing companies
company_ids = [item for item in collection.get()["ids"]]

selected_id = st.selectbox("Select a Company", options=["New Company"] + company_ids)

company_data = {
    "id": "",
    "name": "",
    "location": "",
    "industry": "",
    "description": ""
}

# Load if existing
if selected_id != "New Company":
    data = collection.get(ids=[selected_id])
    if data and data["metadatas"]:
        company_data = data["metadatas"][0]

# Form to update
with st.form("company_form"):
    company_data["id"] = st.text_input("Company ID", value=company_data["id"], disabled=(selected_id != "New Company"))
    company_data["name"] = st.text_input("Company Name", value=company_data["name"])
    company_data["location"] = st.text_input("Location", value=company_data["location"])
    company_data["industry"] = st.text_input("Industry", value=company_data["industry"])
    company_data["description"] = st.text_area("Description", value=company_data["description"])

    submitted = st.form_submit_button("💾 Save / Update")
    if submitted:
        if company_data["id"].strip() == "":
            st.error("Company ID is required!")
        else:
            upsert_company(company_data)
            st.success("✅ Company information saved and embedding updated!")

# Optional: View full collection
with st.expander("📄 View All Companies"):
    docs = collection.get()
    for i, meta in enumerate(docs["metadatas"]):
        st.json(meta)
