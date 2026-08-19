import openai
import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
from typing import List
from data_ingestion import extract_text_from_pdf, clean_rfp_text
from langchain_huggingface import HuggingFaceEmbeddings

def chunk_text(text, max_tokens=500):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        tokens = tokenizer.encode(" ".join(current_chunk))
        if len(tokens) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks



def get_embedding(text, model="text-embedding-ada-002"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
    embedding_vector = embeddings.embed_query(text)

    
    
    return embedding_vector


def load_checklist(path="data/checklist.json"):
    with open(path, 'r') as f:
        return json.load(f)


import json
from sklearn.metrics.pairwise import cosine_similarity
import openai


def get_top_k_chunks(item_embedding, chunk_embeddings, chunks, k=3):
    sims = cosine_similarity([item_embedding], chunk_embeddings)[0]
    top_indices = sims.argsort()[-k:][::-1]
    return [chunks[i] for i in top_indices]


def evaluate_with_gpt(requirement, relevant_chunks):
    prompt = f"""
You are checking if a document satisfies a specific requirement.

Requirement:
"{requirement}"

Relevant sections from the document:
\"\"\"
{chr(10).join(relevant_chunks)}
\"\"\"

Does the document fulfill the requirement? Respond with:
- "✅ Compliant"
- "⚠️ Partially Compliant"
- "❌ Not Compliant"

Then explain why.

Format:
Status: ...
Reason: ...
"""
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


def run_smart_compliance(pdf_path, checklist_path, output_path="smart_compliance_report.json"):
    print("🔍 Extracting PDF text...")
    
    raw_rfp_text = extract_text_from_pdf(pdf_path)
    cleaned_rfp = clean_rfp_text(raw_rfp_text)

    print("🧱 Chunking document...")
    chunks = chunk_text(cleaned_rfp)

    print("📐 Getting embeddings for chunks...")
    chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

    checklist = load_checklist(checklist_path)
    report = {}

    print("🧠 Evaluating checklist...")
    for key, value in checklist.items():
        print(f"🔹 Checking: {key}")
        item_embedding = get_embedding(key)
        top_chunks = get_top_k_chunks(item_embedding, chunk_embeddings, chunks, k=3)
        result = evaluate_with_gpt(key, top_chunks)
        report[key] = {
            "requirement_value": value,
            "result": result
        }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"\n✅ Compliance report saved to {output_path}")


run_smart_compliance("data/ELIGIBLE_RFP_2.pdf", "data/checklist.json")
