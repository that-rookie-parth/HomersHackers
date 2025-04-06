import logging
import requests
import fitz  # PyMuPDF
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

logging.basicConfig(level=logging.INFO)

try:
    import fitz  # PyMuPDF
    # from sentence_transformers import SentenceTransformer, util
    from sentence_transformers import SentenceTransformer

except ImportError as e:
    logging.error(f"Import error: {e}")
    logging.info("Please make sure all required packages are installed correctly")
    raise
def extract_text(pdf_path):
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found: {pdf_path}")
        return None
        
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {str(e)}")
        return None

def semantic_chunks(text, model_name="sentence-transformers/all-MiniLM-L6-v2", threshold=0.7):
    if not text:
        logging.error("No text provided for chunking")
        return []
        
    try:
        model = SentenceTransformer(model_name)
        sentences = text.split("\n")
        if not sentences:
            return []
            
        embeddings = model.encode(sentences, convert_to_tensor=True)
        
        chunks, current_chunk = [], []
        for i in range(len(sentences)-1):
            sim = model.similarity(embeddings[i], embeddings[i+1]).item()
            current_chunk.append(sentences[i])
            if sim < threshold:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
        current_chunk.append(sentences[-1])
        chunks.append("\n".join(current_chunk))
        return chunks
    except Exception as e:
        logging.error(f"Error in semantic chunking: {str(e)}")
        return []

def setup_ner_pipeline():
    model = AutoModelForTokenClassification.from_pretrained("nlpaueb/legal-bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Run on a chunk
def extract_ner_entities(text, ner_pipeline):
    return ner_pipeline(text)

def classify_clause(text, api_key):
    prompt = f"""
You are a legal contract analyst working for an IT services company called ConsultAdd.

You are given a clause from a contract or RFP. Your job is to:
1. Determine whether the clause poses any legal or business risk to ConsultAdd.
2. If the clause seems biased or risky, suggest a more balanced version of the clause.
3. Highlight why the clause could be risky (e.g., unilateral termination, ambiguous liability, etc.)

Clause:
\"\"\"{text}\"\"\"
---

Risk Level: [Low / Medium / High]

Reason: <explain why it's risky>

Suggested Rewrite: <more balanced and fair version of the clause>
---

Answer:"""

    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "llama3-70b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }
    )
    return res.json()['choices'][0]['message']['content']

# Load Legal NER pipeline

def detect_risks(text):
    risk_keywords = ["unilateral", "termination without cause", "indemnify", "no liability", "sole discretion"]
    risky = any(keyword.lower() in text.lower() for keyword in risk_keywords)
    return "⚠️ Risky Clause" if risky else "✔️ OK"
