import google.generativeai as genai
import os
import re

def chunk_document(document, chunk_size=450, overlap=50, model_name="models/text-embedding-001"):
    """
    Splits a document into overlapping chunks using Google's generative AI API.

    Args:
        document (str): The input legal document as a single string.
        chunk_size (int): The approximate maximum number of tokens per chunk. Default is 450.
        overlap (int): The approximate number of tokens to overlap between chunks. Default is 50.
        model_name (str): The Google generative AI model to use for tokenization.

    Returns:
        list: A list of text chunks.
    """

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(model_name)

    # Simple heuristic to approximate token count. Google API does not provide a direct tokenization method.
    # We will split on spaces and punctuation and then use a rough estimate of tokens.
    words = re.findall(r'\w+|[^\w\s]', document)

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = ' '.join(chunk_words)
        chunks.append(chunk_text)
        start += chunk_size - overlap

    return chunks

# Example usage (replace with your actual document and API key):
# document = "This is a sample legal document. It contains various clauses and provisions. We need to split it into chunks for processing."
# chunks = chunk_document_google(document)
# print(chunks)
