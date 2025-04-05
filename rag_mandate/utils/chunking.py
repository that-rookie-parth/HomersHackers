from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


def chunk_and_embed(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
    docs = splitter.create_documents([text])
    texts = [doc.page_content for doc in docs]
    vectors = embeddings_model.embed_documents(texts)
    return texts, vectors
