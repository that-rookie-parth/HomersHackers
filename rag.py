import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from data_ingestion import extract_text_from_pdf

load_dotenv()

VECTOR_DB_PATH = "./app/vector_store/faiss_index"


def retrieve_context(question: str):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vector_store = FAISS.load_local(
        VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True
    )

    retrieved_docs = vector_store.similarity_search(question, k=1)

    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    return {"context": context_text}


def preprocess():

    rfp_path = "./data/ELIGIBLE_RFP_1.pdf"

    rfp_text = extract_text_from_pdf(rfp_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=20,
    )

    text_rfp = text_splitter.create_documents([rfp_text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vector_store = FAISS.from_documents(documents=text_rfp, embedding=embeddings)

    vector_store.save_local("./app/vector_store/faiss_index")

    query = "Years of Experience in Temporary staffing, W-9 Form"

    retrieved_data = vector_store.similarity_search(query)
    llm = ChatGroq(model="gemma2-9b-it", api_key=os.environ.get("GROQ_API_KEY"))

    messages = [
        (
            "system",
            "You are a helpful assistant. Given a context from an RFP document, check if the following information is available:\n"
            "- Is there any mention of 'Years of Experience in Temporary staffing'? If yes, is the experience **less than 7 years**?\n"
            "- Is there any mention of a 'W-9 Form'?\n\n"
            "Respond with clear yes/no answers after complete analysis, and give a warning if something is not available.",
        ),
        ("developer", f"{query}\n\nContext:\n{retrieved_data}"),
    ]

    answer = llm.invoke(messages)
    return answer


if __name__ == "__main__":
    ans = preprocess()
