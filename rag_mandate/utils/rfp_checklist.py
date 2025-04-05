import os

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b", api_key=os.environ.get("GROQ_API_KEY")
)


def analyze_rfp(relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    query = "Years of Experience in Temporary staffing, W-9 Form, qualifications, certifications, licenses"
    messages = [
        (
            "system",
            "You are a helpful assistant. Given a context from an RFP document, check if the following information is available:\n"
            "- Is there any mention of 'Years of Experience in Temporary staffing'? If yes, is the experience **less than 7 years**?\n"
            "- Is there any mention of a 'W-9 Form'?\n"
            "- Is there any mention of Insurance Certificates?\n"
            "- Is there any mention of 'Company Length of Existence'?\n"
            "- Is there any mention of 'Licenses'?\n"
            "Respond with clear yes/no answers after complete analysis, and mark missing requirements clearly in red using markdown.",
        ),
        ("developer", f"{query}\n\nContext:\n{context}"),
    ]
    return llm.invoke(messages)
