import re

import pdfplumber


def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


def clean_rfp_text(raw_text: str) -> str:
    """
    Clean the raw RFP text by removing noise and normalizing it.

    Args:
        raw_text (str): Raw text extracted from the RFP PDF.

    Returns:
        str: Cleaned text ready for chunking.
    """
    cleaned_text = re.sub(r"\s+", " ", raw_text).strip()
    cleaned_text = re.sub(r"Page \d+|\d+ of \d+", "", cleaned_text)
    cleaned_text = re.sub(r"[^\w\s.,;:-]", "", cleaned_text)

    return cleaned_text
