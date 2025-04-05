import pandas as pd
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


def load_company_data(csv_path):
    try:
        data = pd.read_csv(csv_path)
        return data
    except Exception as e:
        print(f"Error loading company data: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    rfp_path = "./data/ELIGIBLE_RFP_2.pdf"
    company_csv_path = "./data/company_data.csv"

    rfp_text = extract_text_from_pdf(rfp_path)
    if rfp_text:
        print("RFP Text Excerpt:")
        print(rfp_text[:500])
    else:
        print("No text extracted from RFP.")

    company_data = load_company_data(company_csv_path)
    if not company_data.empty:
        print("\nCompany Data:")
        print(company_data.head())
    else:
        print("No company data loaded.")
