from util.chunking import chunk_document
from util.data_ingestion import clean_rfp_text, extract_text_from_pdf
from util.logger_config import get_logger

RFP_PATH = "/home/kparth/HomersHackers/parth/data/ELIGIBLE_RFP_2.pdf"

logger = get_logger("rfp_pipeline")


def execute():
    logger.info("Starting RFP processing pipeline.")

    logger.info(f"Extracting text from PDF: {RFP_PATH}")
    raw_rfp_text = extract_text_from_pdf(RFP_PATH)
    logger.info("Extraction complete.")

    logger.info("Cleaning extracted text.")
    cleaned_rfp = clean_rfp_text(raw_rfp_text)
    logger.info("Cleaning complete.")

    document = cleaned_rfp

    logger.info("Chunking the cleaned document.")
    chunks = chunk_document(document)
    return chunks
#     logger.info(f"Document split into {len(chunks)} chunks.")

#     for i, chunk in enumerate(chunks):
#         logger.debug(f"Printing chunk {i+1}")
#         print(f"\n\033[96mChunk {i+1}:\033[0m\n{chunk}\n")

#     logger.info("RFP processing pipeline complete.")


# print(execute())