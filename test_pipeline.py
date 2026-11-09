"""
Test the end-to-end document-processing pipeline:
PDF extraction -> Text cleaning -> Text chunking.
"""

from src.document_loader import load_pdf
from src.text_cleaner import clean_documents
from src.text_splitter import split_documents


def main():
    pdf_path = "data/documents/sample.pdf"

    try:
        # 1. Load PDF pages
        raw_documents = load_pdf(pdf_path)
        print(f"Total original pages: {len(raw_documents)}")

        # 2. Clean extracted documents
        cleaned_documents = clean_documents(raw_documents)
        print(f"Total pages remaining after cleaning: {len(cleaned_documents)}")

        # 3. Split cleaned documents into chunks
        chunks = split_documents(cleaned_documents)
        print(f"Total generated chunks: {len(chunks)}")

        # 4. Preview the first chunk details
        if chunks:
            first_chunk = chunks[0]
            print(f"First chunk page number: {first_chunk['page_number']}")
            print(f"First chunk source: {first_chunk['source']}")
            print(f"First 300 characters of first chunk:\n{first_chunk['text'][:300]}")
        else:
            print("No chunks generated from documents.")

    except FileNotFoundError as err:
        print(f"Error: PDF file not found at '{pdf_path}' -> {err}")
    except Exception as err:
        print(f"Unexpected error in document-processing pipeline: {err}")


if __name__ == "__main__":
    main()
