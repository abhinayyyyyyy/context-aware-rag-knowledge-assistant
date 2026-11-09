"""
Test the text chunking functionality implemented in src/text_splitter.py.
"""

from src.document_loader import load_pdf
from src.text_splitter import split_documents


def main():
    pdf_path = "data/documents/sample.pdf"

    try:
        # Load pages from PDF
        pages = load_pdf(pdf_path)
        print(f"Total original pages: {len(pages)}")

        # Split pages into text chunks
        chunks = split_documents(pages)
        print(f"Total generated chunks: {len(chunks)}")

        # Print details of the first chunk if available
        if chunks:
            first_chunk = chunks[0]
            print(f"First chunk page number: {first_chunk['page_number']}")
            print(f"First chunk source: {first_chunk['source']}")
            print(f"First 300 characters of first chunk:\n{first_chunk['text'][:300]}")
        else:
            print("No chunks were generated from the document.")

    except FileNotFoundError as err:
        print(f"Error: PDF file not found at '{pdf_path}' -> {err}")
    except Exception as err:
        print(f"Unexpected error during document loading/splitting: {err}")


if __name__ == "__main__":
    main()
