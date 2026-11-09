"""
Test the PDF extraction functionality implemented in src/document_loader.py.
"""

from src.document_loader import load_pdf


def main():
    # Path to sample PDF input
    pdf_path = "data/documents/sample.pdf"

    try:
        # Load and extract pages from the PDF
        pages = load_pdf(pdf_path)
        print(f"Total extracted pages: {len(pages)}")

        # If pages are extracted, display details of the first page
        if pages:
            first_page = pages[0]
            print(f"First page number: {first_page['page_number']}")
            print(f"Source file: {first_page['source']}")
            print(f"First 200 characters of text:\n{first_page['text'][:200]}")

    except FileNotFoundError as err:
        print(f"Error: PDF file not found at '{pdf_path}' -> {err}")
    except Exception as err:
        print(f"Unexpected error while loading PDF: {err}")


if __name__ == "__main__":
    main()
