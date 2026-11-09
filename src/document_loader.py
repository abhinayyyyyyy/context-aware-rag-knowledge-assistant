from pathlib import Path
from typing import Any, Dict, List
from pypdf import PdfReader
from pypdf.errors import PdfReadError


def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract text and page-level metadata from a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries with extracted page data,
        where each item contains 'text', 'page_number' (1-indexed), and 'source'.
    """
    path = Path(file_path)

    # Check if the file exists
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    source_name = path.name
    documents: List[Dict[str, Any]] = []

    # Read PDF and extract text page by page
    try:
        reader = PdfReader(str(path))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            documents.append({
                "text": text,
                "page_number": page_num,
                "source": source_name,
            })
    except PdfReadError as err:
        raise ValueError(f"Corrupted or unreadable PDF file '{file_path}': {err}") from err
    except Exception as err:
        raise RuntimeError(f"Error while reading PDF file '{file_path}': {err}") from err

    return documents
