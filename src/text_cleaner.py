import re
from typing import Any, Dict, List


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text by collapsing repeated whitespace
    and stripping leading and trailing whitespace.

    Args:
        text (str): Raw extracted text.

    Returns:
        str: Normalized text with single whitespace separators and trimmed edges.
    """
    if not text:
        return ""

    # Replace repeated whitespace (spaces, newlines, tabs) with a single space
    cleaned = re.sub(r"\s+", " ", text)

    # Remove leading and trailing whitespace
    return cleaned.strip()


def clean_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean and normalize text across a list of document/page dictionaries
    while preserving original metadata.

    Args:
        documents (List[Dict[str, Any]]): List of page dictionaries, each containing
            'text', 'page_number', and 'source'.

    Returns:
        List[Dict[str, Any]]: A new list of dictionaries with cleaned 'text',
            preserving 'page_number' and 'source'. Documents with empty cleaned
            text are omitted.
    """
    cleaned_docs: List[Dict[str, Any]] = []

    for doc in documents:
        raw_text = doc.get("text", "")
        cleaned = clean_text(raw_text)

        # Skip documents whose cleaned text is empty
        if not cleaned:
            continue

        cleaned_docs.append({
            "text": cleaned,
            "page_number": doc.get("page_number"),
            "source": doc.get("source"),
        })

    return cleaned_docs
