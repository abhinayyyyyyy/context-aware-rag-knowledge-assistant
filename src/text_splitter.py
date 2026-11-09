import hashlib
from typing import Any, Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split extracted PDF page documents into smaller text chunks suitable for
    embedding generation and semantic retrieval, adding deterministic identifiers.

    Identifiers:
        - `document_id`: A deterministic SHA-256 hexadecimal digest of the source filename.
          It exists to provide a stable, consistent identifier for each source document across
          executions, enabling multi-document management, tracking, and deduplication.
        - `chunk_id`: A unique, deterministic identifier for each chunk formatted as
          `<document_id>_chunk_<index>`. It exists to facilitate granular chunk-level retrieval,
          citation attribution, and persistent index updates. Chunk numbering is zero-based,
          increments sequentially by 1, and restarts for each source document.

    Args:
        documents (List[Dict[str, Any]]): List of page dictionaries, each containing
            'text', 'page_number', and 'source'.

    Returns:
        List[Dict[str, Any]]: List of chunk dictionaries containing 'text',
            'page_number', 'source', 'document_id', and 'chunk_id'.
    """
    # Configure text splitter for paragraphs, sentences/lines, words, and characters
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: List[Dict[str, Any]] = []
    chunk_counters: Dict[str, int] = {}

    for doc in documents:
        text = doc.get("text", "")
        if not text or not text.strip():
            continue

        page_number = doc.get("page_number")
        source = doc.get("source")

        # Generate deterministic document_id based on the source filename
        source_str = str(source) if source is not None else ""
        document_id = hashlib.sha256(source_str.encode("utf-8")).hexdigest()

        # Split page text into overlapping chunks
        split_texts = splitter.split_text(text)
        for chunk_text in split_texts:
            chunk_index = chunk_counters.get(document_id, 0)
            chunk_id = f"{document_id}_chunk_{chunk_index}"
            chunk_counters[document_id] = chunk_index + 1

            chunks.append({
                "text": chunk_text,
                "page_number": page_number,
                "source": source,
                "document_id": document_id,
                "chunk_id": chunk_id,
            })

    return chunks
