from typing import Any, Dict, List, Optional
from src.document_loader import load_pdf
from src.text_cleaner import clean_documents
from src.text_splitter import split_documents
from src.embedding_generator import EmbeddingGenerator
from src.vector_store import VectorStore
from src.vector_store_interface import BaseVectorStore
from src.vector_store_factory import create_vector_store


def build_vector_store(
    pdf_path: str,
    embedding_generator: Optional[EmbeddingGenerator] = None,
    vector_store: Optional[BaseVectorStore] = None,
) -> BaseVectorStore:
    """
    Build a populated FAISS VectorStore from a PDF document by executing
    the complete ingestion pipeline: load -> clean -> split -> embed -> index.

    Args:
        pdf_path (str): File path to the input PDF document.
        embedding_generator (Optional[EmbeddingGenerator]): Optional pre-loaded
            embedding generator. If None, a new instance is initialized.

    Returns:
        VectorStore: Populated vector store indexed with document chunks and embeddings.

    Raises:
        FileNotFoundError: If the input PDF file does not exist.
        ValueError: If no pages, cleaned text, or chunks can be produced from the file.
    """
    # 1. Load PDF pages
    raw_documents = load_pdf(pdf_path)
    if not raw_documents:
        raise ValueError(f"No pages could be extracted from PDF file: '{pdf_path}'")

    # 2. Clean text and normalize whitespace across documents
    cleaned_documents = clean_documents(raw_documents)
    if not cleaned_documents:
        raise ValueError(f"No readable text remained after cleaning PDF file: '{pdf_path}'")

    # 3. Split cleaned documents into overlapping chunks
    chunks = split_documents(cleaned_documents)
    if not chunks:
        raise ValueError(f"No text chunks were generated from PDF file: '{pdf_path}'")

    # 4. Extract text from each chunk
    chunk_texts = [chunk["text"] for chunk in chunks]

    # 5. Generate numerical embeddings for all chunks
    if embedding_generator is None:
        embedding_generator = EmbeddingGenerator()

    embeddings = embedding_generator.generate_embeddings(chunk_texts)
    if embeddings.size == 0 or embeddings.ndim != 2:
        raise ValueError(f"Failed to generate valid embeddings for PDF file: '{pdf_path}'")

    # 6. Initialize VectorStore matching the embedding dimensionality
    embedding_dim = embeddings.shape[1]
    if vector_store is None:
        vector_store = create_vector_store(dimension=embedding_dim)

    # 7. Add chunks and embeddings to the vector store
    vector_store.add_documents(chunks, embeddings)

    # 8. Return populated VectorStore
    return vector_store


def build_multi_document_vector_store(
    pdf_paths: List[str],
    embedding_generator: Optional[EmbeddingGenerator] = None,
    vector_store: Optional[BaseVectorStore] = None,
) -> BaseVectorStore:
    """
    Build a single populated FAISS VectorStore from multiple PDF documents.

    Each PDF document is processed independently through loading, text cleaning,
    and text splitting. The generated chunks from all valid PDFs are combined,
    embedded using a shared EmbeddingGenerator, and indexed into a single VectorStore.
    PDFs that are empty or contain no readable text are skipped.

    Args:
        pdf_paths (List[str]): List of file paths to input PDF documents.
        embedding_generator (Optional[EmbeddingGenerator]): Optional pre-loaded
            reusable embedding generator. If None, a new instance is created.

    Returns:
        VectorStore: Single vector store containing all indexed chunks and embeddings.

    Raises:
        ValueError: If pdf_paths is empty.
        ValueError: If no valid document chunks were generated from the supplied PDF files.
        FileNotFoundError: If any specified PDF file does not exist.
    """
    if not pdf_paths:
        raise ValueError("pdf_paths must contain at least one PDF file.")

    all_chunks: List[Dict[str, Any]] = []

    for pdf_path in pdf_paths:
        # FileNotFoundError is intentionally not caught and will propagate if the file does not exist.
        # ValueError is caught to skip empty, corrupted, or unreadable PDFs.
        try:
            raw_documents = load_pdf(pdf_path)
        except ValueError:
            continue

        if not raw_documents:
            continue

        cleaned_documents = clean_documents(raw_documents)
        if not cleaned_documents:
            continue

        chunks = split_documents(cleaned_documents)
        if not chunks:
            continue

        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("No valid document chunks were generated from the supplied PDF files.")

    # Extract text from all combined chunks
    chunk_texts = [chunk["text"] for chunk in all_chunks]

    # Use ONE shared EmbeddingGenerator instance
    if embedding_generator is None:
        embedding_generator = EmbeddingGenerator()

    # Generate numerical embeddings for the combined chunk list
    embeddings = embedding_generator.generate_embeddings(chunk_texts)
    if embeddings.size == 0 or embeddings.ndim != 2:
        raise ValueError("Failed to generate valid embeddings for the combined document chunks.")

    # Create exactly ONE VectorStore using the resulting embedding dimension
    embedding_dim = embeddings.shape[1]
    if vector_store is None:
        vector_store = create_vector_store(dimension=embedding_dim)

    # Add all chunks and embeddings to that single VectorStore
    vector_store.add_documents(all_chunks, embeddings)

    return vector_store
