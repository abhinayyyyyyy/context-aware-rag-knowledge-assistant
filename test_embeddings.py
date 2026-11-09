"""
Test embedding generation for document chunks using EmbeddingGenerator.
"""

from src.document_loader import load_pdf
from src.text_cleaner import clean_documents
from src.text_splitter import split_documents
from src.embedding_generator import EmbeddingGenerator


def main():
    pdf_path = "data/documents/sample.pdf"

    try:
        # 1. Load, clean, and split PDF document
        raw_documents = load_pdf(pdf_path)
        cleaned_documents = clean_documents(raw_documents)
        chunks = split_documents(cleaned_documents)

        # 2. Extract text from each chunk
        chunk_texts = [chunk["text"] for chunk in chunks]

        # 3. Initialize generator and create embeddings
        embedding_generator = EmbeddingGenerator()
        embeddings = embedding_generator.generate_embeddings(chunk_texts)

        # 4. Display results
        print(f"Number of text chunks: {len(chunk_texts)}")
        print(f"Embeddings array shape: {embeddings.shape}")
        if len(embeddings) > 0:
            print(f"Length of the first embedding vector: {len(embeddings[0])}")

    except FileNotFoundError as err:
        print(f"Error: PDF file not found at '{pdf_path}' -> {err}")
    except Exception as err:
        print(f"Unexpected error during embedding generation: {err}")


if __name__ == "__main__":
    main()
