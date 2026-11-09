"""
Test the FAISS vector store with document chunks and embeddings.
"""

from src.document_loader import load_pdf
from src.text_cleaner import clean_documents
from src.text_splitter import split_documents
from src.embedding_generator import EmbeddingGenerator
from src.vector_store import VectorStore


def main():
    pdf_path = "data/documents/sample.pdf"

    try:
        # 1. Load, clean, and split PDF documents
        raw_documents = load_pdf(pdf_path)
        cleaned_documents = clean_documents(raw_documents)
        chunks = split_documents(cleaned_documents)

        # 2. Extract text from each chunk
        chunk_texts = [chunk["text"] for chunk in chunks]

        # 3. Generate embeddings for all chunks
        embedding_generator = EmbeddingGenerator()
        embeddings = embedding_generator.generate_embeddings(chunk_texts)

        # 4. Initialize VectorStore with dimension 384 and add documents
        vector_store = VectorStore(dimension=384)
        vector_store.add_documents(chunks, embeddings)

        print(f"Number of chunks added: {len(chunks)}")
        print(f"Number of vectors in FAISS index: {vector_store.index.ntotal}")

        if len(embeddings) == 0:
            print("No embeddings available for search.")
            return

        # 5. Search using the first chunk's embedding as query
        query_embedding = embeddings[0]
        results = vector_store.search(query_embedding, top_k=3)

        print(f"Number of search results: {len(results)}\n")

        # 6. Display search results
        for i, res in enumerate(results, start=1):
            doc = res["document"]
            distance = res["distance"]
            print(f"Result {i}:")
            print(f"  Source: {doc.get('source')}")
            print(f"  Page Number: {doc.get('page_number')}")
            print(f"  Distance: {distance:.4f}")
            print(f"  Text preview: {doc.get('text', '')[:100]}\n")

    except FileNotFoundError as err:
        print(f"Error: PDF file not found at '{pdf_path}' -> {err}")
    except Exception as err:
        print(f"Unexpected error during vector store testing: {err}")


if __name__ == "__main__":
    main()
