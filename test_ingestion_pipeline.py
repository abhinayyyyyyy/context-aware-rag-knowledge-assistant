"""
Test the complete document ingestion pipeline implemented in src/ingestion_pipeline.py.
"""

from src.ingestion_pipeline import build_vector_store
from src.embedding_generator import EmbeddingGenerator


def main() -> None:
    pdf_path = "data/documents/sample.pdf"

    try:
        # Build the vector store using the end-to-end ingestion pipeline
        print(f"Building vector store from: {pdf_path}")
        vector_store = build_vector_store(pdf_path)

        # Print vector store statistics
        total_vectors = vector_store.index.ntotal
        total_documents = len(vector_store.documents)
        dimension = vector_store.dimension

        print(f"Total number of vectors in FAISS index: {total_vectors}")
        print(f"Total number of stored documents: {total_documents}")
        print(f"Embedding dimension of the vector store: {dimension}")

        if total_documents == 0:
            print("No documents found in the vector store.")
            return

        # Use the first stored document's text as a test query
        first_doc = vector_store.documents[0]
        test_query = first_doc["text"]
        print(f"\nUsing first stored document text as test query (preview):\n{test_query[:100]}...\n")

        # Generate an embedding for the test query using EmbeddingGenerator
        embedding_generator = EmbeddingGenerator()
        query_embeddings = embedding_generator.generate_embeddings([test_query])
        query_embedding = query_embeddings[0]

        # Search the vector store for the top 3 results
        results = vector_store.search(query_embedding, top_k=3)
        print(f"Top {len(results)} search results:")

        # Print details for each result
        for i, res in enumerate(results, start=1):
            doc = res["document"]
            distance = res["distance"]
            print(f"\nResult {i}:")
            print(f"  Source: {doc.get('source')}")
            print(f"  Page number: {doc.get('page_number')}")
            print(f"  Distance: {distance:.4f}")
            print(f"  First 100 characters of text: {doc.get('text', '')[:100]}")

    except FileNotFoundError as err:
        print(f"Error: File not found -> {err}")
    except Exception as err:
        print(f"Unexpected error during ingestion pipeline testing: {err}")


if __name__ == "__main__":
    main()
