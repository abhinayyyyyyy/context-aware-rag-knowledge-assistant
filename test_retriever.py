"""
Test the Retriever component using the PDF ingestion pipeline.
"""

from src.ingestion_pipeline import build_vector_store
from src.embedding_generator import EmbeddingGenerator
from src.retriever import Retriever


def main() -> None:
    pdf_path = "data/documents/sample.pdf"

    try:
        # Create one reusable EmbeddingGenerator instance
        embedding_generator = EmbeddingGenerator()

        # Build the vector store using the ingestion pipeline and reusable generator
        print(f"Building vector store from: {pdf_path}")
        vector_store = build_vector_store(pdf_path, embedding_generator=embedding_generator)

        # Create Retriever with the vector store and embedding generator
        retriever = Retriever(vector_store=vector_store, embedding_generator=embedding_generator)

        # Test query
        query = "What does Juspay do?"
        print(f"\nQuery: '{query}'")

        # Retrieve top 3 results
        results = retriever.retrieve(query=query, top_k=3)
        print(f"Retrieved {len(results)} results:\n")

        # Display result details
        for i, res in enumerate(results, start=1):
            doc = res["document"]
            distance = res["distance"]
            print(f"Result {i}:")
            print(f"  Source: {doc.get('source')}")
            print(f"  Page number: {doc.get('page_number')}")
            print(f"  Distance: {distance:.4f}")
            print(f"  First 200 characters of retrieved text:\n  {doc.get('text', '')[:200]}\n")

        # Test empty query validation and confirm ValueError is raised
        print("Testing empty query validation...")
        try:
            retriever.retrieve("")
            print("Failed: Retriever did not raise ValueError for an empty query.")
        except ValueError as err:
            print(f"Confirmed: ValueError raised as expected -> {err}")

    except FileNotFoundError as err:
        print(f"Error: File not found -> {err}")
    except Exception as err:
        print(f"Unexpected error during retriever testing: {err}")


if __name__ == "__main__":
    main()
