"""
End-to-end integration test for the RAG pipeline using local Ollama.
"""

from src.ingestion_pipeline import build_vector_store
from src.embedding_generator import EmbeddingGenerator
from src.retriever import Retriever
from src.rag_generator import RAGGenerator


def test_rag_pipeline() -> None:
    pdf_path = "data/documents/sample.pdf"

    print("=== Initializing RAG Pipeline ===")
    # 1. Create one shared EmbeddingGenerator instance
    embedding_generator = EmbeddingGenerator()

    # 2. Build the vector store using the ingestion pipeline
    print(f"Building vector store from: {pdf_path}")
    vector_store = build_vector_store(pdf_path, embedding_generator=embedding_generator)

    # 3. Create Retriever using the vector store and shared embedding generator
    retriever = Retriever(vector_store=vector_store, embedding_generator=embedding_generator)

    # 4. Create RAGGenerator with Retriever and llama3.2:latest
    rag_generator = RAGGenerator(retriever=retriever, model_name="llama3.2:latest")

    # 5. Query the RAG system
    query = "What does Juspay do?"
    print(f"\nQuestion:\n{query}\n")

    result = rag_generator.generate_answer(query, top_k=3)

    answer = result.get("answer")
    sources = result.get("sources")

    # 6. Basic validation
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Validation failed: Answer must be a non-empty string.")

    if not sources or not isinstance(sources, list):
        raise ValueError("Validation failed: Sources must not be empty.")

    # 7. Print results
    print("Generated Answer:")
    print(answer)

    print(f"\nNumber of Sources: {len(sources)}")
    print("\nSources:")
    for i, src in enumerate(sources, start=1):
        print(f"\n--- Source {i} ---")
        print(f"Filename:    {src.get('source')}")
        print(f"Page Number: {src.get('page_number')}")
        distance = src.get("distance")
        if distance is not None:
            print(f"Distance:    {distance:.4f}")
        else:
            print(f"Distance:    {distance}")

    print("\n=== RAG Pipeline Test Passed Successfully ===")


if __name__ == "__main__":
    test_rag_pipeline()
