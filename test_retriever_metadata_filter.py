from typing import List
import numpy as np
from src.embedding_generator import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import Retriever


class MockEmbeddingGenerator(EmbeddingGenerator):
    """
    Mock embedding generator returning deterministic vectors without
    loading models or performing network calls.
    """

    def __init__(self) -> None:
        pass

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        # Returns a query embedding vector [1.0, 0.0, 0.0] for each text
        return np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (len(texts), 1))


def test_retriever_metadata_filter() -> None:
    mock_generator = MockEmbeddingGenerator()
    vector_store = VectorStore(dimension=3)

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],  # Doc A - chunk 0 (distance to query: 0.0)
            [0.9, 0.1, 0.0],  # Doc A - chunk 1 (distance to query: 0.02)
            [0.8, 0.2, 0.0],  # Doc A - chunk 2 (distance to query: 0.08)
            [0.0, 0.9, 0.1],  # Doc B - chunk 0 (distance to query: 1.82)
            [0.0, 1.0, 0.0],  # Doc B - chunk 1 (distance to query: 2.00)
        ],
        dtype=np.float32,
    )

    doc_a_id = "doc_a_id_123"
    doc_b_id = "doc_b_id_456"

    documents = [
        {
            "text": "Doc A content chunk 0",
            "page_number": 1,
            "source": "document_a.pdf",
            "document_id": doc_a_id,
            "chunk_id": f"{doc_a_id}_chunk_0",
        },
        {
            "text": "Doc A content chunk 1",
            "page_number": 2,
            "source": "document_a.pdf",
            "document_id": doc_a_id,
            "chunk_id": f"{doc_a_id}_chunk_1",
        },
        {
            "text": "Doc A content chunk 2",
            "page_number": 3,
            "source": "document_a.pdf",
            "document_id": doc_a_id,
            "chunk_id": f"{doc_a_id}_chunk_2",
        },
        {
            "text": "Doc B content chunk 0",
            "page_number": 1,
            "source": "document_b.pdf",
            "document_id": doc_b_id,
            "chunk_id": f"{doc_b_id}_chunk_0",
        },
        {
            "text": "Doc B content chunk 1",
            "page_number": 2,
            "source": "document_b.pdf",
            "document_id": doc_b_id,
            "chunk_id": f"{doc_b_id}_chunk_1",
        },
    ]

    vector_store.add_documents(documents, embeddings)
    retriever = Retriever(vector_store=vector_store, embedding_generator=mock_generator)

    # 1. Source-only filtering
    print("1. Testing source-only filtering...")
    results_source = retriever.retrieve("query", top_k=2, source="document_b.pdf")
    assert len(results_source) == 2, f"Expected 2 results, got {len(results_source)}."
    assert all(r["document"]["source"] == "document_b.pdf" for r in results_source), (
        "Found results from unexpected source."
    )
    print("✓ Source-only filtering verified.")

    # 2. Document_id-only filtering
    print("2. Testing document_id-only filtering...")
    results_doc_id = retriever.retrieve("query", top_k=3, document_id=doc_b_id)
    assert len(results_doc_id) == 2, f"Expected 2 results, got {len(results_doc_id)}."
    assert all(r["document"]["document_id"] == doc_b_id for r in results_doc_id), (
        "Found results from unexpected document_id."
    )
    print("✓ Document_id-only filtering verified.")

    # 3. Combined source + document_id filtering
    print("3. Testing combined matching source and document_id...")
    results_combined = retriever.retrieve(
        "query",
        top_k=3,
        source="document_b.pdf",
        document_id=doc_b_id,
    )
    assert len(results_combined) == 2, f"Expected 2 results, got {len(results_combined)}."
    for r in results_combined:
        assert r["document"]["source"] == "document_b.pdf"
        assert r["document"]["document_id"] == doc_b_id
    print("✓ Combined source + document_id filtering verified.")

    # 4. Conflicting filters return []
    print("4. Testing conflicting filters returning empty list...")
    results_conflict = retriever.retrieve(
        "query",
        top_k=3,
        source="document_a.pdf",
        document_id=doc_b_id,
    )
    assert results_conflict == [], f"Expected empty list, got {results_conflict}."
    print("✓ Conflicting filters properly returned empty list.")

    # 5. Metadata filtering combined with max_distance
    print("5. Testing metadata filtering combined with max_distance...")
    # Doc B chunk 0 distance: ~1.82, Doc B chunk 1 distance: ~2.00
    results_dist_filtered = retriever.retrieve(
        "query",
        top_k=3,
        max_distance=1.9,
        source="document_b.pdf",
    )
    assert len(results_dist_filtered) == 1, (
        f"Expected 1 result <= 1.9 max_distance, got {len(results_dist_filtered)}."
    )
    assert results_dist_filtered[0]["document"]["chunk_id"] == f"{doc_b_id}_chunk_0"

    results_none_within_dist = retriever.retrieve(
        "query",
        top_k=3,
        max_distance=1.0,
        source="document_b.pdf",
    )
    assert len(results_none_within_dist) == 0, (
        f"Expected 0 results <= 1.0 max_distance, got {len(results_none_within_dist)}."
    )
    print("✓ Combined metadata filtering with max_distance verified.")

    # 6. Pre-top-k filtering behavior
    print("6. Testing pre-top-k candidate filtering...")
    # Unfiltered top 2 are both Document A
    unfiltered_top_2 = retriever.retrieve("query", top_k=2)
    assert all(r["document"]["source"] == "document_a.pdf" for r in unfiltered_top_2)

    # Document B candidates are ranked 4th and 5th globally, but top_k=2 with source='document_b.pdf'
    # must still successfully return both Document B candidates
    filtered_doc_b = retriever.retrieve("query", top_k=2, source="document_b.pdf")
    assert len(filtered_doc_b) == 2
    assert filtered_doc_b[0]["document"]["chunk_id"] == f"{doc_b_id}_chunk_0"
    assert filtered_doc_b[1]["document"]["chunk_id"] == f"{doc_b_id}_chunk_1"
    print("✓ Pre-top-k filtering verified: retrieved matching candidates even when ranked below non-matching candidates globally.")

    # 7. Invalid empty source validation
    print("7. Testing invalid empty source validation...")
    try:
        retriever.retrieve("query", source="")
        assert False, "Expected ValueError for empty source."
    except ValueError as err:
        assert str(err) == "source must be a non-empty string when provided."
        print(f"✓ Invalid source rejected: {err}")

    # 8. Invalid empty document_id validation
    print("8. Testing invalid empty document_id validation...")
    try:
        retriever.retrieve("query", document_id="")
        assert False, "Expected ValueError for empty document_id."
    except ValueError as err:
        assert str(err) == "document_id must be a non-empty string when provided."
        print(f"✓ Invalid document_id rejected: {err}")

    # 9. Empty VectorStore with filters
    print("9. Testing empty VectorStore with filters...")
    empty_store = VectorStore(dimension=3)
    empty_retriever = Retriever(vector_store=empty_store, embedding_generator=mock_generator)
    results_empty = empty_retriever.retrieve("query", source="document_a.pdf")
    assert results_empty == [], f"Expected empty list from empty VectorStore, got {results_empty}."
    print("✓ Empty VectorStore with metadata filters safely returned empty list.")

    # 10. Backward compatibility without metadata filters
    print("10. Testing backward compatibility...")
    compat_results = retriever.retrieve("query", top_k=3)
    assert len(compat_results) == 3
    assert compat_results[0]["document"]["chunk_id"] == f"{doc_a_id}_chunk_0"
    assert compat_results[1]["document"]["chunk_id"] == f"{doc_a_id}_chunk_1"
    assert compat_results[2]["document"]["chunk_id"] == f"{doc_a_id}_chunk_2"
    print("✓ Backward compatibility without metadata filters verified.")

    print("\nAll Retriever metadata filter tests passed successfully!")


if __name__ == "__main__":
    test_retriever_metadata_filter()
