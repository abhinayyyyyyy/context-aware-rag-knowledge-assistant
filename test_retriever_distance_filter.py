from typing import List
import numpy as np
from src.vector_store import VectorStore
from src.embedding_generator import EmbeddingGenerator
from src.retriever import Retriever


class MockEmbeddingGenerator(EmbeddingGenerator):
    """
    Lightweight mock for EmbeddingGenerator that returns deterministic vectors
    without loading the SentenceTransformer model or downloading weights.
    """

    def __init__(self) -> None:
        # Avoid calling super().__init__() to bypass model loading and network calls
        pass

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Return deterministic (N, 3) embeddings where every query maps to [1.0, 0.0, 0.0].
        """
        return np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (len(texts), 1))


def test_retriever_distance_filter() -> None:
    # 2. Instantiate lightweight mock embedding generator
    mock_generator = MockEmbeddingGenerator()

    # 3. Create VectorStore with dimension 3
    vector_store = VectorStore(dimension=3)

    # 4. Add three documents with corresponding embeddings
    documents = [
        {"id": 1, "text": "Document 1", "source": "doc1.txt"},
        {"id": 2, "text": "Document 2", "source": "doc2.txt"},
        {"id": 3, "text": "Document 3", "source": "doc3.txt"},
    ]

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],  # Document 1 (exact match for query)
            [0.0, 1.0, 0.0],  # Document 2 (distance = 2.0)
            [0.0, 0.0, 1.0],  # Document 3 (distance = 2.0)
        ],
        dtype=np.float32,
    )

    vector_store.add_documents(documents, embeddings)

    # 6. Create Retriever using VectorStore and mock embedding generator
    retriever = Retriever(vector_store=vector_store, embedding_generator=mock_generator)

    # 7. Test normal retrieval
    print("Testing normal retrieval (top_k=3)...")
    results = retriever.retrieve("test query", top_k=3)
    assert len(results) == 3, f"Expected 3 results, got {len(results)}."
    assert results[0]["document"]["id"] == 1, (
        f"Expected closest result to be Document 1, got {results[0]['document']}."
    )
    assert np.isclose(results[0]["distance"], 0.0, atol=1e-5), (
        f"Expected closest distance 0.0, got {results[0]['distance']}."
    )
    print("✓ Normal retrieval verified: 3 results returned, closest is Document 1 with distance 0.0.")

    # 8. Test distance filtering with a threshold that only allows the exact match
    print("Testing distance filtering with max_distance=0.1...")
    filtered_exact = retriever.retrieve("test query", top_k=3, max_distance=0.1)
    assert len(filtered_exact) == 1, (
        f"Expected exactly 1 result with max_distance=0.1, got {len(filtered_exact)}."
    )
    assert filtered_exact[0]["document"]["id"] == 1, (
        f"Expected Document 1, got {filtered_exact[0]['document']}."
    )
    print("✓ Distance filtering verified: exactly 1 result returned and it is Document 1.")

    # 9. Test a threshold that allows all three vectors
    print("Testing distance filtering with max_distance=2.0...")
    filtered_all = retriever.retrieve("test query", top_k=3, max_distance=2.0)
    assert len(filtered_all) == 3, (
        f"Expected 3 results with max_distance=2.0, got {len(filtered_all)}."
    )
    print("✓ Threshold allowing all vectors verified: all 3 results returned.")

    # 10. Test invalid max_distance
    print("Testing negative max_distance validation...")
    try:
        retriever.retrieve("test query", max_distance=-1)
        assert False, "Expected ValueError when max_distance is negative."
    except ValueError as err:
        print(f"✓ Invalid max_distance properly rejected with ValueError: {err}")

    # 11. Test backward compatibility
    print("Testing backward compatibility without max_distance...")
    default_results = retriever.retrieve("test query")
    assert len(default_results) == 3, (
        f"Expected default 3 results, got {len(default_results)}."
    )
    assert default_results[0]["document"]["id"] == 1, (
        f"Expected Document 1, got {default_results[0]['document']}."
    )
    print("✓ Backward compatibility verified: retriever.retrieve('test query') works as expected.")

    print("\nAll Retriever distance filtering tests passed successfully!")


if __name__ == "__main__":
    test_retriever_distance_filter()
