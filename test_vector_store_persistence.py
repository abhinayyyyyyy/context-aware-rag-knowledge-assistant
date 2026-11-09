import os
import tempfile
import numpy as np
from src.vector_store import VectorStore


def test_vector_store_persistence() -> None:
    # 2. Create a VectorStore with dimension 3
    dimension = 3
    vector_store = VectorStore(dimension=dimension)

    # 3. Create 3 simple test document dictionaries containing text, page_number, and source
    documents = [
        {
            "text": "First chunk discussing artificial intelligence.",
            "page_number": 1,
            "source": "doc1.pdf",
        },
        {
            "text": "Second chunk describing machine learning algorithms.",
            "page_number": 2,
            "source": "doc1.pdf",
        },
        {
            "text": "Third chunk about retrieval-augmented generation systems.",
            "page_number": 3,
            "source": "doc2.pdf",
        },
    ]

    # 4. Create 3 corresponding NumPy embeddings with shape (3, 3)
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    assert embeddings.shape == (3, 3), f"Expected shape (3, 3), got {embeddings.shape}."

    # 5. Add the documents and embeddings to the VectorStore
    vector_store.add_documents(documents, embeddings)

    # 6. Search the original store with one of the embeddings and record the results
    query_embedding = embeddings[0]
    original_results = vector_store.search(query_embedding, top_k=3)
    assert len(original_results) == 3, f"Expected 3 search results, got {len(original_results)}."

    # 7. Save the VectorStore to a temporary directory
    with tempfile.TemporaryDirectory() as temp_directory:
        vector_store.save(temp_directory)

        # 8. Verify that index.faiss and documents.pkl exist
        index_file = os.path.join(temp_directory, "index.faiss")
        documents_file = os.path.join(temp_directory, "documents.pkl")

        assert os.path.exists(index_file), f"Expected {index_file} to exist."
        assert os.path.exists(documents_file), f"Expected {documents_file} to exist."
        print(f"✓ Saved files verified: {index_file} and {documents_file} exist.")

        # 9. Load the store using VectorStore.load
        loaded_store = VectorStore.load(temp_directory)

        # 10. Verify loaded properties
        assert loaded_store.index.ntotal == 3, (
            f"Expected loaded vector count to be 3, got {loaded_store.index.ntotal}."
        )
        assert len(loaded_store.documents) == 3, (
            f"Expected loaded document count to be 3, got {len(loaded_store.documents)}."
        )
        assert loaded_store.dimension == 3, (
            f"Expected loaded dimension to be 3, got {loaded_store.dimension}."
        )
        assert loaded_store.documents == documents, (
            "Loaded documents do not match original documents."
        )
        print("✓ Verified loaded vector count (3), document count (3), dimension (3), and document contents.")

        # 11. Search the loaded store using the same query embedding
        loaded_results = loaded_store.search(query_embedding, top_k=3)

        # 12. Verify loaded search results match original search results
        assert len(loaded_results) == len(original_results), (
            f"Result length mismatch: {len(loaded_results)} != {len(original_results)}."
        )

        for i, (orig, loaded) in enumerate(zip(original_results, loaded_results)):
            assert orig["document"] == loaded["document"], (
                f"Document ordering or content mismatch at index {i}: "
                f"{orig['document']} vs {loaded['document']}."
            )
            assert np.isclose(orig["distance"], loaded["distance"], atol=1e-5), (
                f"Distance mismatch at index {i}: {orig['distance']} vs {loaded['distance']}."
            )

        print("✓ Verified search results match in document ordering, contents, and distances.")

    print("\nAll VectorStore persistence tests passed successfully!")


if __name__ == "__main__":
    test_vector_store_persistence()
