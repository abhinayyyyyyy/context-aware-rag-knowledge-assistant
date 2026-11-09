import numpy as np
from src.vector_store import VectorStore


def test_vector_store_metadata_filter() -> None:
    # 3. Create VectorStore with dimension 3
    dimension = 3
    store = VectorStore(dimension=dimension)

    # 4. Create deterministic embeddings for 5 documents
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],  # Doc A - chunk 0 (distance to [1,0,0]: 0.0)
            [0.9, 0.1, 0.0],  # Doc A - chunk 1 (distance to [1,0,0]: 0.02)
            [0.8, 0.2, 0.0],  # Doc A - chunk 2 (distance to [1,0,0]: 0.08)
            [0.0, 0.9, 0.1],  # Doc B - chunk 0 (distance to [1,0,0]: 1.82)
            [0.0, 1.0, 0.0],  # Doc B - chunk 1 (distance to [1,0,0]: 2.00)
        ],
        dtype=np.float32,
    )

    doc_a_id = "doc_a_sha256_hash"
    doc_b_id = "doc_b_sha256_hash"

    # 5. Create corresponding document metadata
    documents = [
        {
            "text": "Document A introductory section.",
            "page_number": 1,
            "source": "document_a.pdf",
            "document_id": doc_a_id,
            "chunk_id": f"{doc_a_id}_chunk_0",
        },
        {
            "text": "Document A details section.",
            "page_number": 2,
            "source": "document_a.pdf",
            "document_id": doc_a_id,
            "chunk_id": f"{doc_a_id}_chunk_1",
        },
        {
            "text": "Document A summary section.",
            "page_number": 3,
            "source": "document_a.pdf",
            "document_id": doc_a_id,
            "chunk_id": f"{doc_a_id}_chunk_2",
        },
        {
            "text": "Document B overview section.",
            "page_number": 1,
            "source": "document_b.pdf",
            "document_id": doc_b_id,
            "chunk_id": f"{doc_b_id}_chunk_0",
        },
        {
            "text": "Document B conclusion section.",
            "page_number": 2,
            "source": "document_b.pdf",
            "document_id": doc_b_id,
            "chunk_id": f"{doc_b_id}_chunk_1",
        },
    ]

    # 6. Add all documents and embeddings to ONE VectorStore
    store.add_documents(documents, embeddings)
    query_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # 7. Test normal search
    print("Testing normal unfiltered search (top_k=3)...")
    unfiltered_results = store.search(query_vector, top_k=3)
    assert len(unfiltered_results) == 3, f"Expected 3 results, got {len(unfiltered_results)}."
    print("✓ Normal search verified: exactly 3 results returned.")

    # 8 & 9. Test source filtering and verify filtering happens BEFORE top_k selection
    print("Testing source filtering and pre-top_k candidate evaluation...")
    # The global top 2 results are both from document_a.pdf.
    # Searching with top_k=2 and source='document_b.pdf' must return 2 results from document_b.pdf.
    global_top_2_sources = [r["document"]["source"] for r in store.search(query_vector, top_k=2)]
    assert global_top_2_sources == ["document_a.pdf", "document_a.pdf"], (
        f"Global top 2 should be Document A, got {global_top_2_sources}."
    )

    doc_b_results = store.search(query_vector, top_k=2, source="document_b.pdf")
    assert len(doc_b_results) == 2, (
        f"Expected 2 results from document_b.pdf, got {len(doc_b_results)}."
    )
    assert all(r["document"]["source"] == "document_b.pdf" for r in doc_b_results), (
        "Found results not belonging to document_b.pdf."
    )
    assert all(r["document"]["source"] != "document_a.pdf" for r in doc_b_results), (
        "Found results belonging to document_a.pdf."
    )
    # Verify distance ordering in filtered results
    assert doc_b_results[0]["distance"] <= doc_b_results[1]["distance"], (
        "Filtered results are not in ascending distance order."
    )
    assert doc_b_results[0]["document"]["chunk_id"] == f"{doc_b_id}_chunk_0"
    assert doc_b_results[1]["document"]["chunk_id"] == f"{doc_b_id}_chunk_1"
    print("✓ Source filtering verified: returned 2 Document B results ordered by distance, proving pre-top_k filtering.")

    # 10. Test document_id filtering
    print("Testing document_id filtering...")
    doc_id_results = store.search(query_vector, top_k=3, document_id=doc_b_id)
    assert len(doc_id_results) == 2, f"Expected 2 results for {doc_b_id}, got {len(doc_id_results)}."
    assert all(r["document"]["document_id"] == doc_b_id for r in doc_id_results), (
        f"Result does not match expected document_id {doc_b_id}."
    )
    print("✓ Document ID filtering verified: every result has the requested document_id.")

    # 11. Test combined filtering
    print("Testing combined source and document_id filtering...")
    combined_matching = store.search(
        query_vector,
        top_k=3,
        source="document_b.pdf",
        document_id=doc_b_id,
    )
    assert len(combined_matching) == 2, f"Expected 2 matching results, got {len(combined_matching)}."
    for res in combined_matching:
        assert res["document"]["source"] == "document_b.pdf"
        assert res["document"]["document_id"] == doc_b_id

    # Test conflicting filters where no document matches both
    combined_mismatch = store.search(
        query_vector,
        top_k=3,
        source="document_a.pdf",
        document_id=doc_b_id,
    )
    assert len(combined_mismatch) == 0, (
        f"Expected 0 results for conflicting filters, got {len(combined_mismatch)}."
    )
    print("✓ Combined filtering verified: requires both source and document_id to match.")

    # 12. Test invalid source validation
    print("Testing invalid source validation...")
    try:
        store.search(query_vector, source="")
        assert False, "Expected ValueError for empty source."
    except ValueError as err:
        assert str(err) == "source must be a non-empty string when provided.", (
            f"Unexpected error message: {err}"
        )
        print(f"✓ Invalid source properly rejected: {err}")

    # 13. Test invalid document_id validation
    print("Testing invalid document_id validation...")
    try:
        store.search(query_vector, document_id="")
        assert False, "Expected ValueError for empty document_id."
    except ValueError as err:
        assert str(err) == "document_id must be a non-empty string when provided.", (
            f"Unexpected error message: {err}"
        )
        print(f"✓ Invalid document_id properly rejected: {err}")

    # 14. Test backward compatibility
    print("Testing backward compatibility without filters...")
    default_results = store.search(query_vector, top_k=2)
    assert len(default_results) == 2, f"Expected 2 results, got {len(default_results)}."
    assert default_results[0]["document"]["chunk_id"] == f"{doc_a_id}_chunk_0"
    assert default_results[1]["document"]["chunk_id"] == f"{doc_a_id}_chunk_1"
    print("✓ Backward compatibility verified: search(query, top_k) behaves as expected.")

    print("\nAll VectorStore metadata filter tests passed successfully!")


if __name__ == "__main__":
    test_vector_store_metadata_filter()
