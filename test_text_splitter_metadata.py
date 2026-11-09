import hashlib
from src.text_splitter import split_documents


def test_text_splitter_metadata() -> None:
    # 2 & 10. In-memory pages including two pages for doc_a, one for doc_b, and empty pages
    doc_a_name = "document_a.pdf"
    doc_b_name = "document_b.pdf"

    sample_text_page1 = (
        "This is the first page of document A covering foundational concepts in information retrieval. "
        "It provides introductory context for semantic indexing and vector storage."
    )
    sample_text_page2 = (
        "This is the second page of document A expanding on document chunking and metadata attribution. "
        "It explains chunk identifiers and document tracking."
    )
    sample_text_page3 = (
        "This is page one of document B presenting an entirely different topic on large language models. "
        "It discusses prompt engineering and context grounding."
    )

    documents = [
        {"text": sample_text_page1, "page_number": 1, "source": doc_a_name},
        {"text": "   \n\t  ", "page_number": 2, "source": doc_a_name},  # Whitespace-only page
        {"text": "", "page_number": 3, "source": doc_a_name},           # Empty page
        {"text": sample_text_page2, "page_number": 4, "source": doc_a_name},
        {"text": sample_text_page3, "page_number": 1, "source": doc_b_name},
    ]

    # 3. Call split_documents
    chunks = split_documents(documents)
    assert len(chunks) > 0, "Expected chunks to be generated."

    # 4. Verify that every generated chunk contains all expected metadata fields
    required_keys = {"text", "page_number", "source", "document_id", "chunk_id"}
    for idx, chunk in enumerate(chunks):
        for key in required_keys:
            assert key in chunk, f"Chunk {idx} missing required key '{key}': {chunk}"
        assert isinstance(chunk["text"], str) and chunk["text"].strip(), f"Chunk {idx} has empty text."
        assert isinstance(chunk["page_number"], int), f"Chunk {idx} page_number is not int."
        assert isinstance(chunk["source"], str), f"Chunk {idx} source is not str."
        assert isinstance(chunk["document_id"], str), f"Chunk {idx} document_id is not str."
        assert isinstance(chunk["chunk_id"], str), f"Chunk {idx} chunk_id is not str."
    print("✓ Metadata fields verified: all chunks contain text, page_number, source, document_id, and chunk_id.")

    # 5. Verify document_id generation against independent SHA-256 calculation
    expected_doc_a_id = hashlib.sha256(doc_a_name.encode("utf-8")).hexdigest()
    expected_doc_b_id = hashlib.sha256(doc_b_name.encode("utf-8")).hexdigest()

    doc_a_chunks = [c for c in chunks if c["source"] == doc_a_name]
    doc_b_chunks = [c for c in chunks if c["source"] == doc_b_name]

    assert len(doc_a_chunks) >= 2, f"Expected at least 2 chunks for document_a, got {len(doc_a_chunks)}."
    assert len(doc_b_chunks) >= 1, f"Expected at least 1 chunk for document_b, got {len(doc_b_chunks)}."

    for chunk in doc_a_chunks:
        assert chunk["document_id"] == expected_doc_a_id, (
            f"Expected document_id '{expected_doc_a_id}', got '{chunk['document_id']}'."
        )

    for chunk in doc_b_chunks:
        assert chunk["document_id"] == expected_doc_b_id, (
            f"Expected document_id '{expected_doc_b_id}', got '{chunk['document_id']}'."
        )
    print("✓ Document ID generation verified: SHA-256 digests match expected hexadecimal hashes.")

    # 6 & 7. Verify chunk ID format and zero-based sequential numbering per document
    for doc_name, doc_id, doc_chunks in [
        (doc_a_name, expected_doc_a_id, doc_a_chunks),
        (doc_b_name, expected_doc_b_id, doc_b_chunks),
    ]:
        for i, chunk in enumerate(doc_chunks):
            expected_chunk_id = f"{doc_id}_chunk_{i}"
            assert chunk["chunk_id"] == expected_chunk_id, (
                f"For {doc_name} chunk {i}: expected chunk_id '{expected_chunk_id}', got '{chunk['chunk_id']}'."
            )
        assert doc_chunks[0]["chunk_id"].endswith("_chunk_0"), (
            f"First chunk for {doc_name} does not end with _chunk_0: {doc_chunks[0]['chunk_id']}"
        )
    print("✓ Chunk ID format and numbering verified: zero-based, sequential, restarting per source document.")

    # 8. Verify determinism across multiple runs
    chunks_second_run = split_documents(documents)
    assert len(chunks) == len(chunks_second_run), "Chunk count differed across separate runs."
    for c1, c2 in zip(chunks, chunks_second_run):
        assert c1["document_id"] == c2["document_id"], "document_id differed between runs."
        assert c1["chunk_id"] == c2["chunk_id"], "chunk_id differed between runs."
        assert c1["text"] == c2["text"], "chunk text differed between runs."
    print("✓ Determinism verified: duplicate runs produced identical document_id and chunk_id values.")

    # 9. Verify source separation
    assert expected_doc_a_id != expected_doc_b_id, "Different sources produced identical document_ids."
    doc_a_ids = {c["document_id"] for c in doc_a_chunks}
    doc_b_ids = {c["document_id"] for c in doc_b_chunks}
    assert doc_a_ids.isdisjoint(doc_b_ids), "Document IDs for distinct sources overlapped."
    print("✓ Source separation verified: distinct filenames have distinct document_id values.")

    # 10. Verify empty text handling
    empty_pages_in_chunks = [
        c for c in chunks if c["source"] == doc_a_name and c["page_number"] in (2, 3)
    ]
    assert len(empty_pages_in_chunks) == 0, (
        f"Found chunks generated for empty/whitespace pages: {empty_pages_in_chunks}"
    )
    print("✓ Empty text handling verified: empty and whitespace-only pages produce no chunks.")

    print("\nAll text splitter metadata tests passed successfully!")


if __name__ == "__main__":
    test_text_splitter_metadata()
