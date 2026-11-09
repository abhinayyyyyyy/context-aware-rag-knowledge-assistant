from pathlib import Path
import tempfile
import pypdf
from src.embedding_generator import EmbeddingGenerator
from src.ingestion_pipeline import build_multi_document_vector_store
from src.vector_store import VectorStore


def test_multi_document_ingestion() -> None:
    # 3. Locate existing sample.pdf
    first_pdf_path = Path("data/documents/sample.pdf").resolve()
    assert first_pdf_path.is_file(), f"Sample PDF not found at: {first_pdf_path}"

    with tempfile.TemporaryDirectory() as temp_dir:
        # 4. Create second valid temporary PDF with different text using pypdf
        second_pdf_path = Path(temp_dir) / "second_document.pdf"
        reader = pypdf.PdfReader(str(first_pdf_path))
        writer = pypdf.PdfWriter()
        # Extract page 2 (index 1) to create distinct content and metadata
        writer.add_page(reader.pages[1])
        with open(second_pdf_path, "wb") as f:
            writer.write(f)

        assert second_pdf_path.is_file(), f"Second PDF was not created: {second_pdf_path}"

        # 13. Create one reusable EmbeddingGenerator instance
        embedding_generator = EmbeddingGenerator()

        # 5. Build multi-document vector store using both PDFs and shared generator
        print("Ingesting multiple PDF documents into shared VectorStore...")
        vector_store = build_multi_document_vector_store(
            [str(first_pdf_path), str(second_pdf_path)],
            embedding_generator=embedding_generator,
        )

        # 6. Verify VectorStore instance and vector counts
        assert isinstance(vector_store, VectorStore), (
            f"Expected VectorStore instance, got {type(vector_store).__name__}."
        )
        assert vector_store.index.ntotal > 0, "VectorStore index is empty."
        assert len(vector_store.documents) == vector_store.index.ntotal, (
            f"Document count ({len(vector_store.documents)}) does not match "
            f"vector count ({vector_store.index.ntotal})."
        )
        print(f"✓ Verified VectorStore populated: {vector_store.index.ntotal} vectors and documents.")

        # 7. Verify chunks from both source documents are present
        sources = {doc["source"] for doc in vector_store.documents}
        assert first_pdf_path.name in sources, f"Missing chunks from {first_pdf_path.name}"
        assert second_pdf_path.name in sources, f"Missing chunks from {second_pdf_path.name}"
        print(f"✓ Verified chunks from both source documents are present: {sources}.")

        # 8. Verify required metadata fields in every chunk
        required_keys = {"text", "page_number", "source", "document_id", "chunk_id"}
        for idx, doc in enumerate(vector_store.documents):
            for key in required_keys:
                assert key in doc and doc[key] is not None, (
                    f"Chunk {idx} missing or empty metadata field '{key}': {doc}"
                )
        print("✓ Verified all chunks contain text, page_number, source, document_id, and chunk_id.")

        # 9 & 10. Verify document_id separation across sources
        doc_ids = {doc["document_id"] for doc in vector_store.documents}
        assert len(doc_ids) >= 2, f"Expected at least 2 distinct document_ids, got {len(doc_ids)}."

        source_to_doc_ids = {}
        for doc in vector_store.documents:
            source_to_doc_ids.setdefault(doc["source"], set()).add(doc["document_id"])

        first_id_set = source_to_doc_ids[first_pdf_path.name]
        second_id_set = source_to_doc_ids[second_pdf_path.name]
        assert len(first_id_set) == 1, f"Expected 1 document_id for first PDF, got {first_id_set}."
        assert len(second_id_set) == 1, f"Expected 1 document_id for second PDF, got {second_id_set}."
        assert first_id_set != second_id_set, "Both documents share the same document_id."
        print("✓ Verified distinct document_ids generated and segregated by source.")

        # 11. Verify uniqueness of all chunk IDs across the entire store
        all_chunk_ids = [doc["chunk_id"] for doc in vector_store.documents]
        assert len(all_chunk_ids) == len(set(all_chunk_ids)), (
            "Chunk IDs are not unique across the combined vector store."
        )
        print("✓ Verified all chunk IDs are globally unique across the vector store.")

        # 12. Verify each document's chunk numbering starts from <document_id>_chunk_0
        for doc_id in doc_ids:
            doc_chunks = [d for d in vector_store.documents if d["document_id"] == doc_id]
            expected_first_chunk_id = f"{doc_id}_chunk_0"
            assert doc_chunks[0]["chunk_id"] == expected_first_chunk_id, (
                f"Expected first chunk to be '{expected_first_chunk_id}', got '{doc_chunks[0]['chunk_id']}'."
            )
        print("✓ Verified chunk numbering starts from <document_id>_chunk_0 for each document.")

        # 14. Test empty input list validation
        print("Testing empty pdf_paths validation...")
        try:
            build_multi_document_vector_store([])
            assert False, "Expected ValueError when pdf_paths is empty."
        except ValueError as err:
            assert str(err) == "pdf_paths must contain at least one PDF file.", (
                f"Unexpected error message: {err}"
            )
            print(f"✓ Empty input validation verified: {err}")

        # 15. Test missing PDF path raises FileNotFoundError
        print("Testing missing PDF path validation...")
        missing_path = str(Path(temp_dir) / "non_existent.pdf")
        try:
            build_multi_document_vector_store([missing_path])
            assert False, "Expected FileNotFoundError when PDF path does not exist."
        except FileNotFoundError as err:
            print(f"✓ Missing PDF path properly raised FileNotFoundError: {err}")

    print("\nAll multi-document ingestion tests passed successfully!")


if __name__ == "__main__":
    test_multi_document_ingestion()
