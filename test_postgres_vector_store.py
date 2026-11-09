"""
Test Suite for PostgreSQL + pgvector Vector Store Implementation.

Validates:
1. Configuration and credential masking (no secrets in repr or logs)
2. Dimension and input parameter validation
3. Schema initialization (CREATE EXTENSION vector, CREATE TABLE, CREATE INDEX)
4. Document and vector insertion (batching, vector format conversion)
5. Nearest-neighbor similarity search (SQL construction with L2 distance operator)
6. top_k behavior
7. max_distance filtering
8. source metadata filtering
9. document_id metadata filtering
10. delete_document() cleanup and return values
11. Returned metadata format compatibility with Retriever and BaseVectorStore
12. Live database detection and execution if available, with graceful non-faked reporting if absent
"""

import os
from unittest.mock import MagicMock, patch
import numpy as np
import psycopg

from src.db_config import get_database_url, mask_database_url
from src.postgres_vector_store import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_TABLE_NAME,
    PostgresVectorStore,
    initialize_schema,
)
from src.vector_store_factory import create_vector_store, get_vector_store_type


def check_live_postgres_available() -> bool:
    """Check if a live PostgreSQL instance with pgvector is accessible."""
    try:
        url = get_database_url()
        with psycopg.connect(url, connect_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
        return True
    except Exception:
        return False


# ==============================================================================
# Unit & Mock-Level Validation Suite (Always executed)
# ==============================================================================

def test_config_and_masking() -> None:
    """Validate database URL masking and credential security."""
    raw_url = "postgresql://myuser:supersecretpass@db.example.com:5432/ragdb"
    masked = mask_database_url(raw_url)
    assert "supersecretpass" not in masked, "Secret password leaked in masked URL!"
    assert ":****@" in masked

    store = PostgresVectorStore(connection_url=raw_url, auto_init_schema=False)
    store_repr = repr(store)
    assert "supersecretpass" not in store_repr, "Secret leaked in PostgresVectorStore repr!"
    assert ":****@" in store_repr
    print("✓ Test 1: Credential masking verified.")


def test_input_validation() -> None:
    """Validate dimension, argument, and count checks."""
    try:
        PostgresVectorStore(dimension=-5, auto_init_schema=False)
        assert False, "Expected ValueError on negative dimension"
    except ValueError as exc:
        assert "Dimension must be positive" in str(exc)

    store = PostgresVectorStore(auto_init_schema=False)

    # Document count mismatch
    try:
        docs = [{"text": "hello"}]
        embs = np.zeros((2, DEFAULT_EMBEDDING_DIMENSION), dtype=np.float32)
        store.add_documents(docs, embs)
        assert False, "Expected ValueError on count mismatch"
    except ValueError as exc:
        assert "Count mismatch" in str(exc)

    # Embedding dimension mismatch
    try:
        docs = [{"text": "hello"}]
        embs = np.zeros((1, 100), dtype=np.float32)
        store.add_documents(docs, embs)
        assert False, "Expected ValueError on dimension mismatch"
    except ValueError as exc:
        assert "Expected embeddings shape" in str(exc)

    # Search top_k validation
    try:
        store.search(np.zeros((DEFAULT_EMBEDDING_DIMENSION,)), top_k=0)
        assert False, "Expected ValueError on top_k <= 0"
    except ValueError as exc:
        assert "top_k must be greater than 0" in str(exc)

    # Search negative max_distance validation
    try:
        store.search(np.zeros((DEFAULT_EMBEDDING_DIMENSION,)), max_distance=-0.5)
        assert False, "Expected ValueError on negative max_distance"
    except ValueError as exc:
        assert "max_distance must be greater than or equal to 0" in str(exc)

    print("✓ Test 2: Input and dimension validations verified.")


def test_mocked_schema_initialization() -> None:
    """Validate idempotent schema SQL and vector extension enablement."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    with patch("psycopg.connect", return_value=mock_conn):
        initialize_schema(connection_url="postgresql://user:pass@localhost:5432/testdb", table_name="custom_chunks")

    executed_statements = [call[0][0] for call in mock_cur.execute.call_args_list]
    assert any("CREATE EXTENSION IF NOT EXISTS vector;" in s for s in executed_statements)
    assert any("CREATE TABLE IF NOT EXISTS custom_chunks" in s for s in executed_statements)
    assert any("idx_custom_chunks_document_id" in s for s in executed_statements)
    assert any("idx_custom_chunks_source" in s for s in executed_statements)
    print("✓ Test 3: Schema initialization SQL verified.")


def test_mocked_document_insertion() -> None:
    """Validate batch insertion SQL generation and vector mapping."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    store = PostgresVectorStore(auto_init_schema=False)
    docs = [
        {
            "document_id": "doc_1",
            "source": "sample.pdf",
            "page_number": 1,
            "chunk_id": "doc_1_chunk_0",
            "text": "First chunk text.",
        },
        {
            "document_id": "doc_1",
            "source": "sample.pdf",
            "page_number": 2,
            "chunk_id": "doc_1_chunk_1",
            "text": "Second chunk text.",
        },
    ]
    embs = np.ones((2, DEFAULT_EMBEDDING_DIMENSION), dtype=np.float32)

    with patch.object(store, "_connect", return_value=mock_conn):
        store.add_documents(docs, embs)

    mock_cur.executemany.assert_called_once()
    sql_call, rows_call = mock_cur.executemany.call_args[0]
    assert "INSERT INTO document_chunks" in sql_call
    assert len(rows_call) == 2
    assert rows_call[0][0] == "doc_1"
    assert rows_call[0][1] == "sample.pdf"
    assert rows_call[0][2] == 1
    assert rows_call[0][3] == "doc_1_chunk_0"
    assert rows_call[0][4] == "First chunk text."
    assert isinstance(rows_call[0][5], list)
    assert len(rows_call[0][5]) == DEFAULT_EMBEDDING_DIMENSION
    print("✓ Test 4: Document and vector insertion verified.")


def test_mocked_search_sql_and_filtering() -> None:
    """Validate search SQL generation with L2 distance, top_k, max_distance, and metadata filters."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    mock_cur.fetchall.return_value = [
        ("doc_1", "sample.pdf", 1, "doc_1_chunk_0", "First chunk text.", 0.25),
        ("doc_1", "sample.pdf", 2, "doc_1_chunk_1", "Second chunk text.", 0.64),
    ]

    store = PostgresVectorStore(auto_init_schema=False)
    query_vec = np.zeros((DEFAULT_EMBEDDING_DIMENSION,), dtype=np.float32)

    with patch.object(store, "_connect", return_value=mock_conn):
        results = store.search(
            query_embedding=query_vec,
            top_k=2,
            max_distance=1.10,
            source="sample.pdf",
            document_id="doc_1",
        )

    mock_cur.execute.assert_called_once()
    sql_call, params_call = mock_cur.execute.call_args[0]

    # Verify SQL characteristics
    assert "((embedding <-> %s) * (embedding <-> %s)) AS distance" in sql_call
    assert "source = %s" in sql_call
    assert "document_id = %s" in sql_call
    assert "((embedding <-> %s) * (embedding <-> %s)) <= %s" in sql_call
    assert "ORDER BY (embedding <-> %s) ASC" in sql_call
    assert "LIMIT %s" in sql_call

    # Verify result structure compatibility with Retriever
    assert len(results) == 2
    assert results[0]["document"]["source"] == "sample.pdf"
    assert results[0]["document"]["page_number"] == 1
    assert results[0]["document"]["document_id"] == "doc_1"
    assert results[0]["document"]["chunk_id"] == "doc_1_chunk_0"
    assert results[0]["document"]["text"] == "First chunk text."
    assert results[0]["distance"] == 0.25
    print("✓ Test 5: Search SQL generation and returned metadata structure verified.")


def test_mocked_delete_document() -> None:
    """Validate document deletion SQL and row count reporting."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_cur.rowcount = 5

    store = PostgresVectorStore(auto_init_schema=False)

    with patch.object(store, "_connect", return_value=mock_conn):
        deleted = store.delete_document("doc_target_to_delete")

    assert deleted == 5
    mock_cur.execute.assert_called_once()
    sql_call, params_call = mock_cur.execute.call_args[0]
    assert "DELETE FROM document_chunks WHERE document_id = %s;" in sql_call
    assert params_call == ("doc_target_to_delete",)
    print("✓ Test 6: Document deletion verified.")


def test_factory_vector_store_selection() -> None:
    """Validate factory selection between 'faiss' and 'postgres'."""
    # Default is faiss
    with patch.dict(os.environ, {}, clear=True):
        assert get_vector_store_type() == "faiss"
        default_store = create_vector_store()
        assert default_store.__class__.__name__ == "VectorStore"

    # Explicit postgres selection
    with patch.dict(os.environ, {"VECTOR_STORE": "postgres"}, clear=True):
        assert get_vector_store_type() == "postgres"
        pg_store = create_vector_store(auto_init_schema=False)
        assert pg_store.__class__.__name__ == "PostgresVectorStore"

    print("✓ Test 7: Vector store factory resolution verified.")


def test_rag_pipeline_with_postgres_store() -> None:
    """Validate full RAGGenerator + Retriever + PostgresVectorStore integration."""
    from src.embedding_generator import EmbeddingGenerator
    from src.retriever import Retriever
    from src.rag_generator import RAGGenerator
    from src.llm_provider_interface import BaseLLMProvider

    class StubLLM(BaseLLMProvider):
        def generate(self, prompt: str, temperature: float = 0.0, max_tokens = None) -> str:
            return "Juspay powers enterprise payment infrastructure at scale."

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    mock_cur.fetchall.return_value = [
        ("doc_1", "sample.pdf", 1, "doc_1_chunk_0", "About Juspay: payments technology company.", 0.45),
    ]

    store = PostgresVectorStore(auto_init_schema=False)
    with patch.object(store, "_connect", return_value=mock_conn):
        emb_gen = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        retriever = Retriever(vector_store=store, embedding_generator=emb_gen)
        rag_gen = RAGGenerator(retriever=retriever, default_max_distance=1.10, llm_provider=StubLLM())

        res = rag_gen.generate_answer("What does Juspay do?", top_k=1)
        assert res["answer"] == "Juspay powers enterprise payment infrastructure at scale."
        assert len(res["sources"]) == 1
        assert res["sources"][0]["source"] == "sample.pdf"
        assert res["sources"][0]["page_number"] == 1
        assert res["sources"][0]["distance"] == 0.45

    print("✓ Test 8: End-to-end RAG pipeline with PostgresVectorStore verified.")


# ==============================================================================
# Live Database Execution (Executed only if PostgreSQL is accessible)
# ==============================================================================

def run_live_postgres_tests() -> None:
    """Run end-to-end integration tests if live PostgreSQL is detected."""
    print("\n--- Running Live PostgreSQL + pgvector Test Suite ---")
    test_table = "test_document_chunks_parity"
    test_doc_id = "test_isolated_doc_999"

    url = get_database_url()
    initialize_schema(connection_url=url, table_name=test_table)
    store = PostgresVectorStore(connection_url=url, table_name=test_table, auto_init_schema=False)

    try:
        # Clean any preexisting test artifacts
        store.delete_document(test_doc_id)

        # 1. Insert test documents
        docs = [
            {"document_id": test_doc_id, "source": "test.pdf", "page_number": 1, "chunk_id": f"{test_doc_id}_c0", "text": "Test chunk 0"},
            {"document_id": test_doc_id, "source": "test.pdf", "page_number": 2, "chunk_id": f"{test_doc_id}_c1", "text": "Test chunk 1"},
        ]
        embs = np.random.randn(2, DEFAULT_EMBEDDING_DIMENSION).astype(np.float32)
        store.add_documents(docs, embs)

        # 2. Search
        res = store.search(embs[0], top_k=2, document_id=test_doc_id)
        assert len(res) == 2
        assert res[0]["document"]["chunk_id"] == f"{test_doc_id}_c0"
        assert abs(res[0]["distance"]) < 1e-4

        # 3. Cleanup
        deleted = store.delete_document(test_doc_id)
        assert deleted == 2
        print("✓ Live PostgreSQL + pgvector integration tests passed successfully.")
    finally:
        # Drop temporary test table
        try:
            with psycopg.connect(url) as conn:
                with conn.cursor() as cur:
                    cur.execute(f"DROP TABLE IF EXISTS {test_table};")
                conn.commit()
        except Exception:
            pass


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 10.3: POSTGRESQL + PGVECTOR VECTOR STORE TESTS")
    print("=" * 70)

    test_config_and_masking()
    test_input_validation()
    test_mocked_schema_initialization()
    test_mocked_document_insertion()
    test_mocked_search_sql_and_filtering()
    test_mocked_delete_document()
    test_factory_vector_store_selection()
    test_rag_pipeline_with_postgres_store()

    print("\n--- Live PostgreSQL Availability Check ---")
    live_available = check_live_postgres_available()
    if live_available:
        print("Live PostgreSQL instance detected! Running live integration tests...")
        run_live_postgres_tests()
    else:
        print("Live PostgreSQL is NOT running/accessible in this environment.")
        print("(Mocked & unit-level validations passed. Live database validation is pending.")
        print(" Per architectural instructions, live tests are not faked.)")

    print("\n" + "=" * 70)
    print("POSTGRES VECTOR STORE TESTS FINISHED.")
    print("=" * 70)
