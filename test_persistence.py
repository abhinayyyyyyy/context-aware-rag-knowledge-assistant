"""
Test Suite for Persistent Application State with PostgreSQL.

Validates:
1. Persistence-disabled behavior (default local mode)
2. Content hash computation (SHA-256)
3. Schema creation DDL (documents, conversations, messages, cascades, indexes)
4. Document creation and content-hash deduplication
5. Conversation creation and linking to documents
6. User message persistence
7. Assistant message persistence with JSONB source citations
8. Chronological ordered conversation retrieval
9. Conversation restoration (reconstructing visible session state)
10. Cascade deletion behavior
11. Live database integration if accessible; truthful reporting if offline
"""

import os
from unittest.mock import MagicMock, patch
import psycopg

from src.database import (
    ConversationRepository,
    DatabaseService,
    DocumentRepository,
    MessageRepository,
    compute_content_hash,
    init_persistence_schema,
)
from src.db_config import get_database_url, is_persistence_enabled


def check_live_postgres_available() -> bool:
    """Check if a live PostgreSQL connection is accessible."""
    try:
        url = get_database_url()
        with psycopg.connect(url, connect_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
        return True
    except Exception:
        return False


def test_persistence_disabled_by_default() -> None:
    """Verify that persistence is disabled by default."""
    with patch.dict(os.environ, {}, clear=True):
        assert not is_persistence_enabled(), "Persistence should be False when env is empty."

    with patch.dict(os.environ, {"PERSISTENCE_ENABLED": "false"}):
        assert not is_persistence_enabled()

    with patch.dict(os.environ, {"PERSISTENCE_ENABLED": "true"}):
        assert is_persistence_enabled()

    print("✓ Test 1: Persistence configuration default behavior verified.")


def test_content_hash_deterministic() -> None:
    """Verify SHA-256 hash calculation on file bytes."""
    data_1 = b"PDF content sample 1"
    data_2 = b"PDF content sample 1"
    data_diff = b"Different content"

    hash_1 = compute_content_hash(data_1)
    hash_2 = compute_content_hash(data_2)
    hash_diff = compute_content_hash(data_diff)

    assert hash_1 == hash_2, "Same content should yield identical hash."
    assert hash_1 != hash_diff, "Different content should yield different hash."
    assert len(hash_1) == 64, "SHA-256 hex digest should be 64 characters."
    print("✓ Test 2: Content hash computation verified.")


def test_mocked_schema_creation() -> None:
    """Verify idempotent table and index creation SQL statements."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    with patch("psycopg.connect", return_value=mock_conn):
        init_persistence_schema("postgresql://user:pass@localhost:5432/testdb")

    mock_cur.execute.assert_called_once()
    ddl_sql = mock_cur.execute.call_args[0][0]

    assert "CREATE TABLE IF NOT EXISTS documents" in ddl_sql
    assert "CREATE TABLE IF NOT EXISTS conversations" in ddl_sql
    assert "CREATE TABLE IF NOT EXISTS messages" in ddl_sql
    assert "REFERENCES documents(id) ON DELETE CASCADE" in ddl_sql
    assert "REFERENCES conversations(id) ON DELETE CASCADE" in ddl_sql
    assert "idx_documents_content_hash" in ddl_sql
    assert "idx_conversations_document_id" in ddl_sql
    assert "idx_messages_conversation_id" in ddl_sql
    print("✓ Test 3: Relational persistence schema DDL verified.")


def test_mocked_document_creation_and_deduplication() -> None:
    """Verify document creation and reuse when content_hash matches existing file."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    repo = DocumentRepository("postgresql://user:pass@localhost:5432/testdb")

    # 1. New document insertion (get_by_content_hash returns None)
    mock_cur.fetchone.side_effect = [
        None,  # get_by_content_hash
        ("2026-10-07 12:00:00",),  # RETURNING created_at
    ]

    with patch.object(repo, "_connect", return_value=mock_conn):
        doc = repo.create_or_get_document(
            filename="sample.pdf",
            content_hash="abc123hash",
            file_size_bytes=1024,
            storage_path="data/sample.pdf",
            chunk_count=12,
        )

    assert doc["filename"] == "sample.pdf"
    assert doc["content_hash"] == "abc123hash"
    assert doc["chunk_count"] == 12

    # 2. Existing document reuse (get_by_content_hash finds existing record)
    mock_cur.fetchone.side_effect = [
        ("doc-uuid-1", "sample.pdf", "abc123hash", 1024, "data/sample.pdf", 12, "2026-10-07 12:00:00"),
    ]

    with patch.object(repo, "_connect", return_value=mock_conn):
        existing_doc = repo.create_or_get_document(
            filename="sample_renamed.pdf",
            content_hash="abc123hash",
            file_size_bytes=1024,
            storage_path="new_path/sample.pdf",
            chunk_count=12,
        )

    assert existing_doc["id"] == "doc-uuid-1"
    assert existing_doc["content_hash"] == "abc123hash"
    assert existing_doc["storage_path"] == "new_path/sample.pdf"
    print("✓ Test 4: Document creation and content hash deduplication verified.")


def test_mocked_conversation_creation() -> None:
    """Verify conversation record creation linked to document."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_cur.fetchone.return_value = ("2026-10-07 12:00:00", "2026-10-07 12:00:00")

    repo = ConversationRepository("postgresql://user:pass@localhost:5432/testdb")
    with patch.object(repo, "_connect", return_value=mock_conn):
        conv = repo.create_conversation("doc-uuid-1", title="Chat: sample.pdf")

    assert conv["document_id"] == "doc-uuid-1"
    assert conv["title"] == "Chat: sample.pdf"
    assert len(conv["id"]) > 10  # valid uuid string
    print("✓ Test 5: Conversation creation verified.")


def test_mocked_message_persistence_and_sources() -> None:
    """Verify persisting user and assistant messages with source citations."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_cur.fetchone.return_value = ("2026-10-07 12:01:00",)

    repo = MessageRepository("postgresql://user:pass@localhost:5432/testdb")

    # User message
    with patch.object(repo, "_connect", return_value=mock_conn):
        user_msg = repo.create_message(
            conversation_id="conv-uuid-1",
            role="user",
            content="What does Juspay do?",
        )
    assert user_msg["role"] == "user"
    assert user_msg["content"] == "What does Juspay do?"
    assert user_msg["sources"] == []

    # Assistant message with sources
    sources = [
        {"source": "sample.pdf", "page_number": 1, "distance": 1.0023, "chunk_id": "chunk_0"}
    ]
    with patch.object(repo, "_connect", return_value=mock_conn):
        asst_msg = repo.create_message(
            conversation_id="conv-uuid-1",
            role="assistant",
            content="Juspay is a payments technology company.",
            sources=sources,
        )
    assert asst_msg["role"] == "assistant"
    assert asst_msg["sources"][0]["page_number"] == 1
    assert asst_msg["sources"][0]["distance"] == 1.0023
    print("✓ Test 6: Message persistence and JSONB sources verified.")


def test_mocked_conversation_restoration() -> None:
    """Verify retrieving ordered messages and reconstructing visible chat state."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    mock_cur.fetchall.return_value = [
        (
            "msg-1", "conv-uuid-1", "user", "What does Juspay do?", [], "2026-10-07 12:01:00"
        ),
        (
            "msg-2",
            "conv-uuid-1",
            "assistant",
            "Juspay is a payments platform.",
            [{"source": "sample.pdf", "page_number": 1, "distance": 1.0023}],
            "2026-10-07 12:01:05",
        ),
    ]

    repo = MessageRepository("postgresql://user:pass@localhost:5432/testdb")
    with patch.object(repo, "_connect", return_value=mock_conn):
        messages = repo.list_by_conversation("conv-uuid-1")

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert len(messages[1]["sources"]) == 1
    assert messages[1]["sources"][0]["source"] == "sample.pdf"

    # Simulate UI reconstruction into st.session_state['messages']
    reconstructed_ui_state = [
        {"role": m["role"], "content": m["content"], "sources": m["sources"]}
        for m in messages
    ]
    assert len(reconstructed_ui_state) == 2
    assert reconstructed_ui_state[1]["sources"][0]["page_number"] == 1
    print("✓ Test 7: Conversation retrieval and UI restoration verified.")


def test_mocked_cascade_deletion() -> None:
    """Verify document deletion executes cascade-safe query."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_cur.rowcount = 1

    repo = DocumentRepository("postgresql://user:pass@localhost:5432/testdb")
    with patch.object(repo, "_connect", return_value=mock_conn):
        success = repo.delete("doc-uuid-1")

    assert success
    mock_cur.execute.assert_called_once_with(
        "DELETE FROM documents WHERE id = %s;", ("doc-uuid-1",)
    )
    print("✓ Test 8: Document cascade deletion verified.")


# ==============================================================================
# Live Database Integration (Executed conditionally if PostgreSQL is running)
# ==============================================================================

def run_live_persistence_tests() -> None:
    """Run live database integration tests against isolated test namespace."""
    print("\n--- Running Live PostgreSQL Persistence Integration Tests ---")
    service = DatabaseService()
    service.initialize()

    test_content = b"PDF live persistence test content bytes"
    c_hash = compute_content_hash(test_content)
    test_file = "test_persistence_doc.pdf"

    # 1. Create document
    doc = service.documents.create_or_get_document(
        filename=test_file,
        content_hash=c_hash,
        file_size_bytes=len(test_content),
        storage_path=f"data/{test_file}",
        chunk_count=5,
    )
    doc_id = doc["id"]

    try:
        # 2. Test duplicate reuse
        doc_dup = service.documents.create_or_get_document(
            filename=test_file,
            content_hash=c_hash,
            file_size_bytes=len(test_content),
            storage_path=f"data/{test_file}",
            chunk_count=5,
        )
        assert doc_dup["id"] == doc_id, "Duplicate upload failed to reuse document ID"

        # 3. Create conversation
        conv = service.conversations.create_conversation(doc_id, title="Test Live Chat")
        conv_id = conv["id"]

        # 4. Create messages
        service.messages.create_message(conv_id, role="user", content="Hello document")
        service.messages.create_message(
            conv_id,
            role="assistant",
            content="Hello user, here is the answer.",
            sources=[{"source": test_file, "page_number": 1, "distance": 0.5}],
        )

        # 5. Retrieve messages
        retrieved = service.messages.list_by_conversation(conv_id)
        assert len(retrieved) == 2
        assert retrieved[0]["role"] == "user"
        assert retrieved[1]["role"] == "assistant"
        assert retrieved[1]["sources"][0]["page_number"] == 1

        print("✓ Live PostgreSQL persistence integration verified successfully.")
    finally:
        # 6. Clean up isolated test document (cascades to conversations & messages)
        service.documents.delete(doc_id)


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 10.4: POSTGRESQL APPLICATION STATE PERSISTENCE TESTS")
    print("=" * 70)

    test_persistence_disabled_by_default()
    test_content_hash_deterministic()
    test_mocked_schema_creation()
    test_mocked_document_creation_and_deduplication()
    test_mocked_conversation_creation()
    test_mocked_message_persistence_and_sources()
    test_mocked_conversation_restoration()
    test_mocked_cascade_deletion()

    print("\n--- Live PostgreSQL Availability Check ---")
    live_available = check_live_postgres_available()
    if live_available:
        print("Live PostgreSQL detected! Running live integration tests...")
        run_live_persistence_tests()
    else:
        print("LIVE DATABASE TESTING: NOT AVAILABLE in this local environment.")
        print("(Mocked & unit-level validations passed. Live database validation is pending.")
        print(" Per architectural instructions, live tests are not faked.)")

    print("\n" + "=" * 70)
    print("PERSISTENCE TESTS FINISHED.")
    print("=" * 70)
