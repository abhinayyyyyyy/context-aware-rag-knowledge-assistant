"""
Test Suite for Phase 10.5: Object Storage Abstraction and Containerization Preparation.

Validates:
1. Local storage upload, retrieval, existence checks, and deletion
2. Unique storage keys, filename sanitization, and path traversal prevention
3. Missing-object and error-handling edge cases
4. Storage provider factory selection (local vs. S3)
5. Mocked S3-compatible cloud storage operations (zero cloud credentials required)
6. Masking of secrets and endpoint URLs
7. Database persistence integration with storage_key and backward compatibility
"""

import io
import os
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

from src.database import (
    DocumentRepository,
    compute_content_hash,
    init_persistence_schema,
)
from src.local_storage import (
    LocalStorageProvider,
    generate_storage_key,
    sanitize_filename,
)
from src.s3_storage import S3StorageProvider
from src.storage_config import (
    get_local_storage_dir,
    get_storage_provider_name,
    mask_endpoint_url,
    mask_secret,
)
from src.storage_factory import create_storage_provider


# ==============================================================================
# 1. Local Storage Tests
# ==============================================================================

def test_filename_sanitization() -> None:
    """Verify filename sanitization strips traversal sequences and dangerous characters."""
    assert sanitize_filename("simple.pdf") == "simple.pdf"
    assert sanitize_filename("../../../secret.pdf") == "secret.pdf"
    assert sanitize_filename("/etc/passwd.pdf") == "passwd.pdf"
    assert sanitize_filename("my report (final)!.pdf") == "my_report__final.pdf"
    assert sanitize_filename("no_extension") == "no_extension.pdf"
    assert sanitize_filename("") == "unnamed_document.pdf"
    assert sanitize_filename(None) == "unnamed_document.pdf"
    print("✓ Test 1: Filename sanitization verified.")


def test_storage_key_generation() -> None:
    """Verify unique storage keys and optional prefixes."""
    key1 = generate_storage_key("report.pdf")
    key2 = generate_storage_key("report.pdf")
    assert key1 != key2, "Generated keys must be unique"
    assert key1.endswith("_report.pdf")

    prefixed_key = generate_storage_key("doc.pdf", prefix="tenant_1")
    assert prefixed_key.startswith("tenant_1/")
    assert prefixed_key.endswith("_doc.pdf")
    print("✓ Test 2: Storage key generation and uniqueness verified.")


def test_local_storage_crud_lifecycle() -> None:
    """Verify complete upload, existence, read, and delete cycle on local filesystem."""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = LocalStorageProvider(storage_dir=temp_dir)
        payload = b"%PDF-1.4 Mock PDF binary data for testing"

        # 1. Upload bytes
        storage_key = storage.upload(payload, filename="test_document.pdf")
        assert storage_key is not None
        assert storage_key.endswith("_test_document.pdf")

        # 2. Check existence
        assert storage.exists(storage_key) is True
        assert storage.exists("non_existent_key.pdf") is False

        # 3. Read content
        retrieved_bytes = storage.get(storage_key)
        assert retrieved_bytes == payload

        # 4. Delete object
        deleted = storage.delete(storage_key)
        assert deleted is True
        assert storage.exists(storage_key) is False

        # 5. Subsequent delete returns False
        assert storage.delete(storage_key) is False
    print("✓ Test 3: Local storage CRUD lifecycle verified.")


def test_local_storage_file_stream_upload() -> None:
    """Verify upload via file-like binary stream (BytesIO)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = LocalStorageProvider(storage_dir=temp_dir)
        data = b"Stream content of sample PDF"
        stream = io.BytesIO(data)

        storage_key = storage.upload(stream, filename="streamed.pdf")
        assert storage.exists(storage_key) is True
        assert storage.get(storage_key) == data
    print("✓ Test 4: Binary stream upload verified.")


def test_local_storage_path_traversal_prevention() -> None:
    """Verify strict rejection of path traversal sequences."""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = LocalStorageProvider(storage_dir=temp_dir)

        # Forbidden keys attempting directory escape
        dangerous_keys = [
            "../../etc/passwd",
            "../secret.txt",
            "/absolute/path/file.pdf",
            "subdir/../../escape.pdf",
            "null\x00byte.pdf",
        ]

        for bad_key in dangerous_keys:
            try:
                storage.get(bad_key)
                assert False, f"Expected ValueError for traversal key: {bad_key}"
            except ValueError:
                pass  # Successfully prevented

            try:
                storage._resolve_safe_path(bad_key)
                assert False, f"Expected ValueError for traversal key: {bad_key}"
            except ValueError:
                pass  # Successfully prevented
    print("✓ Test 5: Path traversal prevention verified.")


def test_local_storage_missing_and_empty_files() -> None:
    """Verify appropriate exceptions for missing objects and empty uploads."""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = LocalStorageProvider(storage_dir=temp_dir)

        # Missing object retrieval raises FileNotFoundError
        try:
            storage.get("missing_key_123.pdf")
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass

        # Empty data upload raises ValueError
        try:
            storage.upload(b"", filename="empty.pdf")
            assert False, "Expected ValueError on empty upload"
        except ValueError:
            pass
    print("✓ Test 6: Missing object and empty upload validation verified.")


# ==============================================================================
# 2. S3 Storage Provider Tests (Mocked - Zero Cloud Credentials Required)
# ==============================================================================

def test_s3_storage_provider_mocked_crud() -> None:
    """Verify S3 provider operations using a mocked boto3 client."""
    mock_client = MagicMock()
    mock_payload = b"%PDF-1.4 S3 mock content"

    # Configure mock get_object response
    mock_body = MagicMock()
    mock_body.read.return_value = mock_payload
    mock_client.get_object.return_value = {"Body": mock_body}
    mock_client.head_object.return_value = {"ContentLength": len(mock_payload)}

    s3_storage = S3StorageProvider(
        bucket_name="test-rag-bucket",
        region_name="us-east-1",
        prefix="documents",
        client=mock_client,
    )

    # 1. Upload
    storage_key = s3_storage.upload(mock_payload, filename="cloud_doc.pdf")
    assert storage_key.startswith("documents/")
    assert storage_key.endswith("_cloud_doc.pdf")
    mock_client.put_object.assert_called_once()
    call_kwargs = mock_client.put_object.call_args[1]
    assert call_kwargs["Bucket"] == "test-rag-bucket"
    assert call_kwargs["Key"] == storage_key
    assert call_kwargs["Body"] == mock_payload

    # 2. Exists
    assert s3_storage.exists(storage_key) is True
    mock_client.head_object.assert_called_with(Bucket="test-rag-bucket", Key=storage_key)

    # 3. Get
    retrieved = s3_storage.get(storage_key)
    assert retrieved == mock_payload
    mock_client.get_object.assert_called_with(Bucket="test-rag-bucket", Key=storage_key)

    # 4. Delete
    deleted = s3_storage.delete(storage_key)
    assert deleted is True
    mock_client.delete_object.assert_called_with(Bucket="test-rag-bucket", Key=storage_key)
    print("✓ Test 7: S3 storage provider CRUD operations verified with mock.")


def test_s3_storage_missing_object_translation() -> None:
    """Verify S3 NoSuchKey ClientError is translated into standard FileNotFoundError."""
    mock_client = MagicMock()
    from botocore.exceptions import ClientError

    client_error_404 = ClientError(
        error_response={"Error": {"Code": "NoSuchKey", "Message": "Key not found"}},
        operation_name="GetObject",
    )
    mock_client.get_object.side_effect = client_error_404
    mock_client.head_object.side_effect = client_error_404

    s3_storage = S3StorageProvider(
        bucket_name="test-rag-bucket",
        client=mock_client,
    )

    # exists() returns False on 404
    assert s3_storage.exists("missing.pdf") is False

    # get() raises FileNotFoundError
    try:
        s3_storage.get("missing.pdf")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass

    # delete() returns False because object does not exist
    assert s3_storage.delete("missing.pdf") is False
    print("✓ Test 8: S3 missing-object error translation verified.")


# ==============================================================================
# 3. Factory & Configuration Tests
# ==============================================================================

def test_storage_factory_selection() -> None:
    """Verify storage factory selects appropriate provider without cloud dependencies."""
    # Local selection
    with patch.dict(os.environ, {"STORAGE_PROVIDER": "local"}):
        provider = create_storage_provider()
        assert isinstance(provider, LocalStorageProvider)

    # S3 selection with mock config
    with patch.dict(os.environ, {
        "STORAGE_PROVIDER": "s3",
        "S3_BUCKET_NAME": "test-bucket",
    }):
        with patch("boto3.client"):
            provider = create_storage_provider()
            assert isinstance(provider, S3StorageProvider)

    # Missing bucket raises ValueError
    with patch.dict(os.environ, {
        "STORAGE_PROVIDER": "s3",
        "S3_BUCKET_NAME": "",
    }):
        try:
            create_storage_provider()
            assert False, "Expected ValueError when S3_BUCKET_NAME is empty"
        except ValueError:
            pass

    # Unsupported provider raises ValueError
    try:
        create_storage_provider("azure_blob")
        assert False, "Expected ValueError for unsupported provider"
    except ValueError:
        pass
    print("✓ Test 9: Storage provider factory selection verified.")


def test_credential_and_url_masking() -> None:
    """Verify secret credentials and endpoint URLs are securely masked."""
    assert mask_secret("AKIAIOSFODNN7EXAMPLE") == "AKIA****MPLE"
    assert mask_secret("short") == "****"
    assert mask_secret("") == ""

    assert mask_endpoint_url("https://user:secret123@minio.internal:9000/bucket") == "https://user:****@minio.internal:9000/bucket"
    assert mask_endpoint_url("https://s3.us-east-1.amazonaws.com") == "https://s3.us-east-1.amazonaws.com"
    print("✓ Test 10: Credential and endpoint masking verified.")


# ==============================================================================
# 4. Database Persistence Integration Tests
# ==============================================================================

def test_database_persistence_with_storage_key() -> None:
    """Verify DocumentRepository stores and retrieves storage_key with backward compatibility."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    repo = DocumentRepository("postgresql://user:pass@localhost:5432/testdb")

    # 1. New document insertion with storage_key
    mock_cur.fetchone.side_effect = [
        None,  # get_by_content_hash returns None
        ("2026-10-09 12:00:00",),  # RETURNING created_at
    ]

    with patch.object(repo, "_connect", return_value=mock_conn):
        doc = repo.create_or_get_document(
            filename="quarterly_report.pdf",
            content_hash="hash_abc_123",
            file_size_bytes=2048,
            storage_path="data/storage/uuid_quarterly_report.pdf",
            storage_key="uuid_quarterly_report.pdf",
            chunk_count=15,
        )

    assert doc["filename"] == "quarterly_report.pdf"
    assert doc["storage_key"] == "uuid_quarterly_report.pdf"
    assert doc["storage_path"] == "data/storage/uuid_quarterly_report.pdf"

    # Verify INSERT query included storage_key
    insert_call = mock_cur.execute.call_args_list[1]
    assert "storage_key" in insert_call[0][0]

    # 2. Backward compatibility: reading legacy record with 7 columns (no storage_key column)
    mock_cur.fetchone.side_effect = [
        ("doc-1", "legacy.pdf", "hash_leg", 1024, "legacy_path.pdf", 5, "2026-10-01 00:00:00"),
    ]

    with patch.object(repo, "_connect", return_value=mock_conn):
        legacy_doc = repo.get_by_content_hash("hash_leg")

    assert legacy_doc["storage_path"] == "legacy_path.pdf"
    assert legacy_doc["storage_key"] == "legacy_path.pdf", "Should fallback to storage_path for legacy records"

    # 3. Modern record with 8 columns (including storage_key)
    mock_cur.fetchone.side_effect = [
        ("doc-2", "modern.pdf", "hash_mod", 1024, "data/storage/key.pdf", 5, "2026-10-01 00:00:00", "key.pdf"),
    ]

    with patch.object(repo, "_connect", return_value=mock_conn):
        modern_doc = repo.get_by_content_hash("hash_mod")

    assert modern_doc["storage_key"] == "key.pdf"
    assert modern_doc["storage_path"] == "data/storage/key.pdf"
    print("✓ Test 11: Database persistence storage_key integration & backward compatibility verified.")


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 10.5: OBJECT STORAGE ABSTRACTION & CONTAINERIZATION TESTS")
    print("=" * 70)

    test_filename_sanitization()
    test_storage_key_generation()
    test_local_storage_crud_lifecycle()
    test_local_storage_file_stream_upload()
    test_local_storage_path_traversal_prevention()
    test_local_storage_missing_and_empty_files()
    test_s3_storage_provider_mocked_crud()
    test_s3_storage_missing_object_translation()
    test_storage_factory_selection()
    test_credential_and_url_masking()
    test_database_persistence_with_storage_key()

    print("\n" + "=" * 70)
    print("ALL PHASE 10.5 TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
