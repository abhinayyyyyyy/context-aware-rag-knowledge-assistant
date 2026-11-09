"""
PostgreSQL Application State Persistence Module.

Manages durable relational persistence for:
1. Document metadata (content hash deduplication, chunk count, file size)
2. Conversations (linked to documents)
3. Messages (ordered turns with JSONB source citations)

Maintains strict separation between database access, RAG logic, and UI.
"""

import hashlib
import json
import os
from typing import Any, Dict, List, Optional
import uuid

import psycopg
from psycopg.types.json import Jsonb

from src.db_config import get_database_url, is_persistence_enabled, mask_database_url


def compute_content_hash(data: bytes) -> str:
    """Compute SHA-256 hex digest of file binary bytes."""
    return hashlib.sha256(data).hexdigest()


def init_persistence_schema(connection_url: Optional[str] = None) -> None:
    """
    Initialize relational tables for persistent application state.
    Safe to execute repeatedly (idempotent).

    Args:
        connection_url: Optional database connection URL. Defaults to get_database_url().

    Raises:
        RuntimeError: If connecting or executing schema DDL fails.
    """
    resolved_url = connection_url or get_database_url()
    masked_url = mask_database_url(resolved_url)

    schema_sql = """
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        content_hash TEXT NOT NULL UNIQUE,
        file_size_bytes BIGINT NOT NULL,
        storage_path TEXT NOT NULL,
        chunk_count INTEGER NOT NULL,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        storage_key TEXT
    );

    ALTER TABLE documents ADD COLUMN IF NOT EXISTS storage_key TEXT;

    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        title TEXT NOT NULL,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        sources JSONB,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents (content_hash);
    CREATE INDEX IF NOT EXISTS idx_conversations_document_id ON conversations (document_id);
    CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages (conversation_id);
    """

    try:
        with psycopg.connect(resolved_url) as conn:
            with conn.cursor() as cur:
                cur.execute(schema_sql)
            conn.commit()
    except psycopg.Error as exc:
        raise RuntimeError(
            f"Failed to initialize persistence schema on PostgreSQL ({masked_url}): {exc}"
        ) from exc


class DocumentRepository:
    """Data access repository for document metadata."""

    def __init__(self, connection_url: Optional[str] = None) -> None:
        self.connection_url = connection_url or get_database_url()

    def _connect(self) -> psycopg.Connection:
        try:
            return psycopg.connect(self.connection_url)
        except psycopg.Error as exc:
            raise RuntimeError(
                f"PostgreSQL connection failed ({mask_database_url(self.connection_url)}): {exc}"
            ) from exc

    def get_by_content_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Look up document record by SHA-256 content hash."""
        sql = """
        SELECT id, filename, content_hash, file_size_bytes, storage_path, chunk_count, created_at, storage_key
        FROM documents
        WHERE content_hash = %s;
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (content_hash.strip(),))
                row = cur.fetchone()
                if not row:
                    return None
                raw_storage_key = row[7] if len(row) > 7 and row[7] is not None else row[4]
                return {
                    "id": row[0],
                    "filename": row[1],
                    "content_hash": row[2],
                    "file_size_bytes": row[3],
                    "storage_path": row[4],
                    "chunk_count": row[5],
                    "created_at": row[6],
                    "storage_key": raw_storage_key,
                }

    def get_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Look up document record by primary ID."""
        sql = """
        SELECT id, filename, content_hash, file_size_bytes, storage_path, chunk_count, created_at, storage_key
        FROM documents
        WHERE id = %s;
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id.strip(),))
                row = cur.fetchone()
                if not row:
                    return None
                raw_storage_key = row[7] if len(row) > 7 and row[7] is not None else row[4]
                return {
                    "id": row[0],
                    "filename": row[1],
                    "content_hash": row[2],
                    "file_size_bytes": row[3],
                    "storage_path": row[4],
                    "chunk_count": row[5],
                    "created_at": row[6],
                    "storage_key": raw_storage_key,
                }

    def create_or_get_document(
        self,
        filename: str,
        content_hash: str,
        file_size_bytes: int,
        storage_path: Optional[str] = None,
        chunk_count: int = 0,
        storage_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new document metadata record, or return the existing one if
        content_hash matches an uploaded file (deduplication).
        """
        resolved_key = storage_key or storage_path or filename
        resolved_path = storage_path or storage_key or filename

        existing = self.get_by_content_hash(content_hash)
        if existing is not None:
            # Update storage path, storage key, and chunk count if needed, then return existing
            update_sql = """
            UPDATE documents
            SET storage_path = %s, storage_key = %s, chunk_count = %s
            WHERE id = %s;
            """
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(update_sql, (resolved_path, resolved_key, chunk_count, existing["id"]))
                conn.commit()
            existing["storage_path"] = resolved_path
            existing["storage_key"] = resolved_key
            existing["chunk_count"] = chunk_count
            return existing

        doc_id = str(uuid.uuid4())
        insert_sql = """
        INSERT INTO documents (
            id, filename, content_hash, file_size_bytes, storage_path, chunk_count, storage_key
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING created_at;
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(insert_sql, (
                    doc_id, filename, content_hash, file_size_bytes, resolved_path, chunk_count, resolved_key
                ))
                created_at = cur.fetchone()[0]
            conn.commit()

        return {
            "id": doc_id,
            "filename": filename,
            "content_hash": content_hash,
            "file_size_bytes": file_size_bytes,
            "storage_path": resolved_path,
            "storage_key": resolved_key,
            "chunk_count": chunk_count,
            "created_at": created_at,
        }

    def list_all(self) -> List[Dict[str, Any]]:
        """List all indexed documents ordered by upload date descending."""
        sql = """
        SELECT id, filename, content_hash, file_size_bytes, storage_path, chunk_count, created_at, storage_key
        FROM documents
        ORDER BY created_at DESC;
        """
        results = []
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                for row in cur.fetchall():
                    raw_storage_key = row[7] if len(row) > 7 and row[7] is not None else row[4]
                    results.append({
                        "id": row[0],
                        "filename": row[1],
                        "content_hash": row[2],
                        "file_size_bytes": row[3],
                        "storage_path": row[4],
                        "storage_key": raw_storage_key,
                        "chunk_count": row[5],
                        "created_at": row[6],
                    })
        return results

    def delete(self, document_id: str) -> bool:
        """Delete document record by ID (cascades to conversations & messages)."""
        sql = "DELETE FROM documents WHERE id = %s;"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id,))
                count = cur.rowcount
            conn.commit()
        return count > 0


class ConversationRepository:
    """Data access repository for conversation sessions."""

    def __init__(self, connection_url: Optional[str] = None) -> None:
        self.connection_url = connection_url or get_database_url()

    def _connect(self) -> psycopg.Connection:
        try:
            return psycopg.connect(self.connection_url)
        except psycopg.Error as exc:
            raise RuntimeError(
                f"PostgreSQL connection failed ({mask_database_url(self.connection_url)}): {exc}"
            ) from exc

    def create_conversation(
        self,
        document_id: str,
        title: str = "New Conversation",
    ) -> Dict[str, Any]:
        """Create a new conversation session linked to a document."""
        conv_id = str(uuid.uuid4())
        sql = """
        INSERT INTO conversations (id, document_id, title)
        VALUES (%s, %s, %s)
        RETURNING created_at, updated_at;
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (conv_id, document_id, title))
                created_at, updated_at = cur.fetchone()
            conn.commit()

        return {
            "id": conv_id,
            "document_id": document_id,
            "title": title,
            "created_at": created_at,
            "updated_at": updated_at,
        }

    def get_by_id(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve conversation record by ID."""
        sql = """
        SELECT id, document_id, title, created_at, updated_at
        FROM conversations
        WHERE id = %s;
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (conversation_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "id": row[0],
                    "document_id": row[1],
                    "title": row[2],
                    "created_at": row[3],
                    "updated_at": row[4],
                }

    def list_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """List all conversations for a document, most recent first."""
        sql = """
        SELECT id, document_id, title, created_at, updated_at
        FROM conversations
        WHERE document_id = %s
        ORDER BY updated_at DESC;
        """
        results = []
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id,))
                for row in cur.fetchall():
                    results.append({
                        "id": row[0],
                        "document_id": row[1],
                        "title": row[2],
                        "created_at": row[3],
                        "updated_at": row[4],
                    })
        return results

    def touch_updated_at(self, conversation_id: str) -> None:
        """Update the conversation's updated_at timestamp."""
        sql = "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = %s;"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (conversation_id,))
            conn.commit()

    def delete(self, conversation_id: str) -> bool:
        """Delete conversation by ID (cascades to messages)."""
        sql = "DELETE FROM conversations WHERE id = %s;"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (conversation_id,))
                count = cur.rowcount
            conn.commit()
        return count > 0


class MessageRepository:
    """Data access repository for persistent chat messages."""

    def __init__(self, connection_url: Optional[str] = None) -> None:
        self.connection_url = connection_url or get_database_url()

    def _connect(self) -> psycopg.Connection:
        try:
            return psycopg.connect(self.connection_url)
        except psycopg.Error as exc:
            raise RuntimeError(
                f"PostgreSQL connection failed ({mask_database_url(self.connection_url)}): {exc}"
            ) from exc

    def create_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Append a message to a conversation.

        Args:
            conversation_id: Target conversation UUID.
            role: 'user' or 'assistant'.
            content: Message body string.
            sources: Optional source citation dictionaries (for assistant messages).
        """
        msg_id = str(uuid.uuid4())
        sources_payload = Jsonb(sources or [])

        sql = """
        INSERT INTO messages (id, conversation_id, role, content, sources)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING created_at;
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (msg_id, conversation_id, role, content, sources_payload))
                created_at = cur.fetchone()[0]
                # Touch parent conversation updated_at
                cur.execute(
                    "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = %s;",
                    (conversation_id,),
                )
            conn.commit()

        return {
            "id": msg_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "sources": sources or [],
            "created_at": created_at,
        }

    def list_by_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Retrieve all messages for a conversation in chronological order."""
        sql = """
        SELECT id, conversation_id, role, content, sources, created_at
        FROM messages
        WHERE conversation_id = %s
        ORDER BY created_at ASC;
        """
        results = []
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (conversation_id,))
                for row in cur.fetchall():
                    raw_sources = row[4]
                    parsed_sources = raw_sources if isinstance(raw_sources, list) else []
                    results.append({
                        "id": row[0],
                        "conversation_id": row[1],
                        "role": row[2],
                        "content": row[3],
                        "sources": parsed_sources,
                        "created_at": row[5],
                    })
        return results


class DatabaseService:
    """Facade service consolidating document, conversation, and message operations."""

    def __init__(self, connection_url: Optional[str] = None) -> None:
        self.connection_url = connection_url or get_database_url()
        self.documents = DocumentRepository(self.connection_url)
        self.conversations = ConversationRepository(self.connection_url)
        self.messages = MessageRepository(self.connection_url)

    def initialize(self) -> None:
        """Initialize all relational persistence tables."""
        init_persistence_schema(self.connection_url)


_db_service_instance: Optional[DatabaseService] = None


def get_db_service() -> DatabaseService:
    """Return singleton DatabaseService instance."""
    global _db_service_instance
    if _db_service_instance is None:
        _db_service_instance = DatabaseService()
    return _db_service_instance
