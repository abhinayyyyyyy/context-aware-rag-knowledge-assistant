"""
PostgreSQL + pgvector Vector Store Implementation.

Implements BaseVectorStore using PostgreSQL and the pgvector extension for
persistent, production-grade vector storage and nearest-neighbor search.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from src.db_config import get_database_url, mask_database_url
from src.vector_store_interface import BaseVectorStore

DEFAULT_TABLE_NAME = "document_chunks"
DEFAULT_EMBEDDING_DIMENSION = 384


def initialize_schema(
    connection_url: Optional[str] = None,
    table_name: str = DEFAULT_TABLE_NAME,
) -> None:
    """
    Initialize the PostgreSQL schema with the pgvector extension and chunk table.
    Safe to execute repeatedly (idempotent).

    Args:
        connection_url: Optional PostgreSQL connection URL. Defaults to configuration.
        table_name: Target table name. Defaults to 'document_chunks'.

    Raises:
        RuntimeError: If connection fails or pgvector extension is not installed in PostgreSQL.
    """
    resolved_url = connection_url or get_database_url()
    masked_url = mask_database_url(resolved_url)

    try:
        with psycopg.connect(resolved_url) as conn:
            with conn.cursor() as cur:
                # 1. Enable pgvector extension
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                except psycopg.Error as exc:
                    raise RuntimeError(
                        f"Failed to enable 'vector' extension in PostgreSQL ({masked_url}). "
                        f"Ensure pgvector is installed on your PostgreSQL server: {exc}"
                    ) from exc

                # 2. Create document chunks table
                create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGSERIAL PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page_number INTEGER,
                    chunk_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding vector({DEFAULT_EMBEDDING_DIMENSION}) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
                """
                cur.execute(create_table_sql)

                # 3. Create metadata indexes
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table_name}_document_id ON {table_name} (document_id);"
                )
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table_name}_source ON {table_name} (source);"
                )
            conn.commit()
    except psycopg.OperationalError as exc:
        raise RuntimeError(
            f"PostgreSQL connection failed for '{masked_url}': {exc}"
        ) from exc


class PostgresVectorStore(BaseVectorStore):
    """
    PostgreSQL vector store backed by pgvector.
    Maintains complete interface and metadata compatibility with BaseVectorStore.
    """

    def __init__(
        self,
        connection_url: Optional[str] = None,
        table_name: str = DEFAULT_TABLE_NAME,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        auto_init_schema: bool = False,
    ) -> None:
        """
        Initialize the PostgresVectorStore.

        Args:
            connection_url: Optional connection URL. Defaults to get_database_url().
            table_name: Table name for vector storage. Defaults to 'document_chunks'.
            dimension: Vector dimension. Must be 384 for all-MiniLM-L6-v2.
            auto_init_schema: If True, executes initialize_schema() on initialization.

        Raises:
            ValueError: If dimension is invalid or connection URL cannot be resolved.
        """
        if dimension <= 0:
            raise ValueError(f"Dimension must be positive, got {dimension}.")

        self.table_name = table_name
        self.dimension = dimension
        self.connection_url = connection_url or get_database_url()

        if auto_init_schema:
            initialize_schema(self.connection_url, self.table_name)

    def _connect(self) -> psycopg.Connection:
        """Create an active connection and register pgvector types."""
        try:
            conn = psycopg.connect(self.connection_url)
            register_vector(conn)
            return conn
        except psycopg.Error as exc:
            masked_url = mask_database_url(self.connection_url)
            raise RuntimeError(
                f"Failed to connect to PostgreSQL ({masked_url}): {exc}"
            ) from exc

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        """
        Add documents and corresponding embeddings to the PostgreSQL store.

        Args:
            documents: List of document metadata dictionaries.
            embeddings: 2D numpy array of float32 embeddings (N, dimension).

        Raises:
            ValueError: If counts mismatch, embeddings are invalid, or dimension != self.dimension.
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Count mismatch: received {len(documents)} documents "
                f"and {len(embeddings)} embeddings."
            )

        if len(documents) == 0:
            return

        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embeddings shape (N, {self.dimension}), "
                f"got {embeddings.shape}."
            )

        rows = []
        for doc, emb in zip(documents, embeddings):
            rows.append((
                str(doc.get("document_id", "unknown")),
                str(doc.get("source", "unknown")),
                doc.get("page_number"),
                str(doc.get("chunk_id", "unknown")),
                str(doc.get("text", "")),
                emb.tolist(),
            ))

        insert_sql = f"""
        INSERT INTO {self.table_name} (
            document_id, source, page_number, chunk_id, text, embedding
        ) VALUES (%s, %s, %s, %s, %s, %s);
        """

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(insert_sql, rows)
            conn.commit()

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        max_distance: Optional[float] = None,
        source: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform nearest-neighbor similarity search using pgvector L2 distance.

        Note on Distance Semantics:
            FAISS IndexFlatL2 computes squared L2 distance: sum((u - v)^2).
            pgvector's '<->' operator computes Euclidean distance: sqrt(sum((u - v)^2)).
            To ensure 100% mathematical and threshold parity with FAISS and the
            calibrated 1.10 threshold, this query computes squared L2 distance via:
            (embedding <-> %s) * (embedding <-> %s).

        Args:
            query_embedding: Query vector (dimension,) or (1, dimension).
            top_k: Number of nearest neighbors to retrieve.
            max_distance: Optional maximum distance threshold for relevance gating.
            source: Optional source metadata filter.
            document_id: Optional document_id metadata filter.

        Returns:
            List of matching records with 'document' and 'distance'.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, got {top_k}.")

        if max_distance is not None and max_distance < 0:
            raise ValueError(
                f"max_distance must be greater than or equal to 0, got {max_distance}."
            )

        flat_query = np.asarray(query_embedding, dtype=np.float32).flatten()
        if len(flat_query) != self.dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.dimension}, got {len(flat_query)}."
            )

        query_vec = flat_query.tolist()

        # Build dynamic query conditions
        where_clauses: List[str] = []
        params: List[Any] = [query_vec, query_vec]  # for distance computation

        if source is not None:
            where_clauses.append("source = %s")
            params.append(source)

        if document_id is not None:
            where_clauses.append("document_id = %s")
            params.append(document_id)

        if max_distance is not None:
            # (embedding <-> query_vec)^2 <= max_distance
            where_clauses.append("((embedding <-> %s) * (embedding <-> %s)) <= %s")
            params.extend([query_vec, query_vec, float(max_distance)])

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        # Order by pgvector index-accelerated operator '<->'
        order_param = query_vec
        params.append(order_param)
        params.append(int(top_k))

        select_sql = f"""
        SELECT
            document_id,
            source,
            page_number,
            chunk_id,
            text,
            ((embedding <-> %s) * (embedding <-> %s)) AS distance
        FROM {self.table_name}
        {where_sql}
        ORDER BY (embedding <-> %s) ASC
        LIMIT %s;
        """

        results: List[Dict[str, Any]] = []
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(select_sql, params)
                for row in cur.fetchall():
                    results.append({
                        "document": {
                            "document_id": row[0],
                            "source": row[1],
                            "page_number": row[2],
                            "chunk_id": row[3],
                            "text": row[4],
                        },
                        "distance": float(row[5]),
                    })

        return results

    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks belonging to a document_id.

        Args:
            document_id: The document identifier to remove.

        Returns:
            Number of deleted chunk rows.
        """
        if not document_id or not document_id.strip():
            raise ValueError("document_id must be a non-empty string.")

        delete_sql = f"DELETE FROM {self.table_name} WHERE document_id = %s;"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(delete_sql, (document_id.strip(),))
                deleted_count = cur.rowcount
            conn.commit()

        return deleted_count

    def __repr__(self) -> str:
        return (
            f"PostgresVectorStore(table='{self.table_name}', dimension={self.dimension}, "
            f"url='{mask_database_url(self.connection_url)}')"
        )
