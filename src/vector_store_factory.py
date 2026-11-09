"""
Vector Store Factory.

Instantiates either FAISS VectorStore or PostgresVectorStore based on configuration.
Defaults to FAISS for local, zero-dependency development.
"""

import os
from typing import Optional

from src.postgres_vector_store import PostgresVectorStore
from src.vector_store import VectorStore
from src.vector_store_interface import BaseVectorStore

DEFAULT_VECTOR_STORE = "faiss"
DEFAULT_EMBEDDING_DIMENSION = 384


def get_vector_store_type() -> str:
    """
    Get the configured vector store backend from the environment.
    Defaults to 'faiss' for zero-dependency local development.
    """
    return os.getenv("VECTOR_STORE", DEFAULT_VECTOR_STORE).strip().lower()


def create_vector_store(
    store_type: Optional[str] = None,
    dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    **kwargs,
) -> BaseVectorStore:
    """
    Create a BaseVectorStore instance matching configuration or parameters.

    Args:
        store_type: Optional store identifier ('faiss' or 'postgres').
                    Defaults to VECTOR_STORE environment variable ('faiss').
        dimension: Embedding vector dimensionality. Defaults to 384.
        **kwargs: Additional backend-specific parameters (e.g. connection_url, table_name).

    Returns:
        BaseVectorStore implementation (VectorStore or PostgresVectorStore).

    Raises:
        ValueError: If an unsupported store type is requested.
    """
    selected = store_type.strip().lower() if store_type else get_vector_store_type()

    if selected == "faiss":
        return VectorStore(dimension=dimension)
    elif selected in ("postgres", "pgvector"):
        return PostgresVectorStore(dimension=dimension, **kwargs)
    else:
        raise ValueError(
            f"Unsupported vector store type: '{selected}'. "
            f"Supported stores are: 'faiss', 'postgres'."
        )
