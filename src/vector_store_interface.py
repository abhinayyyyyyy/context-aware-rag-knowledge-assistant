"""
Vector Store Interface.

Defines the abstract base class for vector store implementations,
ensuring decoupling between retrieval logic and underlying storage backends
(e.g., in-memory FAISS, PostgreSQL + pgvector).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np


class BaseVectorStore(ABC):
    """
    Abstract interface for vector store backends.
    Defines the standard contract for indexing, searching, and deleting document vectors.
    """

    @abstractmethod
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Union[np.ndarray, List[List[float]]],
    ) -> None:
        """
        Add document chunks and corresponding numerical embeddings to the store.

        Args:
            documents: List of document chunk metadata dictionaries.
            embeddings: 2D array or list of numerical vector embeddings matching documents.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: Union[np.ndarray, List[float]],
        top_k: int = 3,
        source: Optional[str] = None,
        document_id: Optional[str] = None,
        max_distance: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the vector store for the top_k most similar document chunks.

        Args:
            query_embedding: Numerical vector embedding of the search query.
            top_k: Maximum number of nearest matching documents to return. Defaults to 3.
            source: Optional source filename filter.
            document_id: Optional document identifier filter.
            max_distance: Optional maximum distance threshold filter.

        Returns:
            List of result dictionaries, each containing 'document' and 'distance'.
        """
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """
        Delete all document chunks associated with a document identifier.

        Args:
            document_id: Unique identifier or source name of the document to delete.
        """
        pass
