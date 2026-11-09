import os
import pickle
from typing import Any, Dict, List, Optional, Union
import faiss
import numpy as np

from src.vector_store_interface import BaseVectorStore


class VectorStore(BaseVectorStore):
    """
    FAISS-based in-memory vector store for indexing document chunk embeddings
    and performing semantic similarity searches, with persistence support.
    """

    def __init__(self, dimension: int = 384) -> None:
        """
        Initialize the VectorStore with a FAISS IndexFlatL2 index.

        Args:
            dimension (int): Dimensionality of the embedding vectors. Defaults to 384.
        """
        self.dimension = dimension
        # Initialize flat L2 index
        self.index = faiss.IndexFlatL2(dimension)
        # Store corresponding document chunks in matching order
        self.documents: List[Dict[str, Any]] = []

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Union[np.ndarray, List[List[float]]],
    ) -> None:
        """
        Convert embeddings to float32, validate shapes and dimensions,
        add them to the FAISS index, and store documents in the same order.

        Args:
            documents (List[Dict[str, Any]]): List of document chunk dictionaries.
            embeddings (Union[np.ndarray, List[List[float]]]): Embeddings matching the documents.

        Raises:
            ValueError: If embeddings is not 2-dimensional.
            ValueError: If the number of embeddings does not equal the number of documents.
            ValueError: If the embedding dimension does not match self.dimension.
        """
        # Convert embeddings to float32 NumPy array
        embeddings_array = np.asarray(embeddings, dtype=np.float32)

        # Validate that embeddings is a 2-dimensional array
        if embeddings_array.ndim != 2:
            raise ValueError(
                f"Embeddings must be a 2-dimensional array, got shape {embeddings_array.shape}."
            )

        # Validate that the number of embeddings matches the number of documents
        if len(documents) != embeddings_array.shape[0]:
            raise ValueError(
                f"Number of documents ({len(documents)}) does not match "
                f"number of embeddings ({embeddings_array.shape[0]})."
            )

        # Validate that embedding dimension matches index dimension
        if embeddings_array.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension ({embeddings_array.shape[1]}) does not match "
                f"index dimension ({self.dimension})."
            )

        if len(documents) == 0:
            return

        # Add vectors to the FAISS index
        self.index.add(embeddings_array)

        # Retain documents in matching order
        self.documents.extend(documents)

    def search(
        self,
        query_embedding: Union[np.ndarray, List[float]],
        top_k: int = 3,
        source: Optional[str] = None,
        document_id: Optional[str] = None,
        max_distance: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the FAISS index for top-k most similar documents to the query embedding,
        with optional metadata filtering by source filename and/or document_id, and
        optional max_distance filtering.

        When metadata filters are supplied, the full index is searched and candidates
        are filtered by document metadata before selecting the top-k results, preserving
        FAISS distance ordering. If no metadata filters are provided, search queries
        only min(top_k, ntotal) candidates for maximum efficiency.

        Args:
            query_embedding (Union[np.ndarray, List[float]]): Query vector embedding.
            top_k (int): Maximum number of nearest matching documents to return. Defaults to 3.
            source (Optional[str]): Optional source filename filter. When provided, only chunks
                with matching document["source"] are returned.
            document_id (Optional[str]): Optional document identifier filter. When provided, only
                chunks with matching document["document_id"] are returned.
            max_distance (Optional[float]): Optional maximum distance threshold filter.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing 'document' and 'distance'.

        Raises:
            ValueError: If top_k is not greater than 0.
            ValueError: If source is provided but is empty or not a valid string.
            ValueError: If document_id is provided but is empty or not a valid string.
            ValueError: If query_embedding dimension does not match self.dimension.
        """
        # Validate that top_k is greater than zero
        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, got {top_k}.")

        # Validate optional metadata filters
        if source is not None and (not isinstance(source, str) or not source.strip()):
            raise ValueError("source must be a non-empty string when provided.")

        if document_id is not None and (not isinstance(document_id, str) or not document_id.strip()):
            raise ValueError("document_id must be a non-empty string when provided.")

        # Convert query embedding to float32 NumPy array
        query_vec = np.asarray(query_embedding, dtype=np.float32)

        # Validate query embedding dimension and reshape to 2D
        if query_vec.ndim == 1:
            if query_vec.shape[0] != self.dimension:
                raise ValueError(
                    f"Query embedding dimension ({query_vec.shape[0]}) does not match "
                    f"index dimension ({self.dimension})."
                )
            query_vec = query_vec.reshape(1, -1)
        elif query_vec.ndim == 2 and query_vec.shape[0] == 1:
            if query_vec.shape[1] != self.dimension:
                raise ValueError(
                    f"Query embedding dimension ({query_vec.shape[1]}) does not match "
                    f"index dimension ({self.dimension})."
                )
        else:
            raise ValueError(
                f"Query embedding must be 1D or 2D with shape (1, {self.dimension}), "
                f"got shape {query_vec.shape}."
            )

        # Return empty list if the index or document store is empty
        if self.index.ntotal == 0 or len(self.documents) == 0:
            return []

        # When filtering by metadata or distance, search all candidates to find matching top_k
        has_filter = (source is not None) or (document_id is not None) or (max_distance is not None)
        num_candidates = self.index.ntotal if has_filter else min(top_k, self.index.ntotal)

        distances, indices = self.index.search(query_vec, num_candidates)

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.documents):
                doc = self.documents[idx]

                # Apply metadata filters
                if source is not None and doc.get("source") != source:
                    continue

                if document_id is not None and doc.get("document_id") != document_id:
                    continue

                # Apply optional distance filter
                if max_distance is not None and float(dist) > max_distance:
                    continue

                results.append({
                    "document": doc,
                    "distance": float(dist),
                })

                if len(results) == top_k:
                    break

        return results

    def delete_document(self, document_id: str) -> None:
        """
        Delete all document chunks associated with a document identifier.
        Rebuilds the in-memory FAISS index with the remaining documents.

        Args:
            document_id: Unique identifier or source name of the document to delete.
        """
        if not document_id:
            return

        indices_to_remove = [
            i for i, doc in enumerate(self.documents)
            if doc.get("document_id") == document_id or doc.get("source") == document_id
        ]
        if not indices_to_remove:
            return

        if len(indices_to_remove) == len(self.documents):
            self.index.reset()
            self.documents.clear()
            return

        try:
            keep_indices = [i for i in range(len(self.documents)) if i not in indices_to_remove]
            if keep_indices:
                kept_vectors = np.array([self.index.reconstruct(i) for i in keep_indices], dtype=np.float32)
                self.index.reset()
                self.index.add(kept_vectors)
                self.documents = [self.documents[i] for i in keep_indices]
            else:
                self.index.reset()
                self.documents.clear()
        except Exception:
            self.documents = [doc for i, doc in enumerate(self.documents) if i not in indices_to_remove]

    def save(self, directory: str) -> None:
        """
        Save the FAISS index and document metadata to disk.

        Args:
            directory (str): Destination directory where index.faiss and documents.pkl will be stored.
        """
        os.makedirs(directory, exist_ok=True)
        index_path = os.path.join(directory, "index.faiss")
        documents_path = os.path.join(directory, "documents.pkl")

        faiss.write_index(self.index, index_path)
        with open(documents_path, "wb") as f:
            pickle.dump(self.documents, f)

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        """
        Load a saved VectorStore from a directory.

        Security Note:
            This method uses Python's standard-library `pickle` module to deserialize
            document metadata. Because `pickle` can execute arbitrary code when loading
            untrusted data, only load vector-store files from trusted sources created
            by this project.

        Args:
            directory (str): Directory containing 'index.faiss' and 'documents.pkl'.

        Returns:
            VectorStore: Restored VectorStore instance.

        Raises:
            FileNotFoundError: If index.faiss or documents.pkl is missing.
            ValueError: If index dimension is invalid, documents metadata is not a list,
                        or number of FAISS vectors does not match the number of stored documents.
        """
        index_path = os.path.join(directory, "index.faiss")
        documents_path = os.path.join(directory, "documents.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")
        if not os.path.exists(documents_path):
            raise FileNotFoundError(f"Document metadata file not found: {documents_path}")

        index = faiss.read_index(index_path)

        with open(documents_path, "rb") as f:
            documents = pickle.load(f)

        if not isinstance(documents, list):
            raise ValueError(
                f"Loaded document metadata must be a list, got {type(documents).__name__}."
            )

        dimension = getattr(index, "d", None)
        if dimension is None or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"Loaded FAISS index has an invalid dimension: {dimension}.")

        store = cls(dimension=dimension)
        store.index = index
        store.documents = documents

        if store.index.ntotal != len(store.documents):
            raise ValueError(
                f"Vector count mismatch: FAISS index has {store.index.ntotal} vectors, "
                f"but documents metadata has {len(store.documents)} records."
            )

        return store
