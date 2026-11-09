from typing import Any, Dict, List, Optional, Union
from src.embedding_generator import EmbeddingGenerator
from src.vector_store import VectorStore
from src.vector_store_interface import BaseVectorStore


class Retriever:
    """
    Component for converting natural-language queries into vector embeddings
    and retrieving relevant document chunks from a BaseVectorStore.
    """

    def __init__(
        self,
        vector_store: Union[BaseVectorStore, VectorStore],
        embedding_generator: EmbeddingGenerator,
    ) -> None:
        """
        Initialize the Retriever with a BaseVectorStore and an EmbeddingGenerator.

        Args:
            vector_store (BaseVectorStore): Populated vector store containing document chunks.
            embedding_generator (EmbeddingGenerator): Embedding generator model instance.
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        max_distance: Optional[float] = None,
        source: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant document chunks for a natural-language query,
        with optional distance-based filtering and metadata filtering.

        FAISS measures similarity using L2 (Euclidean) distance, where a lower L2 distance
        indicates that the retrieved chunk embedding is closer and semantically more similar
        to the query vector.

        Filtering interactions:
            - Metadata filters (`source`, `document_id`) are passed directly to `VectorStore.search()`,
              which performs pre-top-k candidate filtering across the index before selecting
              the closest matching chunks in ascending distance order.
            - Distance filter (`max_distance`) provides an optional relevance boundary applied
              after metadata filtering. Any retrieved chunks exceeding this threshold are excluded.

        Args:
            query (str): User's natural-language query text.
            top_k (int): Number of most relevant documents to retrieve. Defaults to 3.
            max_distance (Optional[float]): Optional maximum L2 distance threshold for relevance.
                Chunks with distance > max_distance are filtered out. Defaults to None (no filtering).
            source (Optional[str]): Optional source filename filter. When provided, only chunks
                matching document["source"] are considered. Defaults to None.
            document_id (Optional[str]): Optional document identifier filter. When provided, only
                chunks matching document["document_id"] are considered. Defaults to None.

        Returns:
            List[Dict[str, Any]]: List of result dictionaries, each containing:
                - 'document': The stored document chunk dictionary.
                - 'distance': Semantic L2 distance score from the query embedding.

        Raises:
            ValueError: If the query is empty or whitespace-only.
            ValueError: If top_k is less than or equal to 0.
            ValueError: If max_distance is provided and less than 0.
            ValueError: If source or document_id validation fails in VectorStore.search().
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, got {top_k}.")

        if max_distance is not None and max_distance < 0:
            raise ValueError(
                f"max_distance must be greater than or equal to 0, got {max_distance}."
            )

        # Generate embedding for the query
        query_embeddings = self.embedding_generator.generate_embeddings([query])
        if len(query_embeddings) == 0:
            raise ValueError(f"Failed to generate embedding for query: '{query}'")

        query_embedding = query_embeddings[0]

        # Search the vector store with optional metadata filtering delegated to VectorStore
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
            source=source,
            document_id=document_id,
        )

        # Apply optional distance filtering on the metadata-filtered results
        if max_distance is not None:
            results = [res for res in results if res["distance"] <= max_distance]

        return results
