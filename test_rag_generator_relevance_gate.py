"""
Unit and integration tests for the RAGGenerator relevance-gating layer.

All tests use mocks/stubs to ensure:
- No network requests occur
- No real Ollama generation occurs
- No SentenceTransformer model is loaded
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from src.rag_generator import RAGGenerator
from src.retriever import Retriever


class MockRetriever:
    """
    Mock retriever providing deterministic search results
    without requiring models or vector databases.
    """

    def __init__(self, canned_results: Optional[List[Dict[str, Any]]] = None) -> None:
        self.canned_results = canned_results if canned_results is not None else []
        self.last_retrieve_kwargs: Dict[str, Any] = {}

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        max_distance: Optional[float] = None,
        source: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        self.last_retrieve_kwargs = {
            "query": query,
            "top_k": top_k,
            "max_distance": max_distance,
            "source": source,
            "document_id": document_id,
        }

        # Filter canned results deterministically based on retriever parameters
        results = list(self.canned_results)

        if source is not None:
            results = [
                r for r in results if r.get("document", {}).get("source") == source
            ]

        if document_id is not None:
            results = [
                r for r in results if r.get("document", {}).get("document_id") == document_id
            ]

        if max_distance is not None:
            results = [r for r in results if r.get("distance", 0.0) <= max_distance]

        return results[:top_k]


def create_sample_chunk(text: str, source: str, page: int, distance: float, doc_id: str = "doc_1") -> Dict[str, Any]:
    return {
        "document": {
            "text": text,
            "source": source,
            "page_number": page,
            "document_id": doc_id,
        },
        "distance": distance,
    }


def test_all_chunks_fail_max_distance() -> None:
    """A. When all retrieved chunks exceed max_distance:
    - Ollama is NOT called
    - sources is empty
    - standard unavailable message is returned
    """
    canned = [
        create_sample_chunk("Irrelevant chunk 1", "doc1.pdf", 1, 1.8),
        create_sample_chunk("Irrelevant chunk 2", "doc1.pdf", 2, 2.2),
    ]
    retriever = MockRetriever(canned_results=canned)
    rag_gen = RAGGenerator(retriever=retriever)

    with patch("ollama.generate") as mock_ollama:
        result = rag_gen.generate_answer(
            query="What is quantum gravity?",
            top_k=3,
            max_distance=1.0,
        )

        mock_ollama.assert_not_called()
        assert result["answer"] == "I could not find relevant information in the provided documents."
        assert result["sources"] == []


def test_some_chunks_pass_and_some_fail() -> None:
    """B. When some chunks pass and some fail:
    - only surviving chunks reach prompt construction & Ollama
    - only surviving chunks appear in sources
    """
    canned = [
        create_sample_chunk("Relevant text alpha", "doc1.pdf", 1, 0.4),
        create_sample_chunk("Weak text beta", "doc1.pdf", 2, 1.5),
        create_sample_chunk("Relevant text gamma", "doc1.pdf", 3, 0.7),
    ]
    retriever = MockRetriever(canned_results=canned)
    rag_gen = RAGGenerator(retriever=retriever)

    with patch("ollama.generate", return_value={"response": "Simulated LLM answer"}) as mock_ollama:
        result = rag_gen.generate_answer(
            query="Explain alpha and gamma",
            top_k=3,
            max_distance=1.0,
        )

        mock_ollama.assert_called_once()
        prompt_used = mock_ollama.call_args[1]["prompt"]

        # Only 0.4 and 0.7 chunks should be in the prompt
        assert "Relevant text alpha" in prompt_used
        assert "Relevant text gamma" in prompt_used
        assert "Weak text beta" not in prompt_used

        # Sources should only contain the 2 surviving chunks
        assert len(result["sources"]) == 2
        assert result["sources"][0]["distance"] == 0.4
        assert result["sources"][0]["page_number"] == 1
        assert result["sources"][1]["distance"] == 0.7
        assert result["sources"][1]["page_number"] == 3
        assert result["answer"] == "Simulated LLM answer"


def test_exact_distance_boundary() -> None:
    """C. Exact distance boundary:
    distance == max_distance must survive (<= condition).
    """
    canned = [
        create_sample_chunk("Exact boundary chunk", "doc1.pdf", 4, 1.0),
        create_sample_chunk("Slightly exceeding chunk", "doc1.pdf", 5, 1.0001),
    ]
    retriever = MockRetriever(canned_results=canned)
    rag_gen = RAGGenerator(retriever=retriever)

    with patch("ollama.generate", return_value={"response": "Boundary answer"}) as mock_ollama:
        result = rag_gen.generate_answer(
            query="Boundary query",
            top_k=2,
            max_distance=1.0,
        )

        mock_ollama.assert_called_once()
        assert len(result["sources"]) == 1
        assert result["sources"][0]["distance"] == 1.0
        assert result["sources"][0]["page_number"] == 4
        assert "Exact boundary chunk" in mock_ollama.call_args[1]["prompt"]
        assert "Slightly exceeding chunk" not in mock_ollama.call_args[1]["prompt"]


def test_empty_vector_store() -> None:
    """D. Empty vector store:
    - Ollama is NOT called
    - clean unavailable response is returned
    """
    retriever = MockRetriever(canned_results=[])
    rag_gen = RAGGenerator(retriever=retriever)

    with patch("ollama.generate") as mock_ollama:
        result = rag_gen.generate_answer(
            query="Any question",
            top_k=3,
        )

        mock_ollama.assert_not_called()
        assert result["answer"] == "I could not find relevant information in the provided documents."
        assert result["sources"] == []


def test_default_max_distance_applied() -> None:
    """E. default_max_distance:
    Instance default is used when query-level threshold is None.
    """
    canned = [
        create_sample_chunk("Chunk 1", "doc1.pdf", 1, 0.6),
        create_sample_chunk("Chunk 2", "doc1.pdf", 2, 1.2),
    ]
    retriever = MockRetriever(canned_results=canned)
    rag_gen = RAGGenerator(retriever=retriever, default_max_distance=0.8)

    with patch("ollama.generate", return_value={"response": "Default threshold answer"}) as mock_ollama:
        result = rag_gen.generate_answer(query="Test query", top_k=3)

        assert retriever.last_retrieve_kwargs["max_distance"] == 0.8
        mock_ollama.assert_called_once()
        assert len(result["sources"]) == 1
        assert result["sources"][0]["distance"] == 0.6


def test_query_level_threshold_override() -> None:
    """F. Query-level threshold override:
    Query-level max_distance overrides instance default_max_distance.
    """
    canned = [
        create_sample_chunk("Chunk 1", "doc1.pdf", 1, 0.4),
        create_sample_chunk("Chunk 2", "doc1.pdf", 2, 0.9),
    ]
    retriever = MockRetriever(canned_results=canned)
    # Instance default allows 0.9 (since 1.5 > 0.9)
    rag_gen = RAGGenerator(retriever=retriever, default_max_distance=1.5)

    with patch("ollama.generate", return_value={"response": "Overridden answer"}) as mock_ollama:
        # Override to 0.5; should filter out Chunk 2 (0.9)
        result = rag_gen.generate_answer(
            query="Test query",
            top_k=3,
            max_distance=0.5,
        )

        assert retriever.last_retrieve_kwargs["max_distance"] == 0.5
        mock_ollama.assert_called_once()
        assert len(result["sources"]) == 1
        assert result["sources"][0]["distance"] == 0.4


def test_invalid_negative_max_distance() -> None:
    """G. Invalid negative max_distance raises ValueError."""
    retriever = MockRetriever()
    rag_gen = RAGGenerator(retriever=retriever)

    try:
        rag_gen.generate_answer("Sample query", max_distance=-0.1)
        raise AssertionError("Expected ValueError was not raised for negative max_distance.")
    except ValueError as err:
        assert "max_distance must be greater than or equal to 0" in str(err)


def test_invalid_negative_default_max_distance() -> None:
    """H. Invalid negative default_max_distance raises ValueError."""
    retriever = MockRetriever()

    try:
        RAGGenerator(retriever=retriever, default_max_distance=-0.01)
        raise AssertionError("Expected ValueError was not raised for negative default_max_distance.")
    except ValueError as err:
        assert "default_max_distance must be greater than or equal to 0" in str(err)


def test_metadata_and_distance_filtering() -> None:
    """I. Metadata + distance filtering:
    source, document_id, and max_distance are forwarded correctly.
    """
    canned = [
        create_sample_chunk("Doc A chunk", "doc_a.pdf", 1, 0.3, doc_id="id_a"),
        create_sample_chunk("Doc B chunk", "doc_b.pdf", 1, 0.3, doc_id="id_b"),
        create_sample_chunk("Doc A distant chunk", "doc_a.pdf", 2, 1.4, doc_id="id_a"),
    ]
    retriever = MockRetriever(canned_results=canned)
    rag_gen = RAGGenerator(retriever=retriever)

    with patch("ollama.generate", return_value={"response": "Filtered answer"}) as mock_ollama:
        result = rag_gen.generate_answer(
            query="Find specific info",
            top_k=3,
            max_distance=0.5,
            source="doc_a.pdf",
            document_id="id_a",
        )

        assert retriever.last_retrieve_kwargs["source"] == "doc_a.pdf"
        assert retriever.last_retrieve_kwargs["document_id"] == "id_a"
        assert retriever.last_retrieve_kwargs["max_distance"] == 0.5
        mock_ollama.assert_called_once()
        assert len(result["sources"]) == 1
        assert result["sources"][0]["source"] == "doc_a.pdf"
        assert result["sources"][0]["distance"] == 0.3


def test_backward_compatibility() -> None:
    """J. Backward compatibility:
    generate_answer(query, top_k=3) works without any distance parameters.
    """
    canned = [
        create_sample_chunk("Standard chunk 1", "doc1.pdf", 1, 0.5),
        create_sample_chunk("Standard chunk 2", "doc1.pdf", 2, 1.2),
    ]
    retriever = MockRetriever(canned_results=canned)
    rag_gen = RAGGenerator(retriever=retriever)

    with patch("ollama.generate", return_value={"response": "Legacy compatible answer"}) as mock_ollama:
        result = rag_gen.generate_answer("Legacy call", top_k=3)

        assert retriever.last_retrieve_kwargs["max_distance"] is None
        mock_ollama.assert_called_once()
        assert result["answer"] == "Legacy compatible answer"
        assert len(result["sources"]) == 2


def test_instantiation_with_and_without_reranker() -> None:
    """K. RAGGenerator constructor accepts reranker as optional parameter:
    - Without reranker argument: reranker defaults to None
    - With reranker=None: reranker is None
    - With reranker=mock_reranker: reranker is stored as attribute
    """
    retriever = MockRetriever()

    # 1. Without reranker argument (default)
    gen_default = RAGGenerator(retriever=retriever)
    assert gen_default.reranker is None

    # 2. Explicit reranker=None
    gen_none = RAGGenerator(retriever=retriever, reranker=None)
    assert gen_none.reranker is None

    # 3. Explicit reranker object
    mock_reranker = MagicMock()
    gen_reranker = RAGGenerator(retriever=retriever, reranker=mock_reranker)
    assert gen_reranker.reranker is mock_reranker


def test_reranker_reordering_execution() -> None:
    """L. Verify that RAGGenerator with reranker reorders candidates according to predict scores."""
    canned = [
        create_sample_chunk("Lower scored chunk", "doc1.pdf", 1, 0.4),
        create_sample_chunk("Higher scored chunk", "doc1.pdf", 2, 0.8),
    ]
    retriever = MockRetriever(canned_results=canned)

    # Mock reranker returns [1.0, 5.0], so second chunk should be reranked to Rank 1
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [1.0, 5.0]

    rag_gen = RAGGenerator(retriever=retriever, reranker=mock_reranker)

    with patch("ollama.generate", return_value={"response": "Grounded answer"}) as mock_ollama:
        result = rag_gen.generate_answer("Test query", top_k=2)

        mock_reranker.predict.assert_called_once()
        assert result["sources"][0]["source"] == "doc1.pdf"
        assert result["sources"][0]["page_number"] == 2  # Higher scored chunk elevated to Rank 1
        assert result["sources"][1]["page_number"] == 1


if __name__ == "__main__":
    print("Running RAGGenerator relevance gate test suite...")
    test_all_chunks_fail_max_distance()
    print("✓ A. All retrieved chunks fail max_distance passed.")
    test_some_chunks_pass_and_some_fail()
    print("✓ B. Some chunks pass and some fail passed.")
    test_exact_distance_boundary()
    print("✓ C. Exact distance boundary passed.")
    test_empty_vector_store()
    print("✓ D. Empty vector store passed.")
    test_default_max_distance_applied()
    print("✓ E. default_max_distance applied passed.")
    test_query_level_threshold_override()
    print("✓ F. Query-level threshold override passed.")
    test_invalid_negative_max_distance()
    print("✓ G. Invalid negative max_distance validation passed.")
    test_invalid_negative_default_max_distance()
    print("✓ H. Invalid negative default_max_distance validation passed.")
    test_metadata_and_distance_filtering()
    print("✓ I. Metadata + distance filtering passed.")
    test_backward_compatibility()
    print("✓ J. Backward compatibility passed.")
    test_instantiation_with_and_without_reranker()
    print("✓ K. RAGGenerator constructor with and without reranker passed.")
    test_reranker_reordering_execution()
    print("✓ L. RAGGenerator reranker reordering execution passed.")
    print("\nAll 12 relevance gate and reranker tests passed successfully!")
