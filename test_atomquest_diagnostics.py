"""
Regression and Diagnostic Test Suite for AtomQUEST 2026 Retrieval.

Diagnoses why queries regarding Stage 1 requirements and submission processes
return the fallback response under the production relevance gate (1.10).

Constraints:
- Strictly diagnostic and regression testing.
- Does not modify any production files (src/*.py, app.py).
"""

import os
from typing import Any, Dict, List

from src.document_loader import load_pdf
from src.text_cleaner import clean_documents
from src.text_splitter import split_documents
from src.embedding_generator import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import Retriever


ATOMQUEST_PDF_PATH = "data/storage/9430022444ef44b4a28119de94538c64_1789468116558-atomQUEST_2026_Tracks___Stage_1_Details.pdf"
PRODUCTION_THRESHOLD = 1.10


def run_atomquest_diagnostics() -> Dict[str, Any]:
    # 1. Load document
    assert os.path.exists(ATOMQUEST_PDF_PATH), f"PDF not found at {ATOMQUEST_PDF_PATH}"
    raw_docs = load_pdf(ATOMQUEST_PDF_PATH)
    assert len(raw_docs) == 2, f"Expected 2 pages, got {len(raw_docs)}"

    # 2. Text inspection
    page1_text = raw_docs[0]["text"]
    page2_text = raw_docs[1]["text"]

    # Verify Stage 1 submission exists on Page 2
    assert "STAGE 1" in page2_text
    assert "ABSTRACT SUBMISSION" in page2_text
    assert "11 October 2026" in page2_text

    # Verify registration is absent from the PDF
    full_text = (page1_text + " " + page2_text).lower()
    assert "registration" not in full_text
    assert "register" not in full_text

    # 3. Clean and split
    cleaned_docs = clean_documents(raw_docs)
    chunks = split_documents(cleaned_docs)
    assert len(chunks) == 9, f"Expected 9 chunks, got {len(chunks)}"

    # Page 2 chunks are chunks 6, 7, 8
    page2_chunks = [c for c in chunks if c["page_number"] == 2]
    assert len(page2_chunks) == 3

    # 4. Embed and Index
    emb_gen = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    embeddings = emb_gen.generate_embeddings([c["text"] for c in chunks])
    vector_store = VectorStore(dimension=embeddings.shape[1])
    vector_store.add_documents(chunks, embeddings)
    retriever = Retriever(vector_store=vector_store, embedding_generator=emb_gen)

    # 5. Queries
    test_queries = [
        {
            "id": "Q_STAGE1",
            "query": "What are the Stage 1 requirements?",
            "is_answerable": True,
            "expected_top_page": 2,
        },
        {
            "id": "Q_SUBMISSION",
            "query": "What is the registration or submission process, if specified in the document?",
            "is_answerable": True,  # Submission is answerable, registration is absent
            "expected_top_page": 2,
        },
        {
            "id": "Q_TRACKS",
            "query": "What are the three product tracks in AtomQUEST?",
            "is_answerable": True,
            "expected_top_page": [1, 2],
        },
        {
            "id": "Q_FIFA",
            "query": "Who won the 2022 FIFA World Cup in Qatar?",
            "is_answerable": False,
            "expected_top_page": None,
        },
    ]

    results = {}
    for tq in test_queries:
        q = tq["query"]
        raw_top5 = retriever.retrieve(q, top_k=5, max_distance=None)
        gated_top3 = retriever.retrieve(q, top_k=3, max_distance=PRODUCTION_THRESHOLD)

        results[tq["id"]] = {
            "query": q,
            "raw_top5": raw_top5,
            "gated_top3": gated_top3,
            "top1_distance": raw_top5[0]["distance"] if raw_top5 else None,
            "top1_page": raw_top5[0]["document"]["page_number"] if raw_top5 else None,
            "gated_count": len(gated_top3),
        }

    return results


def test_atomquest_retrieval_diagnostic_assertions():
    results = run_atomquest_diagnostics()

    # Query 1: Stage 1 requirements
    # Confirms retrieval ranked Page 2 at Rank 1, but distance exceeds 1.10
    q1 = results["Q_STAGE1"]
    assert q1["top1_page"] == 2
    assert q1["top1_distance"] > PRODUCTION_THRESHOLD  # ~1.3049
    assert q1["gated_count"] == 0  # Blocked by 1.10 gate

    # Query 2: Registration or submission process
    # Confirms retrieval ranked Page 2 at Rank 1, but distance exceeds 1.10
    q2 = results["Q_SUBMISSION"]
    assert q2["top1_page"] == 2
    assert q2["top1_distance"] > PRODUCTION_THRESHOLD  # ~1.5422
    assert q2["gated_count"] == 0  # Blocked by 1.10 gate

    # Query 3: Tracks query passes the gate
    q3 = results["Q_TRACKS"]
    assert q3["top1_distance"] < PRODUCTION_THRESHOLD  # ~0.8250
    assert q3["gated_count"] > 0

    # Query 4: FIFA query is blocked by gate
    q4 = results["Q_FIFA"]
    assert q4["top1_distance"] > PRODUCTION_THRESHOLD  # ~1.8873
    assert q4["gated_count"] == 0

    print("All diagnostic assertions passed successfully!")


if __name__ == "__main__":
    test_atomquest_retrieval_diagnostic_assertions()
