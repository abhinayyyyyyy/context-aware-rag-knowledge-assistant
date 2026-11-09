"""
Regression Test Suite for Phase 10.5.1: Relevance Gating Improvement.

Tests:
1. AtomQUEST Stage 1 abstract requirements: Answerable from Page 2.
2. AtomQUEST submission format and deadline: Answerable from Page 2.
3. Registration portal or link: Must not be invented (reported as unavailable, sources pruned).
4. FIFA World Cup: Strictly rejected under distance gate (sources == []).
5. Existing Juspay questions: Continues to retrieve accurately with neural reranker.
"""

import os
from typing import Any, Dict, List

from sentence_transformers import CrossEncoder

from src.document_loader import load_pdf
from src.embedding_generator import EmbeddingGenerator
from src.llm_factory import get_llm_provider_instance
from src.rag_generator import RAGGenerator
from src.retriever import Retriever
from src.text_cleaner import clean_documents
from src.text_splitter import split_documents
from src.vector_store import VectorStore

ATOMQUEST_PDF = "data/storage/9430022444ef44b4a28119de94538c64_1789468116558-atomQUEST_2026_Tracks___Stage_1_Details.pdf"
JUSPAY_PDF = "data/documents/sample.pdf"
CALIBRATED_THRESHOLD = 1.60


def build_pipeline_for_pdf(pdf_path: str, emb_gen: EmbeddingGenerator, reranker: CrossEncoder, llm_provider: Any) -> RAGGenerator:
    assert os.path.exists(pdf_path), f"File {pdf_path} does not exist"
    raw_docs = load_pdf(pdf_path)
    cleaned_docs = clean_documents(raw_docs)
    chunks = split_documents(cleaned_docs)

    embeddings = emb_gen.generate_embeddings([c["text"] for c in chunks])
    vector_store = VectorStore(dimension=embeddings.shape[1])
    vector_store.add_documents(chunks, embeddings)

    retriever = Retriever(vector_store=vector_store, embedding_generator=emb_gen)

    return RAGGenerator(
        retriever=retriever,
        model_name=llm_provider.model_name,
        default_max_distance=CALIBRATED_THRESHOLD,
        llm_provider=llm_provider,
        reranker=reranker,
    )


def test_atomquest_stage1_requirements(rag_aq: RAGGenerator) -> None:
    print("\n--- TEST 1: AtomQUEST Stage 1 Requirements ---")
    query = "What are the Stage 1 requirements?"
    result = rag_aq.generate_answer(query)
    answer = result.get("answer", "")
    sources = result.get("sources", [])

    print(f"Query: {query}")
    print(f"Answer:\n{answer}\n")
    print(f"Sources: {sources}")

    assert "I could not find relevant information" not in answer, "Query was falsely rejected!"
    assert len(sources) > 0, "Expected sources from Page 2"
    source_pages = [s.get("page_number") for s in sources]
    assert 2 in source_pages, f"Expected Page 2 in sources, got {source_pages}"

    # Check key grounded facts from Page 2
    lower_ans = answer.lower()
    assert ("11 october" in lower_ans or "october 2026" in lower_ans or "abstract" in lower_ans), "Answer missing Stage 1 key facts"
    print("✓ TEST 1 Passed: Stage 1 requirements accurately answered from Page 2.")


def test_atomquest_submission_process(rag_aq: RAGGenerator) -> None:
    print("\n--- TEST 2: AtomQUEST Submission Process & Deadline ---")
    query = "What is the registration or submission process, if specified in the document?"
    result = rag_aq.generate_answer(query)
    answer = result.get("answer", "")
    sources = result.get("sources", [])

    print(f"Query: {query}")
    print(f"Answer:\n{answer}\n")
    print(f"Sources: {sources}")

    assert "I could not find relevant information" not in answer, "Submission query was falsely rejected!"
    assert len(sources) > 0, "Expected sources from Page 2"
    source_pages = [s.get("page_number") for s in sources]
    assert 2 in source_pages, f"Expected Page 2 in sources, got {source_pages}"

    lower_ans = answer.lower()
    # Must distinguish: specifies abstract submission requirements but does not provide registration link/portal
    assert (
        "specifies abstract submission" in lower_ans
        or "abstract submission requirements" in lower_ans
    ), "Answer must state abstract submission requirements"
    assert (
        "does not provide a registration" in lower_ans
        or "does not specify a registration" in lower_ans
        or "no registration link or portal" in lower_ans
        or "not provide a registration link or portal" in lower_ans
    ), "Answer must state that document does not provide a registration link or portal"

    # Must NOT claim that all submission information / process is absent
    assert (
        "the registration or submission process is not specified" not in lower_ans
        and "the process is not specified" not in lower_ans
    ), "Answer must not claim that the process is not specified when abstract submission is documented"

    assert ("11 october" in lower_ans or "pdf" in lower_ans or "teamname_trackno" in lower_ans), "Answer missing submission details"
    print("✓ TEST 2 Passed: Submission process & deadline accurately answered from Page 2.")


def test_atomquest_registration_link_not_invented(rag_aq: RAGGenerator) -> None:
    print("\n--- TEST 3: Registration Link Not Invented (Absent Information) ---")
    query = "What is the registration link or portal for AtomQUEST?"
    result = rag_aq.generate_answer(query)
    answer = result.get("answer", "")
    sources = result.get("sources", [])

    print(f"Query: {query}")
    print(f"Answer:\n{answer}\n")
    print(f"Sources: {sources}")

    # Must NOT hallucinate a link or portal
    lower_ans = answer.lower()
    for forbidden in ["http://", "https://", "www.", "unstop.com", "forms.gle", "google.com"]:
        assert forbidden not in lower_ans, f"Hallucinated link found: {forbidden}"

    # Must declare unavailable or not mentioned
    assert any(phrase in lower_ans for phrase in [
        "not available", "not mention", "not present", "not specified", "does not contain", "no registration", "not provide"
    ]), "Model failed to declare that registration link is not mentioned in document"

    # Sources must be cleared for unavailable answer
    assert sources == [], f"Expected empty sources for unevidenced query, got {sources}"
    print("✓ TEST 3 Passed: Registration link correctly declared absent, not invented, and sources pruned.")


def test_fifa_world_cup_rejection(rag_aq: RAGGenerator, rag_juspay: RAGGenerator) -> None:
    print("\n--- TEST 4: FIFA World Cup Strict Distance Gate Rejection ---")
    query = "Who won the 2022 FIFA World Cup in Qatar?"

    # Test on AtomQUEST
    res_aq = rag_aq.generate_answer(query)
    print(f"AtomQUEST FIFA Answer: {res_aq['answer']}")
    assert res_aq["answer"] == "I could not find relevant information in the provided documents."
    assert res_aq["sources"] == []

    # Test on Juspay
    res_jp = rag_juspay.generate_answer(query)
    print(f"Juspay FIFA Answer: {res_jp['answer']}")
    assert res_jp["answer"] == "I could not find relevant information in the provided documents."
    assert res_jp["sources"] == []

    print("✓ TEST 4 Passed: Unrelated FIFA query strictly rejected on both corpora without calling LLM.")


def test_juspay_existing_questions(rag_juspay: RAGGenerator) -> None:
    print("\n--- TEST 5: Existing Juspay Questions Functionality ---")
    q1 = "What national digital initiatives has Juspay contributed to?"
    res1 = rag_juspay.generate_answer(q1)
    ans1 = res1.get("answer", "")
    src1 = res1.get("sources", [])
    print(f"Juspay Q1 Answer Preview: {ans1[:120]}...")
    assert len(src1) > 0
    assert any(term in ans1.lower() for term in ["ocen", "ondc", "namma yatri"]), "Juspay Q1 missing key initiatives"

    q2 = "How does Juspay use AI and feedback loops for payment routing?"
    res2 = rag_juspay.generate_answer(q2)
    ans2 = res2.get("answer", "")
    src2 = res2.get("sources", [])
    print(f"Juspay Q2 Answer Preview: {ans2[:120]}...")
    assert len(src2) > 0
    assert any(term in ans2.lower() for term in ["feedback", "loop", "routing", "reinforcement"]), "Juspay Q2 missing mechanism details"

    print("✓ TEST 5 Passed: Existing Juspay queries continue to work with grounded evidence.")


def main() -> None:
    print("=" * 80)
    print("PHASE 10.5.1 RELEVANCE GATING REGRESSION TEST SUITE")
    print("=" * 80)

    emb_gen = EmbeddingGenerator("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm_provider = get_llm_provider_instance()

    print("\nInitializing AtomQUEST pipeline...")
    rag_aq = build_pipeline_for_pdf(ATOMQUEST_PDF, emb_gen, reranker, llm_provider)

    print("Initializing Juspay pipeline...")
    rag_jp = build_pipeline_for_pdf(JUSPAY_PDF, emb_gen, reranker, llm_provider)

    test_atomquest_stage1_requirements(rag_aq)
    test_atomquest_submission_process(rag_aq)
    test_atomquest_registration_link_not_invented(rag_aq)
    test_fifa_world_cup_rejection(rag_aq, rag_jp)
    test_juspay_existing_questions(rag_jp)

    print("\n" + "=" * 80)
    print("ALL 5 PHASE 10.5.1 REGRESSION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
