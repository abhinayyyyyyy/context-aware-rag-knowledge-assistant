"""
Validation Test Suite for RAG Evaluation Dataset (Phase 11.1).

Validates:
1. Dataset file presence and valid JSON structure.
2. Top-level metadata and taxonomy definitions.
3. Existence and page count parity of referenced source documents.
4. Unique non-empty test case identifiers.
5. Required field schema, non-empty questions, and valid types.
6. Category-specific invariants for answerable, partially answerable, absent, and unrelated queries.
7. Coverage tracking for all previously tested questions across AtomQUEST and Juspay.
8. Verification status tracking.
"""

import json
import os
from typing import Any, Dict, List

from src.document_loader import load_pdf

DATASET_PATH = "data/evaluation/rag_eval_dataset.json"
VALID_CATEGORIES = {"answerable", "partially_answerable", "absent", "unrelated"}
VALID_STATUSES = {"verified", "needs_review"}

HISTORICAL_QUESTIONS = [
    "What are the Stage 1 requirements?",
    "What is the registration or submission process, if specified in the document?",
    "What is the registration link or portal for AtomQUEST?",
    "Who won the 2022 FIFA World Cup in Qatar?",
    "What national digital initiatives has Juspay contributed to?",
    "How does Juspay use AI and feedback loops for payment routing?",
    "What does Juspay do?",
    "What are the three product tracks in AtomQUEST?",
]


def load_dataset() -> Dict[str, Any]:
    """Helper to load and return the dataset JSON."""
    assert os.path.exists(DATASET_PATH), f"Dataset file not found at {DATASET_PATH}"
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def test_dataset_file_and_top_level_keys() -> None:
    print("\n--- TEST: Dataset File & Top-Level Keys ---")
    data = load_dataset()
    for key in ["version", "description", "categories", "documents", "test_cases"]:
        assert key in data, f"Missing required top-level key: {key}"

    # Verify categories
    categories = data["categories"]
    assert isinstance(categories, dict), "categories must be a dictionary"
    assert set(categories.keys()) == VALID_CATEGORIES, (
        f"Categories mismatch: expected {VALID_CATEGORIES}, got {set(categories.keys())}"
    )

    print("✓ Dataset file and top-level metadata keys verified.")


def test_referenced_documents_exist_and_page_counts_match() -> None:
    print("\n--- TEST: Document References & Actual Page Counts ---")
    data = load_dataset()
    documents = data["documents"]
    assert len(documents) >= 2, "Expected at least 2 document definitions (atomquest, juspay)"

    for doc_key, doc_meta in documents.items():
        doc_path = doc_meta.get("path")
        assert doc_path, f"Document {doc_key} missing path"
        assert os.path.exists(doc_path), f"Referenced document path does not exist: {doc_path}"

        expected_pages = doc_meta.get("total_pages")
        assert isinstance(expected_pages, int) and expected_pages > 0

        # Load actual PDF and verify total pages match
        actual_docs = load_pdf(doc_path)
        actual_page_count = len(actual_docs)
        assert actual_page_count == expected_pages, (
            f"Page count mismatch for {doc_key}: metadata says {expected_pages}, actual is {actual_page_count}"
        )
        print(f"  - Document '{doc_key}': Verified path '{doc_path}' with {actual_page_count} pages.")

    print("✓ All referenced documents exist with exact verified page counts.")


def test_unique_ids_and_required_fields() -> None:
    print("\n--- TEST: Unique IDs & Required Field Typing ---")
    data = load_dataset()
    test_cases = data["test_cases"]
    assert isinstance(test_cases, list) and len(test_cases) > 0, "test_cases must be a non-empty list"

    seen_ids = set()
    required_fields = [
        "id",
        "question",
        "category",
        "expected_behavior",
        "expected_key_facts",
        "source_document",
        "expected_pages",
        "verification_status",
    ]

    for tc in test_cases:
        tc_id = tc.get("id")
        assert tc_id, f"Missing id in test case: {tc}"
        assert isinstance(tc_id, str) and tc_id.strip(), f"id must be non-empty string: {tc_id}"
        assert tc_id not in seen_ids, f"Duplicate test case id found: {tc_id}"
        seen_ids.add(tc_id)

        # Check all required fields present
        for f in required_fields:
            assert f in tc, f"Test case {tc_id} missing required field '{f}'"

        # Check field types
        question = tc["question"]
        assert isinstance(question, str) and question.strip(), f"Test case {tc_id} has empty question"

        category = tc["category"]
        assert category in VALID_CATEGORIES, f"Test case {tc_id} has invalid category '{category}'"

        behavior = tc["expected_behavior"]
        assert isinstance(behavior, str) and behavior.strip(), f"Test case {tc_id} has empty expected_behavior"

        facts = tc["expected_key_facts"]
        assert isinstance(facts, list), f"Test case {tc_id}: expected_key_facts must be a list"
        for fact in facts:
            assert isinstance(fact, str) and fact.strip(), f"Test case {tc_id} has empty fact in expected_key_facts"

        expected_pages = tc["expected_pages"]
        assert isinstance(expected_pages, list), f"Test case {tc_id}: expected_pages must be a list"
        for p in expected_pages:
            assert isinstance(p, int) and p > 0, f"Test case {tc_id}: page number must be positive integer, got {p}"

        status = tc["verification_status"]
        assert status in VALID_STATUSES, f"Test case {tc_id}: invalid verification_status '{status}'"

    print(f"✓ All {len(test_cases)} test cases have unique IDs, non-empty questions, and valid field types.")


def test_category_invariants() -> None:
    print("\n--- TEST: Category Invariants ---")
    data = load_dataset()
    test_cases = data["test_cases"]

    counts = {cat: 0 for cat in VALID_CATEGORIES}

    for tc in test_cases:
        tc_id = tc["id"]
        category = tc["category"]
        counts[category] += 1

        source_doc = tc["source_document"]
        pages = tc["expected_pages"]
        facts = tc["expected_key_facts"]

        if category == "answerable":
            assert source_doc is not None, f"{tc_id}: answerable case must specify source_document"
            assert os.path.exists(source_doc), f"{tc_id}: source_document '{source_doc}' does not exist"
            assert len(pages) > 0, f"{tc_id}: answerable case must have at least one expected page"
            assert len(facts) > 0, f"{tc_id}: answerable case must have at least one expected key fact"

        elif category == "partially_answerable":
            assert source_doc is not None, f"{tc_id}: partially_answerable case must specify source_document"
            assert os.path.exists(source_doc), f"{tc_id}: source_document '{source_doc}' does not exist"
            assert len(pages) > 0, f"{tc_id}: partially_answerable case must have at least one documented page"
            assert len(facts) > 0, f"{tc_id}: partially_answerable case must specify expected facts for documented parts"

        elif category == "absent":
            assert source_doc is not None, f"{tc_id}: absent case must specify target source_document"
            assert os.path.exists(source_doc), f"{tc_id}: source_document '{source_doc}' does not exist"
            assert pages == [], f"{tc_id}: absent case must have empty expected_pages, got {pages}"

        elif category == "unrelated":
            assert source_doc is None, f"{tc_id}: unrelated case must have source_document None, got {source_doc}"
            assert pages == [], f"{tc_id}: unrelated case must have empty expected_pages, got {pages}"
            assert facts == [], f"{tc_id}: unrelated case must have empty expected_key_facts, got {facts}"

    print("  Category distribution:")
    for cat, cnt in counts.items():
        print(f"    - {cat:22s}: {cnt} test cases")
        assert cnt > 0, f"Category '{cat}' must have at least one test case"

    print("✓ Category-specific structural invariants passed across all categories.")


def test_historical_questions_covered() -> None:
    print("\n--- TEST: Historical Questions Coverage ---")
    data = load_dataset()
    dataset_questions = {tc["question"].strip().lower() for tc in data["test_cases"]}

    missing = []
    for hq in HISTORICAL_QUESTIONS:
        if hq.strip().lower() not in dataset_questions:
            missing.append(hq)

    assert not missing, f"The following historical questions are missing from the dataset: {missing}"
    print(f"✓ All {len(HISTORICAL_QUESTIONS)} historical questions from previous phases are present in dataset.")


def test_verification_status_all_verified() -> None:
    print("\n--- TEST: Verification Status ---")
    data = load_dataset()
    unverified = [tc["id"] for tc in data["test_cases"] if tc.get("verification_status") != "verified"]

    assert not unverified, f"The following test cases require review/unverified: {unverified}"
    print(f"✓ All {len(data['test_cases'])} test cases verified against document text.")


def main() -> None:
    print("=" * 80)
    print("RAG EVALUATION DATASET VALIDATION TEST SUITE (PHASE 11.1)")
    print("=" * 80)

    test_dataset_file_and_top_level_keys()
    test_referenced_documents_exist_and_page_counts_match()
    test_unique_ids_and_required_fields()
    test_category_invariants()
    test_historical_questions_covered()
    test_verification_status_all_verified()

    print("\n" + "=" * 80)
    print("ALL DATASET VALIDATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
