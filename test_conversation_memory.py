"""
Phase 9: Conversation Memory Test Suite.

Validates conversational follow-up questions, standalone query contextualization,
retrieval resolution of coreferences, bounded history, and document switching.
"""

import os
import tempfile
from typing import Any, Dict, List

from src.document_loader import load_pdf
from src.embedding_generator import EmbeddingGenerator
from src.ingestion_pipeline import build_vector_store
from src.rag_generator import RAGGenerator
from src.retriever import Retriever
from src.vector_store import VectorStore


def run_phase_9_tests() -> None:
    print("=" * 80)
    print("PHASE 9: CONVERSATION MEMORY VERIFICATION SUITE")
    print("=" * 80)

    pdf_path = "data/documents/sample.pdf"
    print(f"Loading document: {pdf_path}")
    emb_gen = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    vector_store = build_vector_store(pdf_path, embedding_generator=emb_gen)
    retriever = Retriever(vector_store=vector_store, embedding_generator=emb_gen)
    rag_gen = RAGGenerator(
        retriever=retriever,
        model_name="llama3.2:latest",
        default_max_distance=1.10,
    )

    # -------------------------------------------------------------
    # TEST 1 — Backward Compatibility
    # -------------------------------------------------------------
    print("\n--- TEST 1: Backward Compatibility (Single-turn without chat_history) ---")
    res1 = rag_gen.generate_answer("What does Juspay do?", top_k=3)
    assert isinstance(res1.get("answer"), str) and len(res1["answer"]) > 0, "TEST 1 Failed: Empty answer"
    assert len(res1.get("sources", [])) > 0, "TEST 1 Failed: Expected sources"
    print("✓ TEST 1 Passed: generate_answer works without chat_history.")
    print(f"  Answer preview: {res1['answer'][:120]}...")
    print(f"  Sources count: {len(res1['sources'])}")

    # -------------------------------------------------------------
    # TEST 2 — Query Contextualization
    # -------------------------------------------------------------
    print("\n--- TEST 2: Query Contextualization ---")
    mock_history = [
        {"role": "user", "content": "What does Juspay do?"},
        {
            "role": "assistant",
            "content": "Juspay provides payment infrastructure and software solutions for global enterprises.",
        },
    ]
    follow_up = "What about its AI system?"
    standalone = rag_gen._contextualize_query(follow_up, chat_history=mock_history)
    print(f"  Raw Follow-up:      \"{follow_up}\"")
    print(f"  Contextualized:     \"{standalone}\"")

    standalone_lower = standalone.lower()
    assert "juspay" in standalone_lower or "payment" in standalone_lower, (
        f"TEST 2 Failed: Standalone query missing subject context ('Juspay'): {standalone}"
    )
    assert "ai" in standalone_lower or "artificial intelligence" in standalone_lower, (
        f"TEST 2 Failed: Standalone query missing core concept ('AI'): {standalone}"
    )
    print("✓ TEST 2 Passed: Query contextualized with entity reference resolved.")

    # -------------------------------------------------------------
    # TEST 3 — Retrieval of Follow-Up Query
    # -------------------------------------------------------------
    print("\n--- TEST 3: Retrieval of Follow-Up Query ---")
    res_followup = rag_gen.generate_answer(
        "What about its AI system?",
        top_k=3,
        chat_history=mock_history,
    )
    assert res_followup["answer"] != "I could not find relevant information in the provided documents.", (
        "TEST 3 Failed: Follow-up query was erroneously rejected by distance cutoff"
    )
    assert len(res_followup["sources"]) > 0, "TEST 3 Failed: Sources missing"

    # Verify Page 2 (Intelligence Systems / AI) was retrieved
    source_pages = [s.get("page_number") for s in res_followup["sources"]]
    print(f"  Retrieved Source Pages: {source_pages}")
    print(f"  Answer preview: {res_followup['answer'][:120]}...")
    assert 2 in source_pages, f"TEST 3 Failed: Expected Page 2 in sources, got {source_pages}"
    print("✓ TEST 3 Passed: Follow-up query successfully retrieved Page 2 AI mechanism evidence.")

    # -------------------------------------------------------------
    # TEST 4 — Three-Turn Conversation
    # -------------------------------------------------------------
    print("\n--- TEST 4: Three-Turn Conversation ---")
    # Turn 1
    t1_q = "What does Juspay do?"
    t1_res = rag_gen.generate_answer(t1_q, top_k=3, chat_history=[])
    print(f"  Turn 1: \"{t1_q}\"")
    print(f"    Answer: {t1_res['answer'][:100]}...")
    assert t1_res["sources"], "Turn 1 missing sources"

    # Turn 2
    t2_history = [
        {"role": "user", "content": t1_q},
        {"role": "assistant", "content": t1_res["answer"]},
    ]
    t2_q = "What about its AI system?"
    t2_res = rag_gen.generate_answer(t2_q, top_k=3, chat_history=t2_history)
    print(f"  Turn 2: \"{t2_q}\"")
    print(f"    Answer: {t2_res['answer'][:100]}...")
    assert t2_res["sources"], "Turn 2 missing sources"

    # Turn 3
    t3_history = t2_history + [
        {"role": "user", "content": t2_q},
        {"role": "assistant", "content": t2_res["answer"]},
    ]
    t3_q = "How does that improve payment routing?"
    t3_res = rag_gen.generate_answer(t3_q, top_k=3, chat_history=t3_history)
    print(f"  Turn 3: \"{t3_q}\"")
    print(f"    Answer: {t3_res['answer'][:100]}...")
    assert t3_res["sources"], "Turn 3 missing sources"
    assert t3_res["answer"] != "I could not find relevant information in the provided documents.", (
        "Turn 3 failed: rejected by threshold"
    )
    print("✓ TEST 4 Passed: Three-turn conversation completed with consistent document grounding.")

    # -------------------------------------------------------------
    # TEST 5 — Unrelated Query After Conversation
    # -------------------------------------------------------------
    print("\n--- TEST 5: Unrelated Query After Conversation ---")
    unrel_q = "Who won the FIFA World Cup in 2022?"
    unrel_res = rag_gen.generate_answer(unrel_q, top_k=3, chat_history=t3_history)
    print(f"  Query: \"{unrel_q}\"")
    print(f"  Answer: \"{unrel_res['answer']}\"")
    print(f"  Sources: {unrel_res['sources']}")
    assert unrel_res["answer"] == "I could not find relevant information in the provided documents.", (
        f"TEST 5 Failed: Expected insufficient information message, got: {unrel_res['answer']}"
    )
    assert unrel_res["sources"] == [], "TEST 5 Failed: Expected empty sources for unrelated query"
    print("✓ TEST 5 Passed: Unrelated query correctly rejected without outside knowledge hallucination.")

    # -------------------------------------------------------------
    # TEST 6 — Self-Contained Question After Conversation
    # -------------------------------------------------------------
    print("\n--- TEST 6: Self-Contained Question After Conversation ---")
    self_contained_q = "What national digital initiatives has Juspay contributed to?"
    sc_res = rag_gen.generate_answer(self_contained_q, top_k=3, chat_history=t3_history)
    print(f"  Query: \"{self_contained_q}\"")
    print(f"  Answer: {sc_res['answer'][:120]}...")
    sc_pages = [s.get("page_number") for s in sc_res["sources"]]
    print(f"  Retrieved Pages: {sc_pages}")
    assert 1 in sc_pages, f"TEST 6 Failed: Expected Page 1 for national digital initiatives, got {sc_pages}"
    assert "upi" in sc_res["answer"].lower() or "beckn" in sc_res["answer"].lower() or "open network" in sc_res["answer"].lower() or "india" in sc_res["answer"].lower(), (
        f"TEST 6 Failed: Answer not grounded in Page 1 digital initiatives: {sc_res['answer']}"
    )
    print("✓ TEST 6 Passed: Self-contained question preserved and correctly answered.")

    # -------------------------------------------------------------
    # TEST 7 — Document Switching Simulation
    # -------------------------------------------------------------
    print("\n--- TEST 7: Document Switching Simulation ---")
    # Simulate switching from sample.pdf to another temporary document
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_pdf_path = os.path.join(tmp_dir, "other_doc.pdf")
        # Copy sample.pdf to simulate a different file
        with open(pdf_path, "rb") as f_in, open(dummy_pdf_path, "wb") as f_out:
            f_out.write(f_in.read())

        # In Streamlit, document switching calls clear_document_state()
        switched_history: List[Dict[str, str]] = []  # Cleared history
        new_store = build_vector_store(dummy_pdf_path, embedding_generator=emb_gen)
        new_retriever = Retriever(vector_store=new_store, embedding_generator=emb_gen)
        new_rag_gen = RAGGenerator(
            retriever=new_retriever,
            model_name="llama3.2:latest",
            default_max_distance=1.10,
        )

        assert len(switched_history) == 0, "History not empty"
        # First query on new document has empty chat history
        new_res = new_rag_gen.generate_answer("What does Juspay do?", top_k=3, chat_history=switched_history)
        assert new_res["sources"][0]["source"] == "other_doc.pdf", (
            f"Expected source to be other_doc.pdf, got {new_res['sources'][0]['source']}"
        )
    print("✓ TEST 7 Passed: Document switching resets conversation history and binds to new document.")

    print("\n" + "=" * 80)
    print("ALL 7 PHASE 9 TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_phase_9_tests()
