"""
Diagnostic script for inspecting retrieval behavior on calibration queries Q3 and Q4.

Purpose:
--------
Diagnose why queries Q3 and Q4 do not retrieve the expected page (Page 2) at Rank 1.
This script is strictly diagnostic:
- No LLM / Ollama calls
- No changes to chunking, embeddings, or retrieval thresholds
- Uses existing project components: load_pdf, clean_documents, split_documents,
  EmbeddingGenerator, VectorStore, and Retriever.
"""

from typing import Any, Dict, List

from src.document_loader import load_pdf
from src.text_cleaner import clean_documents
from src.text_splitter import split_documents
from src.embedding_generator import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import Retriever


DIAGNOSTIC_QUERIES = [
    {
        "id": "Q3",
        "query": "What is the transaction volume and uptime availability of Juspay systems?",
        "expected_page": 2,
    },
    {
        "id": "Q4",
        "query": "How does Juspay use AI and feedback loops for payment routing?",
        "expected_page": 2,
    },
]


def run_diagnostics(pdf_path: str = "data/documents/sample.pdf") -> None:
    print("=" * 80)
    print("RETRIEVAL DIAGNOSTICS FOR CALIBRATION QUERIES Q3 & Q4")
    print("=" * 80)

    # 1. Load PDF
    print(f"\n[Step 1] Loading document from: {pdf_path}")
    raw_docs = load_pdf(pdf_path)
    print(f"Loaded {len(raw_docs)} raw document page(s).")

    # 2. Run existing document processing pipeline (clean & split)
    print("\n[Step 2] Cleaning and splitting documents...")
    cleaned_docs = clean_documents(raw_docs)
    chunks = split_documents(cleaned_docs)
    print(f"Generated {len(chunks)} chunk(s) across {len(cleaned_docs)} cleaned page(s).")

    # 3. Generate embeddings using existing model
    print("\n[Step 3] Generating embeddings with all-MiniLM-L6-v2...")
    embedding_generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embedding_generator.generate_embeddings(chunk_texts)
    print(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}.")

    # 4. Build existing FAISS VectorStore & Retriever
    print("\n[Step 4] Initializing FAISS VectorStore and Retriever...")
    vector_store = VectorStore(dimension=embeddings.shape[1])
    vector_store.add_documents(chunks, embeddings)
    retriever = Retriever(vector_store=vector_store, embedding_generator=embedding_generator)

    # -----------------------------------------------------------------------
    # Print All Chunks Belonging to Page 1 and Page 2 (Step 8)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("INVENTORY OF INDEXED CHUNKS FOR PAGE 1 AND PAGE 2")
    print("=" * 80)

    for target_page in [1, 2]:
        page_chunks = [c for c in chunks if c.get("page_number") == target_page]
        print(f"\n--- Page {target_page} ({len(page_chunks)} Chunks) ---")
        for idx, ch in enumerate(page_chunks, start=1):
            print(f"\n[Page {target_page} | Chunk {idx}]")
            print(f"Chunk ID:    {ch.get('chunk_id')}")
            print(f"Page Number: {ch.get('page_number')}")
            print("Full Text:")
            print("-" * 60)
            print(ch.get("text", ""))
            print("-" * 60)

    # -----------------------------------------------------------------------
    # Run Retrieval for Q3 and Q4 and Print Top 5 Chunks (Steps 5, 6, 7)
    # -----------------------------------------------------------------------
    diagnostic_summaries: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print("DETAILED RETRIEVAL RESULTS (TOP 5 CHUNKS PER QUERY)")
    print("=" * 80)

    for item in DIAGNOSTIC_QUERIES:
        qid = item["id"]
        query_text = item["query"]
        expected_page = item["expected_page"]

        print(f"\n{'#' * 80}")
        print(f"Query {qid}: \"{query_text}\"")
        print(f"Expected Target Page: {expected_page}")
        print(f"{'#' * 80}")

        results = retriever.retrieve(query_text, top_k=5)

        rank_1_item = results[0] if results else None
        rank_1_page = rank_1_item["document"].get("page_number") if rank_1_item else None
        rank_1_dist = rank_1_item.get("distance") if rank_1_item else None

        first_expected_match_rank = None
        first_expected_match_item = None

        for rank, res in enumerate(results, start=1):
            doc = res.get("document", {})
            distance = res.get("distance", float("inf"))
            page_num = doc.get("page_number")
            source = doc.get("source")
            chunk_id = doc.get("chunk_id")
            full_text = doc.get("text", "")

            is_rank_1 = (rank == 1)
            is_expected_page = (page_num == expected_page)

            if is_expected_page and first_expected_match_rank is None:
                first_expected_match_rank = rank
                first_expected_match_item = res

            print(f"\n------------------------------------------------------------")
            print(f"Query:       {query_text}")
            print(f"Rank:        {rank}" + (" [RANK 1 RESULT]" if is_rank_1 else ""))
            print(f"Distance:    {distance:.4f}")
            print(f"Page Number: {page_num}" + (f" [MATCHES EXPECTED PAGE {expected_page}]" if is_expected_page else f" (Expected: Page {expected_page})"))
            print(f"Source:      {source}")
            print(f"Chunk ID:    {chunk_id}")
            print(f"Full Text:\n{full_text}")

        # Analysis of Rank 1 vs Expected Page Result
        print("\n--- Target Alignment Inspection ---")
        if rank_1_item:
            print(f"Rank 1 Result: Page {rank_1_page} | Distance {rank_1_dist:.4f} | Chunk ID {rank_1_item['document'].get('chunk_id')}")

        if first_expected_match_rank is not None and first_expected_match_item is not None:
            expected_dist = first_expected_match_item.get("distance", float("inf"))
            dist_diff = expected_dist - rank_1_dist
            print(f"First Result Matching Expected Page {expected_page}:")
            print(f"  Rank:                {first_expected_match_rank}")
            print(f"  Distance:            {expected_dist:.4f}")
            print(f"  Chunk ID:            {first_expected_match_item['document'].get('chunk_id')}")
            print(f"  Distance Difference: {dist_diff:+.4f} (Expected Dist - Rank 1 Dist)")
        else:
            dist_diff = None
            print(f"No results matching expected Page {expected_page} were found in the top 5 retrieved chunks.")

        diagnostic_summaries.append({
            "qid": qid,
            "query": query_text,
            "expected_page": expected_page,
            "rank_1_page": rank_1_page,
            "rank_1_dist": rank_1_dist,
            "expected_page_rank": first_expected_match_rank,
            "expected_page_dist": first_expected_match_item.get("distance") if first_expected_match_item else None,
            "distance_difference": dist_diff,
        })

    # -----------------------------------------------------------------------
    # Step 9: Diagnostic Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 9: DIAGNOSTIC SUMMARY")
    print("=" * 80)

    for summary in diagnostic_summaries:
        qid = summary["qid"]
        query = summary["query"]
        exp_p = summary["expected_page"]
        r1_p = summary["rank_1_page"]
        r1_d = summary["rank_1_dist"]
        exp_r = summary["expected_page_rank"]
        exp_d = summary["expected_page_dist"]
        diff = summary["distance_difference"]

        print(f"\n--- Diagnostic Summary for {qid} ---")
        print(f"Query:                 \"{query}\"")
        print(f"Expected Target Page:  Page {exp_p}")
        print(f"Rank 1 Page:           Page {r1_p} (Distance: {r1_d:.4f})")
        print(f"Expected Page Rank:    Rank {exp_r} (Distance: {exp_d:.4f})")
        print(f"Distance Gap:          {diff:+.4f}")
        print("Observation:")
        if exp_r == 1:
            print(f"  - Query {qid} retrieved Page {exp_p} at Rank 1.")
        elif exp_r is not None:
            print(
                f"  - The expected Page {exp_p} context was retrieved within top-k (at Rank {exp_r}), "
                f"but Page {r1_p} ranked higher by {diff:.4f} in FAISS L2 distance."
            )
        else:
            print(f"  - The expected Page {exp_p} context did not appear in the top 5 retrieved candidates.")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_diagnostics()
