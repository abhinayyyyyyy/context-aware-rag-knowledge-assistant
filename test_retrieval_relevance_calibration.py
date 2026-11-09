"""
Retrieval Relevance Calibration and Distance Distribution Evaluation Suite.

Purpose:
--------
This script empirically evaluates FAISS L2 distance distributions for the
retrieval layer using real embeddings (all-MiniLM-L6-v2) and VectorStore/Retriever
against data/documents/sample.pdf.

It does NOT invoke Ollama or perform any LLM generation. It focuses strictly on
retrieval relevance and semantic distance calibration across three query categories:
  1. Answerable queries targeting specific pages/sections of the document
  2. Ambiguous/boundary queries (adjacent domain but content absent)
  3. Unrelated/out-of-domain queries (sports, cooking, science)

The outputs provide measured distance statistics, target context match validation,
separation gap analysis, and candidate threshold simulations to inform future
threshold calibration without modifying production defaults.
"""

import os
import statistics
from typing import Any, Dict, List, Optional

from src.embedding_generator import EmbeddingGenerator
from src.ingestion_pipeline import build_vector_store
from src.retriever import Retriever


# ---------------------------------------------------------------------------
# Evaluation Query Dataset Definition
# ---------------------------------------------------------------------------

EVALUATION_QUERIES: List[Dict[str, Any]] = [
    # Category 1: Clearly Answerable Queries
    {
        "category": "Answerable",
        "query": "What national digital initiatives has Juspay contributed to?",
        "expected_pages": [1],
        "description": "Page 1 - Mentions OCEN, HyperCredit, ONDC, and Namma Yatri.",
    },
    {
        "category": "Answerable",
        "query": "What services are offered in Juspay's payment suite?",
        "expected_pages": [1],
        "description": "Page 1 - Mentions orchestration, tokenization, authentication, and fraud.",
    },
    {
        "category": "Answerable",
        "query": "What is the transaction volume and uptime availability of Juspay systems?",
        "expected_pages": [2],
        "description": "Page 2 - Mentions 350M+ daily transactions and 99.999% availability.",
    },
    {
        "category": "Answerable",
        "query": "How does Juspay use AI and feedback loops for payment routing?",
        "expected_pages": [2],
        "description": "Page 2 - Mentions control-theoretic feedback loops and AI routing engine.",
    },
    {
        "category": "Answerable",
        "query": "What is Xyne Spaces and what enterprise signals does it unify?",
        "expected_pages": [3],
        "description": "Page 3 - Mentions AI-native workspace unifying tickets, workflows, conversations.",
    },
    {
        "category": "Answerable",
        "query": "What is Namma DSL and what language is it built with?",
        "expected_pages": [3],
        "description": "Page 3 - Mentions Rust-based declarative infrastructure language.",
    },

    # Category 2: Ambiguous / Boundary Queries
    {
        "category": "Ambiguous/Boundary",
        "query": "How do I integrate a checkout payment gateway on my website?",
        "expected_pages": [],
        "description": "Payments-adjacent topic, but developer checkout integration steps are not in the PDF.",
    },
    {
        "category": "Ambiguous/Boundary",
        "query": "What cloud providers are supported for cloud deployment?",
        "expected_pages": [],
        "description": "Infrastructure-adjacent, but external cloud vendor support is not specified in the PDF.",
    },
    {
        "category": "Ambiguous/Boundary",
        "query": "What is the salary range and interview process for software engineers?",
        "expected_pages": [],
        "description": "Hiring-adjacent culture section exists on Page 3, but compensation details are absent.",
    },

    # Category 3: Unrelated / Out-of-Domain Queries
    {
        "category": "Unrelated",
        "query": "Who won the FIFA World Cup in 2022?",
        "expected_pages": [],
        "description": "Sports trivia - completely out of domain.",
    },
    {
        "category": "Unrelated",
        "query": "How do photosynthesis and chlorophyll work in plants?",
        "expected_pages": [],
        "description": "Biological science - completely out of domain.",
    },
    {
        "category": "Unrelated",
        "query": "What is the recipe for making sourdough bread from scratch?",
        "expected_pages": [],
        "description": "Culinary arts - completely out of domain.",
    },
    {
        "category": "Unrelated",
        "query": "Explain Einstein's theory of general relativity.",
        "expected_pages": [],
        "description": "Theoretical physics - completely out of domain.",
    },
]

# Candidate thresholds for simulation analysis only (no production modification)
SIMULATION_THRESHOLDS: List[float] = [0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40]


# ---------------------------------------------------------------------------
# Core Evaluation Runner
# ---------------------------------------------------------------------------

def run_retrieval_calibration(
    pdf_path: str = "data/documents/sample.pdf",
    k_values: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Run empirical retrieval calibration against the specified document.

    Args:
        pdf_path (str): Path to evaluation PDF document.
        k_values (Optional[List[int]]): Top-k retrieval depths to evaluate. Defaults to [3, 5].

    Returns:
        Dict[str, Any]: Structured evaluation results containing per-query metrics,
                        category statistics, separation analysis, and threshold simulations.
    """
    if k_values is None:
        k_values = [3, 5]

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Evaluation PDF not found at: {pdf_path}")

    print("=" * 80)
    print("RETRIEVAL RELEVANCE CALIBRATION & DISTANCE EVALUATION SUITE")
    print("=" * 80)
    print(f"Target Document:        {pdf_path}")
    print(f"Evaluation Depth k:     {k_values}")
    print(f"Total Test Queries:     {len(EVALUATION_QUERIES)}")
    print("Embedding Model:        all-MiniLM-L6-v2 (Real)")
    print("Vector Store:           FAISS IndexFlatL2 (Real)")
    print("Generation LLM:         None (Ollama is not invoked)")
    print("=" * 80)

    # 1. Initialize real embedding generator and vector store
    embedding_generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    vector_store = build_vector_store(pdf_path, embedding_generator=embedding_generator)
    retriever = Retriever(vector_store=vector_store, embedding_generator=embedding_generator)

    max_k = max(k_values)
    results_by_k: Dict[int, List[Dict[str, Any]]] = {k: [] for k in k_values}

    # 2. Execute retrieval for each test query
    for q_idx, item in enumerate(EVALUATION_QUERIES, start=1):
        category = item["category"]
        query = item["query"]
        expected_pages = item["expected_pages"]

        # Retrieve up to max_k candidates
        retrieved_raw = retriever.retrieve(query, top_k=max_k)

        for k in k_values:
            k_results = retrieved_raw[:k]
            chunk_records: List[Dict[str, Any]] = []

            for rank, chunk in enumerate(k_results, start=1):
                doc = chunk.get("document", {})
                distance = chunk.get("distance", float("inf"))
                page_num = doc.get("page_number", -1)
                chunk_id = doc.get("chunk_id", "N/A")
                text_snippet = doc.get("text", "").strip()[:80].replace("\n", " ")

                chunk_records.append({
                    "rank": rank,
                    "source": doc.get("source", "Unknown"),
                    "page_number": page_num,
                    "chunk_id": chunk_id,
                    "preview": text_snippet,
                    "distance": distance,
                })

            distances = [c["distance"] for c in chunk_records]
            top1_distance = distances[0] if distances else None
            topk_avg_distance = sum(distances) / len(distances) if distances else None
            topk_spread = (max(distances) - min(distances)) if distances else None

            # Verify whether retrieved context matches expected target pages
            top1_page = chunk_records[0]["page_number"] if chunk_records else None
            retrieved_pages = [c["page_number"] for c in chunk_records]

            if expected_pages:
                top1_match = top1_page in expected_pages
                topk_match = any(p in expected_pages for p in retrieved_pages)
            else:
                top1_match = None
                topk_match = None

            results_by_k[k].append({
                "query_index": q_idx,
                "category": category,
                "query": query,
                "expected_pages": expected_pages,
                "chunks": chunk_records,
                "top1_distance": top1_distance,
                "topk_avg_distance": topk_avg_distance,
                "topk_spread": topk_spread,
                "top1_match": top1_match,
                "topk_match": topk_match,
            })

    # 3. Compute statistics and print reports for primary k (k=3) and secondary k (k=5)
    evaluation_summary: Dict[str, Any] = {
        "k_evaluations": {},
        "simulation_thresholds": SIMULATION_THRESHOLDS,
    }

    for k in k_values:
        eval_data = results_by_k[k]
        cat_stats = _compute_category_statistics(eval_data)
        separation_gap = _compute_separation_gap(cat_stats)
        threshold_simulations = _simulate_thresholds(eval_data, SIMULATION_THRESHOLDS)

        evaluation_summary["k_evaluations"][k] = {
            "query_results": eval_data,
            "category_statistics": cat_stats,
            "separation_gap": separation_gap,
            "threshold_simulations": threshold_simulations,
        }

        _print_k_report(k, eval_data, cat_stats, separation_gap, threshold_simulations)

    return evaluation_summary


# ---------------------------------------------------------------------------
# Statistics and Metric Computation Helpers
# ---------------------------------------------------------------------------

def _compute_category_statistics(query_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute min, max, mean, and median top-1 distances per query category."""
    distances_by_cat: Dict[str, List[float]] = {}

    for res in query_results:
        cat = res["category"]
        dist = res["top1_distance"]
        if dist is not None:
            distances_by_cat.setdefault(cat, []).append(dist)

    stats: Dict[str, Dict[str, float]] = {}
    for cat, dists in distances_by_cat.items():
        if dists:
            stats[cat] = {
                "count": len(dists),
                "min": min(dists),
                "max": max(dists),
                "mean": sum(dists) / len(dists),
                "median": statistics.median(dists),
            }

    return stats


def _compute_separation_gap(cat_stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    Calculate the distance separation gap between unrelated and answerable queries.
    Gap = min(unrelated top1) - max(answerable top1).
    A positive gap indicates complete separation without overlap.
    """
    ans_stats = cat_stats.get("Answerable")
    unrel_stats = cat_stats.get("Unrelated")

    if not ans_stats or not unrel_stats:
        return {"gap": None, "separated": False, "note": "Insufficient data"}

    ans_max = ans_stats["max"]
    unrel_min = unrel_stats["min"]
    gap = unrel_min - ans_max

    return {
        "answerable_max_top1": ans_max,
        "unrelated_min_top1": unrel_min,
        "gap": gap,
        "is_cleanly_separated": gap > 0,
    }


def _simulate_thresholds(
    query_results: List[Dict[str, Any]],
    thresholds: List[float],
) -> List[Dict[str, Any]]:
    """
    Simulate the effect of various candidate max_distance thresholds.
    For each threshold, reports:
      - Answerable Recall = passed answerable / total answerable
      - Answerable Precision = passed answerable / (passed answerable + passed unrelated)
      - Ambiguous Pass Rate = passed ambiguous / total ambiguous
      - Unrelated Rejection Rate = rejected unrelated / total unrelated
    """
    simulations: List[Dict[str, Any]] = []

    for tau in thresholds:
        ans_pass = 0
        ans_total = 0
        amb_pass = 0
        amb_total = 0
        unrel_pass = 0
        unrel_total = 0

        for r in query_results:
            cat = r["category"]
            dist = r["top1_distance"]
            passes = dist is not None and dist <= tau

            if cat == "Answerable":
                ans_total += 1
                if passes:
                    ans_pass += 1
            elif cat == "Ambiguous/Boundary":
                amb_total += 1
                if passes:
                    amb_pass += 1
            elif cat == "Unrelated":
                unrel_total += 1
                if passes:
                    unrel_pass += 1

        ans_recall = (ans_pass / ans_total * 100) if ans_total else 0.0
        denom = ans_pass + unrel_pass
        ans_precision = (ans_pass / denom * 100) if denom > 0 else 0.0
        amb_pass_rate = (amb_pass / amb_total * 100) if amb_total else 0.0
        unrel_rejection = ((unrel_total - unrel_pass) / unrel_total * 100) if unrel_total else 0.0

        simulations.append({
            "threshold": tau,
            "answerable_passed": f"{ans_pass}/{ans_total}",
            "answerable_recall_pct": ans_recall,
            "answerable_precision_pct": ans_precision,
            "ambiguous_passed": f"{amb_pass}/{amb_total}",
            "ambiguous_pass_rate_pct": amb_pass_rate,
            "unrelated_passed": f"{unrel_pass}/{unrel_total}",
            "unrelated_rejection_pct": unrel_rejection,
        })

    return simulations


# ---------------------------------------------------------------------------
# Formatted Terminal Reporting
# ---------------------------------------------------------------------------

def _print_k_report(
    k: int,
    query_results: List[Dict[str, Any]],
    cat_stats: Dict[str, Dict[str, float]],
    separation_gap: Dict[str, Any],
    simulations: List[Dict[str, Any]],
) -> None:
    """Print readable evaluation report for a given k depth."""
    print("\n" + "#" * 80)
    print(f"EVALUATION RESULTS FOR TOP-k = {k}")
    print("#" * 80)

    # 1. Per-Query Retrieval Breakdown
    print(f"\n1. PER-QUERY RETRIEVAL BREAKDOWN (k = {k}):")
    print("-" * 80)
    print(f"{'#':<3} {'Category':<18} {'Top-1 Dist':<11} {'Avg Dist':<10} {'Spread':<8} {'Target Match':<12} {'Query'}")
    print("-" * 80)

    for r in query_results:
        idx = r["query_index"]
        cat = r["category"]
        top1 = f"{r['top1_distance']:.4f}" if r["top1_distance"] is not None else "N/A"
        avg_d = f"{r['topk_avg_distance']:.4f}" if r["topk_avg_distance"] is not None else "N/A"
        spread = f"{r['topk_spread']:.4f}" if r["topk_spread"] is not None else "N/A"

        if r["top1_match"] is True:
            target_str = "YES (P1)"
        elif r["top1_match"] is False and r["topk_match"] is True:
            target_str = "YES (Top-k)"
        elif r["top1_match"] is False and r["topk_match"] is False:
            target_str = "MISMATCH"
        else:
            target_str = "N/A (No Exp)"

        q_short = (r["query"][:42] + "..") if len(r["query"]) > 44 else r["query"]
        print(f"{idx:<3} {cat:<18} {top1:<11} {avg_d:<10} {spread:<8} {target_str:<12} {q_short}")

    # 2. Detailed Chunk Retrieval Table
    print(f"\n2. RETRIEVED CHUNK METRICS (Rank 1 to {k}):")
    print("-" * 80)
    for r in query_results:
        print(f"\n[Q{r['query_index']}] ({r['category']}) \"{r['query']}\"")
        for chunk in r["chunks"]:
            print(
                f"   Rank {chunk['rank']}: Dist={chunk['distance']:.4f} | "
                f"Page={chunk['page_number']} | ChunkID={chunk['chunk_id']} | "
                f"Text=\"{chunk['preview']}...\""
            )

    # 3. Category Top-1 Statistics
    print(f"\n3. CATEGORY TOP-1 DISTANCE STATISTICS (k = {k}):")
    print("-" * 80)
    print(f"{'Category':<20} {'Count':<7} {'Min':<10} {'Max':<10} {'Mean':<10} {'Median':<10}")
    print("-" * 80)
    for cat, s in cat_stats.items():
        print(
            f"{cat:<20} {s['count']:<7} {s['min']:<10.4f} {s['max']:<10.4f} "
            f"{s['mean']:<10.4f} {s['median']:<10.4f}"
        )

    # 4. Separation Analysis
    print(f"\n4. SEPARATION GAP ANALYSIS (k = {k}):")
    print("-" * 80)
    ans_max = separation_gap.get("answerable_max_top1")
    unrel_min = separation_gap.get("unrelated_min_top1")
    gap = separation_gap.get("gap")
    is_sep = separation_gap.get("is_cleanly_separated")

    if gap is not None:
        print(f"Max Top-1 Distance for Answerable Queries:   {ans_max:.4f}")
        print(f"Min Top-1 Distance for Unrelated Queries:    {unrel_min:.4f}")
        print(f"Distance Separation Gap (Unrelated - Ans):   {gap:+.4f}")
        if is_sep:
            print("Status: [CLEAN SEPARATION] Unrelated distances do not overlap with answerable queries.")
        else:
            print("Status: [OVERLAP DETECTED] An overlap exists between answerable and unrelated queries.")
    print("-" * 80)

    # 5. Candidate Threshold Simulation Table
    print(f"\n5. CANDIDATE THRESHOLD SIMULATIONS (FOR ANALYSIS ONLY - NO MODIFICATIONS):")
    print("-" * 95)
    print(
        f"{'Threshold':<10} {'Ans Passed':<12} {'Recall':<9} {'Precision':<11} "
        f"{'Amb Passed':<12} {'Amb Pass %':<12} {'Unrel Passed':<14} {'Unrel Reject %'}"
    )
    print("-" * 95)
    for sim in simulations:
        print(
            f"{sim['threshold']:<10.2f} {sim['answerable_passed']:<12} "
            f"{sim['answerable_recall_pct']:<8.1f}% {sim['answerable_precision_pct']:<10.1f}% "
            f"{sim['ambiguous_passed']:<12} {sim['ambiguous_pass_rate_pct']:<11.1f}% "
            f"{sim['unrelated_passed']:<14} {sim['unrelated_rejection_pct']:.1f}%"
        )
    print("-" * 95)
    print("Note: The above simulations are purely observational and do not change any production code.")


# ---------------------------------------------------------------------------
# Standalone Execution Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_retrieval_calibration()
