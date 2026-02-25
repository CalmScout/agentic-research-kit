"""Simple evaluation system for multi-agent RAG.

Implements basic evaluation metrics using ground truth pairs from the dataset.
"""

import logging
from typing import Any, cast

import pandas as pd

logger = logging.getLogger(__name__)


async def evaluate_retrieval(
    query_func,
    csv_path: str = "data/claim_matching_dataset.csv",
    test_size: int = 20,
    top_k_values: list[int] | None = None,
) -> dict[str, Any]:
    """Evaluate retrieval performance using ground truth pairs.

    Args:
        query_func: Async query function (e.g., query_with_agents)
        csv_path: Path to CSV dataset
        test_size: Number of test queries
        top_k_values: K values for Precision@K and Recall@K

    Returns:
        Dict with evaluation metrics
    """
    if top_k_values is None:
        top_k_values = [5, 10]
    logger.info(f"Starting evaluation with {test_size} test queries...")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Normalize column names (handle spaces vs underscores)
    # CSV has: "unverified claim", "reviewed claim"
    # Code expects: "unverified_claim", "reviewed_claim"
    column_map = {
        "unverified claim": "unverified_claim",
        "reviewed claim": "reviewed_claim",
    }
    df = df.rename(columns=column_map)

    # Get ground truth pairs (similarity=1)
    ground_truth = df[df["similarity"] == 1].head(test_size)

    logger.info(f"Loaded {len(ground_truth)} ground truth pairs")

    # Metrics storage
    results: dict[str, Any] = {
        "total_queries": len(ground_truth),
        "successful_queries": 0,
        "precision_at_k": {k: [] for k in top_k_values},
        "recall_at_k": {k: [] for k in top_k_values},
        "mrr": [],
    }

    # Run evaluation
    for _idx, row in ground_truth.iterrows():
        unverified_claim = row["unverified_claim"]
        reviewed_claim = row["reviewed_claim"]

        try:
            # Run query
            result = await query_func(unverified_claim)

            # Check if reviewed claim is in sources
            sources = result.get("sources", [])
            source_texts = [s.get("text", s.get("content", "")) for s in sources]

            # Check if reviewed claim is in retrieved documents
            is_retrieved = any(reviewed_claim.lower() in text.lower() for text in source_texts)

            if is_retrieved:
                # Find rank (position in sources)
                rank = None
                for i, text in enumerate(source_texts):
                    if reviewed_claim.lower() in text.lower():
                        rank = i + 1
                        break

                # Calculate metrics
                for k in top_k_values:
                    if rank and rank <= k:
                        results["precision_at_k"][k].append(1.0)
                        results["recall_at_k"][k].append(1.0)
                    else:
                        results["precision_at_k"][k].append(0.0)
                        results["recall_at_k"][k].append(0.0)

                # MRR (Mean Reciprocal Rank)
                if rank:
                    results["mrr"].append(1.0 / rank)
                else:
                    results["mrr"].append(0.0)

                results["successful_queries"] += 1

        except Exception as e:
            logger.error(f"Error evaluating query '{unverified_claim}': {e}")
            # Add zeros for failed queries
            for k in top_k_values:
                results["precision_at_k"][k].append(0.0)
                results["recall_at_k"][k].append(0.0)
            results["mrr"].append(0.0)

    # Calculate final metrics
    metrics: dict[str, Any] = {}
    for k in top_k_values:
        precision = (
            sum(results["precision_at_k"][k]) / len(results["precision_at_k"][k])
            if results["precision_at_k"][k]
            else 0.0
        )
        recall = (
            sum(results["recall_at_k"][k]) / len(results["recall_at_k"][k])
            if results["recall_at_k"][k]
            else 0.0
        )
        metrics[f"precision_at_{k}"] = precision
        metrics[f"recall_at_{k}"] = recall
        metrics[f"f1_at_{k}"] = (
            2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        )

    metrics["mrr"] = sum(results["mrr"]) / len(results["mrr"]) if results["mrr"] else 0.0
    metrics["success_rate"] = float(results["successful_queries"]) / cast(
        int, results["total_queries"]
    )

    # Include raw counts for display
    metrics["total_queries"] = cast(float, results["total_queries"])
    metrics["successful_queries"] = cast(float, results["successful_queries"])

    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total queries: {results['total_queries']}")
    logger.info(f"Successful queries: {results['successful_queries']}")
    logger.info(f"Success rate: {metrics['success_rate']:.2%}")

    for k in top_k_values:
        logger.info(f"\nPrecision@{k}: {metrics[f'precision_at_{k}']:.3f}")
        logger.info(f"Recall@{k}: {metrics[f'recall_at_{k}']:.3f}")
        logger.info(f"F1@{k}: {metrics[f'f1_at_{k}']:.3f}")

    logger.info(f"\nMRR: {metrics['mrr']:.3f}")

    return metrics


def calculate_retrieval_metrics(
    retrieved_docs: list[dict[str, Any]],
    ground_truth_doc_id: str,
    top_k: int = 10,
) -> dict[str, Any]:
    """Calculate retrieval metrics for a single query.

    Args:
        retrieved_docs: List of retrieved documents
        ground_truth_doc_id: ID of the ground truth document
        top_k: Top K for precision calculation

    Returns:
        Dict with precision, recall, and rank
    """
    # Find rank of ground truth
    rank = None
    for i, doc in enumerate(retrieved_docs[:top_k], 1):
        if doc.get("doc_id") == ground_truth_doc_id:
            rank = i
            break

    precision = 1.0 / top_k if rank and rank <= top_k else 0.0
    recall = 1.0 if rank else 0.0

    return {"precision": precision, "recall": recall, "rank": rank}
