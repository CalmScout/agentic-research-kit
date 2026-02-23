"""Tests for evaluation module."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
from io import StringIO

from src.evaluation.simple_eval import (
    evaluate_retrieval,
    calculate_retrieval_metrics
)


@pytest.mark.asyncio
async def test_evaluate_retrieval_basic():
    """Test basic retrieval evaluation."""
    # Mock query function
    async def mock_query_func(query):
        return {
            "sources": [
                {"text": "Climate change is real", "content": "Climate change is real"},
                {"text": "Temperature is rising", "content": "Temperature is rising"}
            ]
        }

    # Mock CSV data
    mock_csv_data = StringIO("""unverified_claim,reviewed_claim,similarity
"Is climate change real?","Climate change is real",1
"Is temperature rising?","Temperature is rising",1""")

    with patch("pandas.read_csv", return_value=pd.read_csv(mock_csv_data)):
        results = await evaluate_retrieval(
            query_func=mock_query_func,
            test_size=2,
            top_k_values=[5]
        )

        # Verify results structure (metrics only, not total_queries)
        assert "precision_at_5" in results
        assert "recall_at_5" in results
        assert "f1_at_5" in results
        assert "mrr" in results
        assert "success_rate" in results
        assert results["success_rate"] == 1.0  # Both queries should succeed


def test_calculate_retrieval_metrics():
    """Test single query retrieval metrics calculation."""
    retrieved_docs = [
        {"doc_id": "doc1", "text": "First result"},
        {"doc_id": "doc2", "text": "Second result"},
        {"doc_id": "doc3", "text": "Third result"},
    ]

    metrics = calculate_retrieval_metrics(
        retrieved_docs=retrieved_docs,
        ground_truth_doc_id="doc2",
        top_k=10
    )

    # Verify metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "rank" in metrics
    assert metrics["rank"] == 2
    assert metrics["precision"] == 0.1  # 1/10
    assert metrics["recall"] == 1.0  # Found in top 10


def test_calculate_retrieval_metrics_not_found():
    """Test metrics when ground truth is not retrieved."""
    retrieved_docs = [
        {"doc_id": "doc1", "text": "First result"},
        {"doc_id": "doc3", "text": "Third result"},
    ]

    metrics = calculate_retrieval_metrics(
        retrieved_docs=retrieved_docs,
        ground_truth_doc_id="doc2",
        top_k=10
    )

    # Ground truth not found
    assert metrics["rank"] is None
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


@pytest.mark.asyncio
async def test_evaluate_retrieval_with_failures():
    """Test evaluation handles query failures gracefully."""
    # Mock query function that fails
    async def failing_query_func(query):
        raise Exception("Query failed")

    mock_csv_data = StringIO("""unverified_claim,reviewed_claim,similarity
"Test query","Test result",1""")

    with patch("pandas.read_csv", return_value=pd.read_csv(mock_csv_data)):
        results = await evaluate_retrieval(
            query_func=failing_query_func,
            test_size=1,
            top_k_values=[5]
        )

        # Should handle failures gracefully
        assert "precision_at_5" in results
        assert "recall_at_5" in results
        assert "mrr" in results
        assert results["mrr"] == 0.0  # Failed queries get MRR of 0
        assert results["success_rate"] == 0.0  # Query failed
