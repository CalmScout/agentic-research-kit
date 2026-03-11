import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import InstructorBaseRagasLLM

from src.evaluation.ragas_evaluator import RAGASEvaluator, create_evaluator_from_settings

# Mock HuggingFaceEmbeddings to be recognized as a valid Ragas embedding
class MockRagasEmbedding(BaseRagasEmbedding):
    def embed_query(self, text: str) -> list[float]: return [0.0]
    def embed_documents(self, texts: list[str]) -> list[list[float]]: return [[0.0]]
    def embed_text(self, text: str) -> list[float]: return [0.0]
    async def aembed_query(self, text: str) -> list[float]: return [0.0]
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]: return [[0.0]]
    async def aembed_text(self, text: str) -> list[float]: return [0.0]

mock_hf_module = MagicMock()
mock_hf_module.HuggingFaceEmbeddings = MagicMock(return_value=MockRagasEmbedding())
sys.modules["ragas.embeddings"] = mock_hf_module

@pytest.fixture
def mock_llm():
    class MockInstructorLLM(InstructorBaseRagasLLM):
        def generate(self, *args, **kwargs): return MagicMock()
        async def agenerate(self, *args, **kwargs): return MagicMock()
    
    return MockInstructorLLM()

@pytest.fixture
def evaluator(mock_llm):
    # Mock RAGAS components during init
    with patch("ragas.cache.DiskCacheBackend"):
        return RAGASEvaluator(evaluator_llm=mock_llm, enable_cache=False)

@pytest.mark.asyncio
async def test_collect_workflow_data(evaluator):
    async def mock_query_func(query):
        return {
            "response": f"Response to {query}",
            "sources": [{"text": "Source 1"}, {"text": "Source 2"}],
            "phoenix_trace_id": "trace-123"
        }

    ground_truth = pd.DataFrame({
        "unverified_claim": ["q1"],
        "reviewed_claim": ["a1"]
    })

    df = await evaluator._collect_workflow_data(mock_query_func, ground_truth)

    assert len(df) == 1
    assert df["question"].iloc[0] == "q1"
    assert df["answer"].iloc[0] == "Response to q1"
    assert len(df["contexts"].iloc[0]) == 2
    assert df["phoenix_trace_id"].iloc[0] == "trace-123"

@pytest.mark.asyncio
async def test_collect_workflow_data_failure(evaluator):
    async def mock_query_func(query):
        raise Exception("Query failed")

    ground_truth = pd.DataFrame({
        "unverified_claim": ["q1"],
        "reviewed_claim": ["a1"]
    })

    df = await evaluator._collect_workflow_data(mock_query_func, ground_truth)

    assert len(df) == 1
    assert "ERROR" in df["answer"].iloc[0]
    assert len(df["contexts"].iloc[0]) == 0

@pytest.mark.asyncio
@patch("ragas.evaluate")
@patch("ragas.EvaluationDataset")
async def test_run_ragas_evaluation(mock_dataset, mock_evaluate, evaluator):
    # Mock result from ragas.evaluate
    mock_result = MagicMock()
    mock_result.to_pandas.return_value = pd.DataFrame({
        "faithfulness": [0.8, 0.9],
        "answer_relevancy": [0.7, 0.8]
    })

    eval_df = pd.DataFrame({
        "question": ["q1", "q2"],
        "answer": ["a1", "a2"],
        "contexts": [["c1"], ["c2"]],
        "ground_truth": ["gt1", "gt2"],
        "phoenix_trace_id": ["t1", "t2"]
    })

    # Mock loop.run_in_executor to return our mock_result directly
    with patch("asyncio.get_event_loop") as mock_loop:
        mock_loop_instance = MagicMock()
        mock_loop_instance.run_in_executor = AsyncMock(return_value=mock_result)
        mock_loop.return_value = mock_loop_instance

        # HuggingFaceEmbeddings is now mocked via sys.modules
        results = await evaluator._run_ragas_evaluation(eval_df, metrics=["faithfulness", "answer_relevancy"])

    assert "faithfulness" in results["metrics"]
    assert results["metrics"]["faithfulness"]["score"] == pytest.approx(0.85)
    assert len(results["per_query"]) == 2
    assert results["trace_ids"] == ["t1", "t2"]

@pytest.mark.asyncio
@patch.object(RAGASEvaluator, "_collect_workflow_data")
@patch.object(RAGASEvaluator, "_run_ragas_evaluation")
async def test_evaluate_workflow(mock_run, mock_collect, evaluator):
    mock_collect.return_value = pd.DataFrame()
    mock_run.return_value = {"metrics": {"m1": {"score": 0.9}}, "per_query": [], "trace_ids": []}

    # Mock pd.read_csv to return a valid DF
    mock_df = pd.DataFrame({
        "unverified claim": ["q1"],
        "reviewed claim": ["a1"],
        "similarity": [1]
    })

    with patch("pandas.read_csv", return_value=mock_df):
        results = await evaluator.evaluate_workflow(
            query_func=AsyncMock(),
            csv_path="dummy.csv",
            test_size=1
        )

    assert results["metrics"]["m1"]["score"] == 0.9
    assert results["config"]["test_size"] == 1

def test_create_evaluator_from_settings_openai():
    # Targeted mock to avoid side effects on other env vars (like RAGAS_APP_DIR)
    def mock_getenv(key, default=None):
        if key in ["OPENAI_API_KEY", "RAGAS_OPENAI_API_KEY"]:
            return "fake-key"
        return None

    with patch("os.getenv", side_effect=mock_getenv), \
         patch("src.evaluation.ragas_evaluator.ChatOpenAI"), \
         patch("src.evaluation.ragas_evaluator.RAGASEvaluator"), \
         patch("openai.OpenAI"), \
         patch("ragas.llms.llm_factory"):

        evaluator = create_evaluator_from_settings(llm_provider="openai")
        assert evaluator is not None
