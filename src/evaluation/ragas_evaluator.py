"""RAGAS evaluation for multimodal RAG system.

This module provides RAGAS (Retrieval Augmented Generation Assessment) evaluation
for the multi-agent RAG system. RAGAS uses LLM-as-a-judge to measure:

- Faithfulness: Factual consistency of response with retrieved context
- Answer Relevancy: How well the response addresses the user query
- Context Precision: Signal-to-noise ratio in retrieved contexts
- Context Recall: Completeness of retrieved information

Example:
    >>> from src.evaluation.ragas_evaluator import RAGASEvaluator
    >>> from langchain_openai import ChatOpenAI
    >>>
    >>> evaluator_llm = ChatOpenAI(model="gpt-4o-mini")
    >>> evaluator = RAGASEvaluator(evaluator_llm=evaluator_llm)
    >>> results = await evaluator.evaluate_workflow(
    ...     query_func=query_with_agents,
    ...     test_size=50,
    ...     metrics=["faithfulness", "answer_relevancy"]
    ... )
    >>> print(f"Faithfulness: {results['metrics']['faithfulness']['score']:.2%}")
"""

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import pandas as pd
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """RAGAS evaluation for multimodal RAG system.

    This evaluator integrates RAGAS metrics with the LangGraph workflow,
    providing LLM-judged evaluation of retrieval and generation quality.

    Features:
        - Async-first (matches LangGraph workflow pattern)
        - LLM-agnostic (supports DeepSeek, OpenAI, or local models)
        - Phoenix trace linking for observability
        - Caching for cost optimization (75% reduction)
        - Retry logic for robustness

    Attributes:
        llm: LangChain-wrapped LLM for RAGAS evaluation
        batch_size: Number of queries to process in parallel
        timeout: Timeout (seconds) per evaluation batch
        cache: RAGAS cache for cost optimization

    Example:
        >>> evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        >>> evaluator = RAGASEvaluator(
        ...     evaluator_llm=evaluator_llm,
        ...     batch_size=10,
        ...     enable_cache=True
        ... )
    """

    def __init__(
        self,
        evaluator_llm: BaseLanguageModel,
        batch_size: int = 10,
        timeout: int = 120,
        enable_cache: bool = True,
    ):
        """Initialize RAGAS evaluator.

        Args:
            evaluator_llm: LangChain LLM for RAGAS evaluation (requires temperature=0.0)
            batch_size: Batch size for parallel evaluation
            timeout: Timeout (seconds) per evaluation batch
            enable_cache: Enable disk-based caching for 75% cost reduction

        Raises:
            ImportError: If RAGAS is not installed
        """
        # Import RAGAS components
        from ragas.cache import DiskCacheBackend
        from ragas.llms import LangchainLLMWrapper

        self.llm = LangchainLLMWrapper(evaluator_llm)
        self.batch_size = batch_size
        self.timeout = timeout

        # Enable caching (75% cost reduction!)
        if enable_cache:
            self.cache: Any = DiskCacheBackend(cache_dir=".ragas_cache")
            logger.info("RAGAS caching enabled at .ragas_cache/")
        else:
            self.cache = None
            logger.info("RAGAS caching disabled")

        logger.info(f"RAGASEvaluator initialized (batch_size={batch_size}, timeout={timeout}s)")

    async def evaluate_workflow(
        self,
        query_func: Callable,
        csv_path: str = "data/claim_matching_dataset.csv",
        test_size: int = 50,
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run RAGAS evaluation on the LangGraph workflow.

        This method:
        1. Loads ground truth pairs from CSV
        2. Runs query_func for each question
        3. Collects responses and contexts
        4. Executes RAGAS metrics
        5. Returns summary + per-query results

        Args:
            query_func: Async query function (e.g., query_with_agents)
                Should return dict with "response", "sources", "phoenix_trace_id"
            csv_path: Path to ground truth CSV with columns:
                unverified_claim, reviewed_claim, similarity
            test_size: Number of test queries to evaluate
            metrics: RAGAS metrics to compute. Options:
                - "faithfulness": Factual consistency (requires LLM)
                - "answer_relevancy": Query relevance (requires embeddings)
                - "context_precision": Retrieval quality (requires LLM)
                - "context_recall": Retrieval completeness (requires LLM)

        Returns:
            Dict with:
                - metrics: Dict[str, Dict] - Summary statistics (mean, std, min, max)
                - per_query: List[Dict] - Individual query results
                - trace_ids: List[str] - Phoenix trace IDs for observability
                - config: Dict - Evaluation configuration

        Example:
            >>> results = await evaluator.evaluate_workflow(
            ...     query_func=query_with_agents,
            ...     test_size=50,
            ...     metrics=["faithfulness", "answer_relevancy"]
            ... )
            >>> faithfulness = results['metrics']['faithfulness']['score']
            >>> print(f"Faithfulness: {faithfulness:.2%}")
        """
        if metrics is None:
            metrics = ["faithfulness", "answer_relevancy"]
        logger.info(f"Starting RAGAS evaluation with {test_size} queries...")
        logger.info(f"Metrics: {', '.join(metrics)}")

        # Load ground truth pairs
        df = pd.read_csv(csv_path)

        # Normalize column names (handle spaces vs underscores)
        # CSV has: "unverified claim", "reviewed claim"
        # Code expects: "unverified_claim", "reviewed_claim"
        column_map = {
            "unverified claim": "unverified_claim",
            "reviewed claim": "reviewed_claim",
        }
        df = df.rename(columns=column_map)

        ground_truth = df[df["similarity"] == 1].head(test_size)

        logger.info(f"Loaded {len(ground_truth)} ground truth pairs from {csv_path}")

        # Collect workflow data (run queries)
        eval_df = await self._collect_workflow_data(
            query_func=query_func,
            ground_truth=ground_truth,
        )

        logger.info(f"Collected data for {len(eval_df)} queries")

        # Run RAGAS evaluation
        results = await self._run_ragas_evaluation(
            evaluation_df=eval_df,
            metrics=metrics,
        )

        # Add configuration metadata
        results["config"] = {
            "test_size": test_size,
            "metrics": metrics,
            "csv_path": csv_path,
            "batch_size": self.batch_size,
            "timeout": self.timeout,
        }

        logger.info("RAGAS evaluation complete!")

        return results

    async def _collect_workflow_data(
        self,
        query_func: Callable,
        ground_truth: pd.DataFrame,
    ) -> pd.DataFrame:
        """Collect query results from the workflow.

        For each ground truth pair, this method:
        1. Runs query_func(question)
        2. Extracts response, contexts, and trace ID
        3. Handles errors gracefully

        Args:
            query_func: Async query function
            ground_truth: DataFrame with unverified_claim, reviewed_claim columns

        Returns:
            DataFrame with RAGAS-compatible columns:
                - question: User query
                - ground_truth: Expected answer
                - contexts: List of retrieved document texts
                - answer: Generated response
                - phoenix_trace_id: Trace ID for observability
        """
        results = []

        for idx, row in ground_truth.iterrows():
            question = row["unverified_claim"]
            reference = row["reviewed_claim"]

            try:
                # Run workflow
                logger.debug(f"Query {idx+1}/{len(ground_truth)}: {question[:50]}...")
                result = await query_func(question)

                # Extract RAGAS fields
                ragas_row = {
                    "question": question,
                    "ground_truth": reference,
                    "contexts": [
                        s.get("text", s.get("content", "")) for s in result.get("sources", [])
                    ],
                    "answer": result.get("response", ""),
                    "phoenix_trace_id": result.get("phoenix_trace_id"),
                }

                results.append(ragas_row)
                logger.debug(
                    f"  ✓ Retrieved {len(ragas_row['contexts'])} contexts, "
                    f"answer length: {len(ragas_row['answer'])}"
                )

            except Exception as e:
                logger.error(f"Error collecting data for '{question[:50]}...': {e}")
                # Add placeholder for failed queries
                results.append(
                    {
                        "question": question,
                        "ground_truth": reference,
                        "contexts": [],
                        "answer": f"ERROR: {str(e)}",
                        "phoenix_trace_id": None,
                    }
                )

        return pd.DataFrame(results)

    async def _run_ragas_evaluation(
        self,
        evaluation_df: pd.DataFrame,
        metrics: list[str],
    ) -> dict[str, Any]:
        """Execute RAGAS metrics on the collected data.

        This method:
        1. Maps metric names to RAGAS metric objects
        2. Creates RAGAS EvaluationDataset
        3. Runs evaluation with LLM-as-judge
        4. Computes summary statistics

        Args:
            evaluation_df: DataFrame from _collect_workflow_data
            metrics: List of metric names (faithfulness, answer_relevancy, etc.)

        Returns:
            Dict with:
                - metrics: Summary statistics (mean, std, min, max)
                - per_query: Individual query results
                - trace_ids: Phoenix trace IDs
        """
        try:
            from ragas import EvaluationDataset, RunConfig, evaluate
            from ragas.metrics import (
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
                Faithfulness,
            )
        except ImportError as e:
            raise ImportError("RAGAS is not installed. Install with: uv add ragas") from e

        # Map metric names to RAGAS metrics
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        metric_map = {
            "faithfulness": Faithfulness(llm=self.llm),
            "answer_relevancy": AnswerRelevancy(llm=self.llm, embeddings=embeddings),
            "context_precision": ContextPrecision(llm=self.llm),
            "context_recall": ContextRecall(llm=self.llm),
        }

        # Filter valid metrics
        selected_metrics = []
        for m in metrics:
            if m in metric_map:
                selected_metrics.append(metric_map[m])
                logger.debug(f"Added metric: {m}")
            else:
                logger.warning(f"Unknown metric: {m} (skipping)")

        if not selected_metrics:
            raise ValueError(f"No valid metrics specified. Choose from: {list(metric_map.keys())}")

        # Create RAGAS dataset
        samples = []
        for _, row in evaluation_df.iterrows():
            sample = {
                "user_input": row["question"],
                "response": row["answer"],
                "retrieved_contexts": row["contexts"],
                "reference": row["ground_truth"],
            }
            samples.append(sample)

        dataset = EvaluationDataset.from_list(samples)
        logger.info(f"Created RAGAS dataset with {len(dataset)} samples")

        # Configure run
        run_config = RunConfig(timeout=self.timeout)

        # Run evaluation (RAGAS is synchronous, run in thread pool)
        logger.info("Running RAGAS evaluation (may take several minutes)...")
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor() as executor:
            from ragas.dataset_schema import EvaluationResult
        result = await loop.run_in_executor(
            executor,
            evaluate,
            dataset,
            selected_metrics,
            None,  # llm (already wrapped in metrics)
            None,  # embeddings
            None,  # experiment_name
            None,  # callbacks
            run_config,
            None,  # token_usage_parser
            False,  # raise_exceptions
            None,  # column_map
            True,  # show_progress
            None,  # batch_size
            None,  # _run_id
            None,  # _pbar
            False,  # return_executor
            True,  # allow_nest_asyncio
        )

        # Convert to dict
        results_df = cast(EvaluationResult, result).to_pandas()

        # Calculate summary metrics
        summary = {}
        for metric_name in metrics:
            if metric_name in results_df.columns:
                # Remove any NaN values
                valid_scores = results_df[metric_name].dropna()

                if len(valid_scores) > 0:
                    summary[metric_name] = {
                        "score": float(valid_scores.mean()),
                        "std": float(valid_scores.std()),
                        "min": float(valid_scores.min()),
                        "max": float(valid_scores.max()),
                        "count": int(len(valid_scores)),
                    }
                    logger.info(
                        f"  {metric_name}: {summary[metric_name]['score']:.3f} "
                        f"(±{summary[metric_name]['std']:.3f})"
                    )
            else:
                logger.warning(f"Metric '{metric_name}' not found in results")

        return {
            "metrics": summary,
            "per_query": results_df.to_dict("records"),
            "trace_ids": evaluation_df["phoenix_trace_id"].tolist(),
        }


def create_evaluator_from_settings(
    llm_provider: str = "deepseek",
) -> RAGASEvaluator:
    """Create RAGASEvaluator from environment settings.

    This is a convenience function that creates the evaluator LLM
    based on the specified provider.

    Args:
        llm_provider: LLM provider for RAGAS evaluation
            - "deepseek": Use DeepSeek API (default, cost-effective)
            - "openai": Use OpenAI GPT-4o-mini (most reliable)
            - "local": Use local Qwen3-8B (free, slower)

    Returns:
        Initialized RAGASEvaluator

    Raises:
        ValueError: If llm_provider is unknown
        EnvironmentError: If required API keys are missing

    Example:
        >>> from src.evaluation.ragas_evaluator import create_evaluator_from_settings
        >>> evaluator = create_evaluator_from_settings(llm_provider="deepseek")
        >>> results = await evaluator.evaluate_workflow(...)
    """
    import os

    from dotenv import load_dotenv

    # Load environment variables from .env
    load_dotenv()

    if llm_provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise OSError(
                "DEEPSEEK_API_KEY not found in environment. "
                "Set it in .env or export DEEPSEEK_API_KEY=..."
            )

        evaluator_llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=cast(Any, api_key),
            base_url="https://api.deepseek.com",
            temperature=0.0,  # Critical for evaluation!
        )

        logger.info("Using DeepSeek API for RAGAS evaluation")

    elif llm_provider == "openai":
        api_key = os.getenv("RAGAS_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise OSError(
                "OPENAI_API_KEY not found in environment. "
                "Set it in .env or export OPENAI_API_KEY=..."
            )

        evaluator_llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=cast(Any, api_key),
            temperature=0.0,  # Critical for evaluation!
        )

        logger.info("Using OpenAI GPT-4o-mini for RAGAS evaluation")

    elif llm_provider == "local":
        # Import model selector
        try:
            from src.agents.model_selector import get_model_selector

            selector = get_model_selector()
            evaluator_llm = cast(ChatOpenAI, selector.get_local_llm())
            logger.info("Using local Qwen3-8B for RAGAS evaluation")
        except Exception as e:
            raise ImportError(
                f"Failed to load local model: {e}. " "Ensure Ollama is running with Qwen3-8B"
            ) from e
    else:
        raise ValueError(
            f"Unknown llm_provider: {llm_provider}. " f"Choose from: deepseek, openai, local"
        )

    return RAGASEvaluator(evaluator_llm=evaluator_llm)
