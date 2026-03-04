import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from src.agents.workflow import query_with_agents
from src.evaluation.ragas_evaluator import create_evaluator_from_settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Run RAGAS evaluation on the current ARK workflow."""
    logger.info("Starting ARK RAGAS Evaluation...")

    try:
        # 1. Initialize Evaluator (using DeepSeek)
        evaluator = create_evaluator_from_settings(llm_provider="deepseek")

        # 2. Define metrics to evaluate
        metrics = ["faithfulness", "answer_relevancy", "context_precision"]

        # 3. Run evaluation
        results = await evaluator.evaluate_workflow(
            query_func=query_with_agents,
            csv_path="data/research_golden_dataset.csv",
            test_size=10,
            metrics=metrics
        )

        # 4. Save results
        output_dir = Path("reports/evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"ragas_eval_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Evaluation complete! Results saved to {output_file}")

        # 5. Print summary
        print("\n" + "="*50)
        print("RAGAS EVALUATION SUMMARY")
        print("="*50)
        for metric, stats in results["metrics"].items():
            print(f"{metric.replace('_', ' ').title()}: {stats['score']:.2%}")
        print("="*50)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
