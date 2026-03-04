#!/usr/bin/env python
"""CLI interface for Agentic Research Kit (ARK)."""

import click
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Agentic Research Kit (ARK) for Deep Research and Analysis.

    A flexible multi-agent RAG framework with configurable prompts,
    multiple document format support, and hybrid LLM strategy.
    """
    pass


@cli.command()
@click.argument("query")
@click.option("--mode", "-m", type=click.Choice(["naive", "local", "global", "hybrid"]), default="hybrid", help="Retrieval mode for LightRAG")
@click.option("--session", "-s", default="default", help="Session ID for conversation history")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--prompt-template", "-p", default="research", help="Prompt template to use (research, analysis, qa, custom)")
@click.option("--hide-sources", is_flag=True, help="Hide source documents in output")
@click.option("--image", "-i", type=click.Path(exists=True), help="Path to image for multimodal query")
def query(query: str, mode: str, session: str, debug: bool, prompt_template: str, hide_sources: bool, image: str):
    """Query the research system using multi-agent workflow.

    Example:
        ark query "How has bias in biomedical AI evolved over years?" --mode hybrid
        ark query "Analyze this chart" --image ./chart.png
    """
    import asyncio
    import logging

    from src.agents.workflow import query_with_agents
    from src.agents.utils import format_response_for_display
    from src.utils.logger import setup_logging
    from src.utils.observability import setup_observability
    from src.utils.config import get_settings

    # Override log level if debug flag is set
    if debug:
        settings = get_settings()
        settings.log_level = "DEBUG"

    # Configure logging and observability
    setup_logging()
    setup_observability()

    click.echo(f"🔍 Query: {query}")
    if image:
        click.echo(f"🖼️  Image: {image}")
    click.echo(f"🎯 Mode: {mode}")
    click.echo(f"📝 Session: {session}")

    # Run multi-agent workflow
    try:
        # Pass image if provided
        result = asyncio.run(query_with_agents(
            query=query, 
            query_image=image,
            retrieval_mode=mode, 
            debug=debug
        ))

        # Display formatted response
        formatted = format_response_for_display(
            response=result["response"],
            sources=result.get("sources", []),
            show_sources=not hide_sources
        )

        click.echo("\n" + "=" * 60)
        click.echo("💡 RESPONSE")
        click.echo("=" * 60)
        click.echo(formatted)

        # Show additional info if debug
        if debug:
            click.echo(f"\n📊 Metadata:")
            click.echo(f"  Entities: {result.get('entities', [])}")
            click.echo(f"  Retrieved: {result.get('retrieved_count', 0)} docs")
            click.echo(f"  Session: {session}")

    except Exception as e:
        click.echo(f"\n❌ Query failed: {e}", err=True)
        if debug:
            import traceback
            traceback.print_exc()
        raise SystemExit(1)


@cli.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option("--pattern", "-p", default="*.pdf", help="File pattern (e.g., '*.pdf', '*.docx', '**/*.*')")
@click.option("--recursive", "-r", is_flag=True, default=True, help="Search recursively")
@click.option("--max-items", "-n", default=None, type=int, help="Maximum number of items to ingest (for testing)")
@click.option("--working-dir", "-w", default="./rag_storage", help="RAG storage directory")
def ingest_dir(directory: str, pattern: str, recursive: bool, max_items: int | None, working_dir: str):
    """Ingest documents from directory (PDF, DOCX, HTML, TXT).

    Automatically detects file format and processes accordingly.

    Examples:
        # Ingest all PDFs recursively
        ark ingest-dir ./papers --pattern "*.pdf"

        # Ingest all Word documents
        ark ingest-dir ./reports --pattern "*.docx"

        # Ingest all supported formats
        ark ingest-dir ./documents --pattern "*.*" --recursive

        # Test with limited files
        ark ingest-dir ./papers --pattern "*.pdf" --max-items 10
    """
    click.echo(f"📁 Ingesting documents from {directory}")
    click.echo(f"🔍 Pattern: {pattern}")
    click.echo(f"🔄 Recursive: {recursive}")
    if max_items:
        click.echo(f"🧪 Test mode: Processing only first {max_items} files")
    click.echo(f"💾 Working directory: {working_dir}")

    # Import here to avoid circular imports
    from src.data_ingestion.universal_pipeline import run_sync_directory_ingestion

    try:
        # Run universal ingestion pipeline
        stats = run_sync_directory_ingestion(
            dir_path=directory,
            pattern=pattern,
            working_dir=working_dir,
            recursive=recursive,
            max_items=max_items
        )

        # Display final statistics
        click.echo("\n" + "=" * 60)
        click.echo("✓ INGESTION COMPLETE!")
        click.echo("=" * 60)
        click.echo(f"Total files: {stats['total_files']}")
        click.echo(f"Successful: {stats['successful_files']}")
        click.echo(f"Failed: {stats['failed_files']}")
        click.echo(f"Total chunks: {stats['total_chunks']}")
        click.echo(f"Items ingested: {stats['ingest_stats'].get('total_items', 0)}")

        if stats.get('failed_files', 0) > 0:
            click.echo("\n⚠️  Failed files:")
            for fail in stats.get('failed_details', [])[:5]:
                click.echo(f"  - {fail['file']}: {fail['error']}")
            if len(stats.get('failed_details', [])) > 5:
                click.echo(f"  ... and {len(stats['failed_details']) - 5} more")

    except FileNotFoundError as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        raise SystemExit(1)
    except ValueError as e:
        click.echo(f"\n❌ Validation error: {e}", err=True)
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"\n❌ Ingestion failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise SystemExit(1)


@cli.command()
@click.option("--output", "-o", default="reports/evaluation.json", help="Output file for results")
@click.option("--test-size", "-n", default=20, help="Number of test queries")
@click.option("--metrics", "-m", multiple=True,
              type=click.Choice(["simple", "ragas", "all"]),
              default=["simple"], help="Evaluation type to run")
@click.option("--ragas-metrics", multiple=True,
              type=click.Choice(["faithfulness", "answer_relevancy",
                                "context_precision", "context_recall"]),
              default=["faithfulness", "answer_relevancy"],
              help="RAGAS metrics to run (when --metrics ragas or all)")
@click.option("--llm", type=click.Choice(["deepseek", "openai", "local"]),
              default="deepseek", help="LLM for RAGAS evaluation")
def evaluate(output: str, test_size: int, metrics: tuple, ragas_metrics: tuple, llm: str):
    """Run evaluation metrics.

    Calculates:
    - Simple metrics: Retrieval metrics (Precision@K, Recall@K, MRR)
    - RAGAS metrics: Faithfulness, Answer Relevancy (LLM-judged quality)

    Examples:
        # Simple metrics only (default)
        ark evaluate -n 20

        # RAGAS metrics only
        ark evaluate --metrics ragas -n 50

        # All metrics
        ark evaluate --metrics all -n 50

        # Use OpenAI for RAGAS (most reliable)
        ark evaluate --metrics ragas --llm openai -n 30

        # Specific RAGAS metrics
        ark evaluate --metrics ragas --ragas-metrics faithfulness -n 20
    """
    import asyncio
    import json
    from pathlib import Path
    from dotenv import load_dotenv
    from src.evaluation.simple_eval import evaluate_retrieval
    from src.agents.workflow import query_with_agents

    # Load environment variables
    load_dotenv()

    click.echo(f"📊 Running evaluation with {test_size} test queries")
    click.echo(f"📁 Output: {output}")

    # Create output directory if needed
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}

    # Run simple evaluation
    if "simple" in metrics or "all" in metrics:
        click.echo("Running simple evaluation (Precision@K, Recall@K, MRR)...")

        async def run_simple():
            return await evaluate_retrieval(
                query_func=query_with_agents,
                test_size=test_size,
                top_k_values=[5, 10]
            )

        results["simple"] = asyncio.run(run_simple())

    # Run RAGAS evaluation
    if "ragas" in metrics or "all" in metrics:
        click.echo(f"Running RAGAS evaluation: {', '.join(ragas_metrics)}")
        click.echo(f"LLM Provider: {llm}")

        from src.evaluation.ragas_evaluator import create_evaluator_from_settings

        async def run_ragas():
            evaluator = create_evaluator_from_settings(llm_provider=llm)
            return await evaluator.evaluate_workflow(
                query_func=query_with_agents,
                test_size=test_size,
                metrics=list(ragas_metrics),
            )

        results["ragas"] = asyncio.run(run_ragas())

    # Save results to JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Display results
    _display_evaluation_results(results)

    click.echo(f"\n✅ Results saved to: {output_path}")


def _display_evaluation_results(results: dict):
    """Display evaluation results in a formatted way.

    Args:
        results: Dict with 'simple' and/or 'ragas' results
    """
    click.echo("\n" + "=" * 60)
    click.echo("EVALUATION RESULTS")
    click.echo("=" * 60)

    # Display simple metrics
    if "simple" in results:
        simple = results["simple"]
        click.echo(f"\n📈 Simple Metrics (Retrieval Quality)")
        click.echo(f"Total queries: {simple.get('total_queries', 'N/A')}")
        click.echo(f"Successful queries: {simple.get('successful_queries', 'N/A')}")
        click.echo(f"Success rate: {simple.get('success_rate', 0):.2%}")

        for k in [5, 10]:
            if f"precision_at_{k}" in simple:
                click.echo(f"\n  Precision@{k}: {simple[f'precision_at_{k}']:.3f}")
                click.echo(f"  Recall@{k}: {simple[f'recall_at_{k}']:.3f}")
                click.echo(f"  F1@{k}: {simple[f'f1_at_{k}']:.3f}")

        if "mrr" in simple:
            click.echo(f"\n  MRR: {simple['mrr']:.3f}")

    # Display RAGAS metrics
    if "ragas" in results:
        ragas = results["ragas"]
        click.echo(f"\n🤖 RAGAS Metrics (LLM-Judged Quality)")

        if "config" in ragas:
            click.echo(f"  LLM Provider: {ragas['config'].get('metrics', 'N/A')}")
            click.echo(f"  Test Size: {ragas['config'].get('test_size', 'N/A')}")

        if "metrics" in ragas:
            for metric_name, metric_data in ragas["metrics"].items():
                score = metric_data.get("score", 0)
                std = metric_data.get("std", 0)
                count = metric_data.get("count", 0)

                click.echo(f"\n  {metric_name.replace('_', ' ').title()}:")
                click.echo(f"    Score: {score:.3f} (±{std:.3f})")
                click.echo(f"    Samples: {count}")

                # Add interpretation
                if score >= 0.8:
                    interpretation = "Excellent"
                elif score >= 0.6:
                    interpretation = "Good"
                elif score >= 0.4:
                    interpretation = "Fair"
                else:
                    interpretation = "Needs Improvement"
                click.echo(f"    Interpretation: {interpretation}")


@cli.command()
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind to")
@click.option("--port", "-p", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host: str, port: int, reload: bool):
    """Start API server.

    Example:
        ark serve --host 0.0.0.0 --port 8000
    """
    import uvicorn

    try:
        from src.api.main import app
    except ImportError:
        click.echo("\n❌ API module not found. Make sure src/api/main.py exists.", err=True)
        click.echo("   Run: uv run ark serve")
        raise SystemExit(1)

    click.echo(f"🚀 Starting API server on {host}:{port}")
    if reload:
        click.echo("   Auto-reload enabled")

    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload
        )
    except Exception as e:
        click.echo(f"\n❌ Server failed to start: {e}", err=True)
        raise SystemExit(1)


@cli.command()
def gateway():
    """Start the research gateway (Telegram, etc.)."""
    import asyncio
    from src.utils.config import get_settings
    from src.agents.channels.manager import ChannelManager
    from src.agents.channels.telegram import TelegramChannel
    
    settings = get_settings()
    manager = ChannelManager()
    
    if settings.telegram_enabled:
        click.echo("🔌 Enabling Telegram channel...")
        manager.register_channel(TelegramChannel())
    
    if not manager._channels:
        click.echo("⚠️  No channels enabled. Enable at least one channel in .env (e.g., TELEGRAM_ENABLED=true).")
        return

    click.echo("🚀 Starting Research Gateway...")
    try:
        asyncio.run(manager.start())
    except KeyboardInterrupt:
        click.echo("\n🛑 Stopping Gateway...")
        asyncio.run(manager.stop())


@cli.command()
def status():
    """Show system status and configuration."""
    click.echo("📋 System Status")
    click.echo("=" * 50)

    # TODO: Implement status check
    # 1. Check .env file exists
    # 2. Check Ollama connection
    # 3. Check dataset exists
    # 4. Check RAG storage
    # 5. Display configuration summary

    click.echo("\n⚠️  Status functionality not yet implemented.")


if __name__ == "__main__":
    cli()
