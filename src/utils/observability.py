"""Observability and distributed tracing configuration using Phoenix (Arize AI).

Integrates OpenTelemetry to capture traces from agents, LLMs, and RAG components.
"""

from typing import Any

from src.utils.config import get_settings
from src.utils.logger import logger

# Global flags to track initialization state
_observability_initialized = False
_tracer_provider: Any = None


def setup_observability() -> None:
    """Initialize Phoenix observability and OpenTelemetry instrumentation.

    This function sets up distributed tracing based on the application settings.
    It should be called at the application entry point (e.g., main.py).
    """
    global _observability_initialized, _tracer_provider

    if _observability_initialized:
        return

    settings = get_settings()

    if not settings.phoenix_enabled:
        logger.debug("Phoenix observability disabled (PHOENIX_ENABLED=false)")
        return

    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from opentelemetry import trace
        from phoenix.otel import register as phoenix_register

        logger.info(
            f"Initializing Phoenix observability for project '{settings.phoenix_project_name}'..."
        )

        # Register Phoenix with OpenTelemetry
        _tracer_provider = phoenix_register(
            project_name=settings.phoenix_project_name,
            endpoint=settings.phoenix_collector_endpoint,
        )

        # Set as global tracer provider
        trace.set_tracer_provider(_tracer_provider)

        # Instrument LangChain (which LangGraph uses internally)
        LangChainInstrumentor().instrument(tracer_provider=_tracer_provider)

        _observability_initialized = True
        logger.info("✓ Phoenix + OpenTelemetry instrumentation enabled")
        logger.info(f"  Traces: {settings.phoenix_collector_endpoint.replace('/v1/traces', '')}")

    except ImportError as e:
        logger.warning(f"Phoenix dependencies not installed: {e}")
        logger.warning(
            "  Install with: uv add arize-phoenix openinference-instrumentation-langchain"
        )
    except Exception as e:
        logger.error(f"Failed to initialize Phoenix: {e}")
        import traceback

        logger.debug(traceback.format_exc())
