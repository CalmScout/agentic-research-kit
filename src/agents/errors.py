"""Structured error classes for multi-agent RAG system.

Provides context-rich error handling with agent-specific information.
Complements retry logic with tenacity for automatic error recovery.

Example:
    >>> try:
    ...     await enhanced_retriever_agent(state)
    >>> except RetrievalError as e:
    ...     logger.error(f"Retrieval failed in {e.agent}: {e.message}")
    ...     logger.debug(f"Context: {e.context}")
"""


class AgentError(Exception):
    """Base class for agent errors with context.

    Attributes:
        message: Error message
        agent: Agent name where error occurred
        context: Additional context dictionary
    """

    def __init__(self, message: str, agent: str = "unknown", context: dict | None = None):
        self.message = message
        self.agent = agent
        self.context = context or {}
        super().__init__(f"[{agent}] {message}")

    def to_dict(self) -> dict:
        """Convert error to dictionary for logging/serialization.

        Returns:
            dict: Error details
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "agent": self.agent,
            "context": self.context,
        }


class RetrievalError(AgentError):
    """Retrieval operation failed.

    Raised when document retrieval fails (vector search, BM25, etc.).
    """

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message, agent="enhanced_retriever", context=context)


class RerankingError(AgentError):
    """Reranking operation failed.

    Raised when document reranking fails.
    """

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message, agent="reranker", context=context)


class GenerationError(AgentError):
    """Response generation failed.

    Raised when LLM response generation fails.
    """

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message, agent="enhanced_response_generator", context=context)


class ToolExecutionError(AgentError):
    """Tool execution failed.

    Raised when a tool in the tool registry fails to execute.
    """

    def __init__(self, tool_name: str, message: str, context: dict | None = None):
        context = context or {}
        context["tool_name"] = tool_name
        super().__init__(message, agent=f"tool:{tool_name}", context=context)


class EmbeddingError(AgentError):
    """Embedding generation failed.

    Raised when query or document embedding fails.
    """

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message, agent="embedder", context=context)


class ConfigurationError(AgentError):
    """Configuration error.

    Raised when system configuration is invalid or missing.
    """

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message, agent="configuration", context=context)
