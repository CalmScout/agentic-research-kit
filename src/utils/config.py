"""Configuration management using Pydantic settings."""

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=[".env.defaults", ".env"],
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Multi-Provider LLM Configuration
    # -------------------------------------------------------------------------
    llm_provider: Literal["deepseek", "openai", "ollama", "local"] = Field(
        default="deepseek", description="Primary LLM provider"
    )
    llm_mode: Literal["api", "local"] = Field(
        default="api", description="LLM mode: 'api' (default) or 'local'"
    )
    fallback_providers: list[str] = Field(
        default=["local"],
        description="Fallback provider chain (e.g., ['local'] or ['openai', 'local'])",
    )
    llm_fallback_to_local: bool = Field(
        default=True, description="Fallback to local model if API fails"
    )

    # -------------------------------------------------------------------------
    # DeepSeek-specific settings (backward compatibility)
    # -------------------------------------------------------------------------
    deepseek_api_key: str | None = Field(
        default=None, description="DeepSeek API key (required when provider=deepseek and mode=api)"
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com", description="DeepSeek API base URL (OpenAI-compatible)"
    )
    deepseek_model: str = Field(
        default="deepseek-chat",
        description="DeepSeek model for API calls (deepseek-chat or deepseek-reasoner)",
    )

    # -------------------------------------------------------------------------
    # OpenAI Configuration
    # -------------------------------------------------------------------------
    openai_api_key: str | None = Field(
        default=None, description="OpenAI API key (optional, for provider=openai)"
    )

    # -------------------------------------------------------------------------
    # Ollama Configuration
    # -------------------------------------------------------------------------
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama API base URL"
    )
    ollama_model: str = Field(default="llama3:8b", description="Ollama model name")

    # -------------------------------------------------------------------------
    # HuggingFace Configuration
    # -------------------------------------------------------------------------
    embedding_model: str = Field(
        default="Qwen/Qwen3-VL-Embedding-2B", description="HuggingFace embedding model"
    )
    reranker_model: str = Field(
        default="Qwen/Qwen3-VL-Reranker-2B", description="HuggingFace reranker model"
    )
    hf_home: str = Field(
        default="/root/.cache/huggingface", description="HuggingFace cache directory"
    )

    # -------------------------------------------------------------------------
    # RAG Backend Configuration
    # -------------------------------------------------------------------------
    rag_working_dir: str = Field(
        default="./rag_storage", description="RAG system working directory"
    )
    rag_enable_image_processing: bool = Field(
        default=True, description="Enable image processing in RAG system"
    )
    rag_enable_table_processing: bool = Field(
        default=True, description="Enable table processing in RAG system"
    )
    rag_enable_equation_processing: bool = Field(
        default=True, description="Enable equation processing in RAG system"
    )

    # -------------------------------------------------------------------------
    # API Configuration
    # -------------------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_reload: bool = Field(default=False, description="Enable API auto-reload")

    # -------------------------------------------------------------------------
    # LightRAG HTTP API Configuration
    # -------------------------------------------------------------------------
    lightrag_api_host: str = Field(default="localhost", description="LightRAG API server host")
    lightrag_api_port: int = Field(
        default=9621, description="LightRAG API server port (default: 9621)"
    )
    lightrag_use_http: bool = Field(
        default=False,  # Disabled by default - requires OpenAI API key for query embeddings
        description="Use HTTP API for LightRAG (avoids async conflicts, but requires API key)",
    )
    lightrag_auto_start_server: bool = Field(
        default=True, description="Auto-start LightRAG server on first query"
    )

    # -------------------------------------------------------------------------
    # Logging Configuration
    # -------------------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )
    log_format: Literal["json", "text"] = Field(default="json", description="Log output format")
    log_dir: str = Field(default="./logs", description="Log directory")

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    data_dir: str = Field(default="./data", description="Data directory")
    image_dir: str = Field(default="./data/images", description="Image directory")

    # Legacy CSV support (backward compatibility)
    csv_path: str = Field(
        default="./data/claim_matching_dataset.csv",
        description="Path to CSV dataset (legacy support)",
    )

    # Document processing settings
    supported_formats: list = Field(
        default=[".pdf", ".docx", ".txt", ".html", ".csv"],
        description="Supported document formats for ingestion",
    )

    # Image download settings
    image_download_timeout: int = Field(default=10, description="Image download timeout in seconds")
    image_download_workers: int = Field(
        default=10, description="Number of concurrent image downloads"
    )
    image_download_retry: int = Field(
        default=3, description="Number of retry attempts for failed downloads"
    )

    # -------------------------------------------------------------------------
    # Prompt Configuration
    # -------------------------------------------------------------------------
    prompt_template: str = Field(
        default="research", description="Default prompt template (research, analysis, qa)"
    )
    custom_prompt_path: str | None = Field(
        default=None, description="Path to custom prompt template file (JSON or text)"
    )

    # -------------------------------------------------------------------------
    # Evaluation Configuration
    # -------------------------------------------------------------------------
    test_set_size: int = Field(default=20, description="Number of test queries")
    eval_output_dir: str = Field(default="./reports", description="Evaluation output directory")

    # -------------------------------------------------------------------------
    # Observability Configuration (Phoenix / Arize AI)
    # -------------------------------------------------------------------------
    phoenix_enabled: bool = Field(
        default=False, description="Enable Phoenix observability for tracing"
    )
    phoenix_collector_endpoint: str = Field(
        default="http://localhost:6006/v1/traces",
        description="Phoenix collector endpoint for OpenTelemetry traces",
    )
    phoenix_project_name: str = Field(
        default="agentic-research-kit", description="Project name for Phoenix traces"
    )

    # -------------------------------------------------------------------------
    # Agent Configuration
    # -------------------------------------------------------------------------
    retrieval_top_k: int = Field(default=50, description="Number of documents to retrieve")
    rerank_top_k: int = Field(default=10, description="Number of documents to rerank")
    confidence_threshold: float = Field(
        default=0.5, description="Minimum confidence threshold for responses"
    )

    mcp_servers: list[dict] = Field(
        default_factory=list,
        description="List of MCP server configurations (name, command, args, url, etc.)",
    )

    brave_api_key: str | None = Field(
        default=None, description="Brave Search API key for web research"
    )

    # -------------------------------------------------------------------------
    # Telegram Configuration
    # -------------------------------------------------------------------------
    telegram_enabled: bool = Field(default=False, description="Enable Telegram channel")
    telegram_token: str | None = Field(default=None, description="Telegram bot token")
    telegram_allowed_users: list[str] = Field(
        default_factory=list,
        description="List of user IDs or usernames allowed to use the bot (empty = allow all)",
    )
    telegram_proxy: str | None = Field(default=None, description="Proxy URL for Telegram")

    # -------------------------------------------------------------------------
    # Performance Settings
    # -------------------------------------------------------------------------
    embedding_batch_size: int = Field(default=32, description="Batch size for embedding generation")
    max_concurrent_requests: int = Field(default=5, description="Maximum concurrent requests")
    request_timeout: int = Field(default=120, description="Request timeout in seconds")

    # -------------------------------------------------------------------------
    # Backward Compatibility (Deprecated aliases)
    # -------------------------------------------------------------------------
    @property
    def deepseek_mode(self) -> str:
        """Backward compatibility property for llm_mode."""
        return self.llm_mode

    @property
    def deepseek_fallback_to_local(self) -> bool:
        """Backward compatibility property for llm_fallback_to_local."""
        return self.llm_fallback_to_local

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------
    @field_validator("deepseek_api_key")
    @classmethod
    def validate_deepseek_api_key(cls, v: str | None, info) -> str | None:
        """Validate DeepSeek API key when mode is 'api'."""
        mode = info.data.get("llm_mode") or info.data.get("deepseek_mode")
        provider = info.data.get("llm_provider")

        if (mode == "api" or provider == "deepseek") and not v:
            # Only require API key if using DeepSeek in API mode
            if provider == "deepseek":
                raise ValueError(
                    "DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek and LLM_MODE=api. "
                    "Set DEEPSEEK_API_KEY in .env or change LLM_MODE to 'local'."
                )
        return v

    @field_validator("confidence_threshold")
    @classmethod
    def validate_confidence_threshold(cls, v: float) -> float:
        """Validate confidence threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("CONFIDENCE_THRESHOLD must be between 0 and 1")
        return v


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get global settings instance (singleton pattern)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def clear_settings_cache() -> None:
    """Clear global settings cache (useful for testing)."""
    global _settings
    _settings = None
