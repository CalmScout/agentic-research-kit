"""Model selector for multi-provider LLM strategy.

Implements intelligent model selection with automatic fallback:
- Supports: DeepSeek, OpenAI, local Qwen3.5-4B via vLLM
- Configurable provider chain with automatic fallback
- Zero abstraction overhead with direct API calls
"""

import logging
import os
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.agents.providers import find_by_name
from src.utils.config import Settings

logger = logging.getLogger(__name__)


import re

class ThinkingProcessStripper(ChatOpenAI):
    """Wrapper that strips 'Thinking Process:' and '<thought>' blocks from LLM responses.
    
    This ensures that internal JSON parsers (like those in LightRAG or the agents)
    don't fail when the model outputs its reasoning chain.
    """
    def _strip_thinking(self, content: str) -> str:
        if not content:
            return content
            
        # 1. Strip <thought>...</thought> blocks
        content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL)
        
        # 2. Aggressive JSON extraction: if string contains '{' and '}', try to extract just that
        # This fixes issues where LightRAG or other tools expect raw JSON but get conversational prefix
        if "{" in content and "}" in content:
            try:
                # Find the first '{' and last '}'
                start = content.find("{")
                end = content.rfind("}") + 1
                potential_json = content[start:end]
                # Basic validation: ensure it has key characters
                if ":" in potential_json and '"' in potential_json:
                    return potential_json
            except Exception:
                pass

        # 3. Strip 'Thinking Process:' blocks if no clean JSON was found
        if "Thinking Process:" in content:
            parts = content.split("Thinking Process:", 1)
            after_thinking = parts[1]
            # Try to find a clear break after the thinking block
            if "\n\n" in after_thinking:
                content = after_thinking.split("\n\n", 1)[1]
            else:
                content = after_thinking
                    
        return content.strip()

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        result = super()._generate(*args, **kwargs)
        for generation in result.generations:
            if isinstance(generation.message, AIMessage):
                generation.message.content = self._strip_thinking(cast(str, generation.message.content))
        return result

    async def _agenerate(self, *args: Any, **kwargs: Any) -> ChatResult:
        result = await super()._agenerate(*args, **kwargs)
        for generation in result.generations:
            if isinstance(generation.message, AIMessage):
                generation.message.content = self._strip_thinking(cast(str, generation.message.content))
        return result


class ModelSelector:
    """Selects appropriate LLM based on configuration and availability.

    Implements multi-provider strategy:
    1. Try primary provider (DeepSeek, OpenAI, or local)
    2. Fallback through configured provider chain
    3. Final fallback to local Qwen3.5-4B
    4. Configurable via environment variables
    """

    def __init__(self, settings: Settings | None = None):
        """Initialize model selector.

        Args:
            settings: Application settings (uses defaults if None)
        """
        self.settings = settings or Settings()
        self._local_llm: BaseChatModel | None = None
        self._provider_cache: dict = {}  # Cache for provider instances

    def get_local_llm(self) -> BaseChatModel:
        """Get local Qwen3.5-4B model via vLLM for fast, cost-free inference.

        Returns:
            BaseChatModel: Local Qwen3.5-4B model instance (LangChain-compatible)

        Raises:
            RuntimeError: If local model cannot be initialized
        """
        if self._local_llm is None:
            try:
                import httpx
                logger.info("Initializing local Qwen3.5-4B model via vLLM...")
                
                # Fast check if vLLM server is alive to trigger fallback quickly if down
                try:
                    response = httpx.get("http://localhost:8001/v1/models", timeout=2.0)
                    response.raise_for_status()
                except Exception as e:
                    raise RuntimeError(f"Local vLLM server is not reachable at http://localhost:8001: {e}. Ensure the 'vllm' docker container is running.")
                
                # We use LangChain's standard ChatOpenAI pointing to the local vLLM server
                # Native vLLM handles <thought> blocks and continuous batching efficiently
                # Wrapped in ThinkingProcessStripper for clean JSON/text extraction
                self._local_llm = ThinkingProcessStripper(
                    model="Qwen/Qwen3.5-4B",
                    base_url="http://localhost:8001/v1",
                    api_key="EMPTY",  # vLLM default doesn't require a real API key
                    temperature=0.0,
                    timeout=120.0,
                )
                
                logger.info("✓ Local Qwen3.5-4B model initialized via vLLM")
            except Exception as e:
                logger.error(f"Failed to initialize local LLM: {e}")
                raise RuntimeError(f"Local LLM initialization failed: {e}") from e

        return self._local_llm

    def get_llm_for_provider(self, provider_name: str) -> BaseChatModel:
        """Get LLM instance for specified provider using registry.

        Args:
            provider_name: Provider name (deepseek, openai, local, etc.)

        Returns:
            BaseChatModel: Provider-specific LLM instance

        Raises:
            ValueError: If provider unknown or credentials missing
            RuntimeError: If provider cannot be initialized
        """
        # Check cache
        if provider_name in self._provider_cache:
            return cast(BaseChatModel, self._provider_cache[provider_name])

        spec = find_by_name(provider_name)
        if not spec:
            raise ValueError(f"Unknown provider: {provider_name}")

        logger.info(f"Initializing LLM for provider: {spec.label}")

        if spec.name == "local":
            llm = self.get_local_llm()
        else:
            # Handle API providers (OpenAI-compatible)
            api_key = getattr(self.settings, f"{spec.name}_api_key", None)
            if not api_key:
                # Fallback to direct env var check
                api_key = os.environ.get(spec.env_key)

            if not api_key:
                raise ValueError(
                    f"API key missing for provider {spec.name} (checked {spec.env_key})"
                )

            # Get model name from settings or default
            model_name = getattr(self.settings, f"{spec.name}_model", None)
            if not model_name:
                # Default model names if not in settings
                defaults = {"deepseek": "deepseek-chat", "openai": "gpt-4"}
                model_name = defaults.get(spec.name, "gpt-4")

            base_url = getattr(self.settings, f"{spec.name}_base_url", spec.default_api_base)

            llm = ChatOpenAI(
                model=model_name,
                api_key=cast(Any, api_key),
                base_url=base_url if base_url else None,
                temperature=0.0,
                timeout=120.0,
            )

        logger.info(f"✓ {spec.label} LLM initialized")
        self._provider_cache[provider_name] = llm
        return cast(BaseChatModel, llm)

    def get_llm_with_fallback(self) -> BaseChatModel:
        """Get LLM with automatic fallback.

        Strategy:
        1. Use configured primary provider from settings
        2. Fallback through provider chain if available
        3. Final fallback to local Qwen3.5-4B

        Returns:
            BaseChatModel: Selected LLM instance
        """
        # Determine primary provider
        primary_provider = self.settings.llm_provider

        # Get fallback chain
        fallback_providers = getattr(self.settings, "fallback_providers", ["local"])

        # Try primary provider
        try:
            return self.get_llm_for_provider(primary_provider)
        except Exception as e:
            logger.warning(f"Primary provider {primary_provider} failed: {e}")

        # Try fallback providers
        for provider in fallback_providers:
            try:
                logger.info(f"Trying fallback provider: {provider}")
                return self.get_llm_for_provider(provider)
            except Exception as e:
                logger.warning(f"Fallback provider {provider} failed: {e}")
                continue

        # All providers failed
        raise RuntimeError(
            f"All LLM providers failed. Tried: {primary_provider}, {fallback_providers}"
        )


# Singleton instance for efficiency
_selector: ModelSelector | None = None


def get_model_selector() -> ModelSelector:
    """Get singleton model selector instance."""
    global _selector
    if _selector is None:
        _selector = ModelSelector()
    return _selector
