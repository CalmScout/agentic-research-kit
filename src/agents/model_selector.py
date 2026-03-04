"""Model selector for multi-provider LLM strategy.

Implements intelligent model selection with automatic fallback:
- Supports: DeepSeek, OpenAI, Ollama, local Qwen3-8B
- Configurable provider chain with automatic fallback
- Zero abstraction overhead with direct API calls
"""

import logging
from collections.abc import Callable, Sequence
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from src.agents.providers import find_by_name
from src.agents.utils import QwenToolParser
from src.utils.config import Settings

# Import Ollama support (optional)
try:
    from langchain_community.chat_models import ChatOllama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("langchain-community not installed, Ollama support unavailable")

logger = logging.getLogger(__name__)


class Qwen2LangChainWrapper(BaseChatModel):
    """LangChain-compatible wrapper for Qwen2TextLLM.

    Adapts the existing Qwen2TextLLM to work with LangChain's async interface.
    Now supports tool calling via QwenToolParser.
    """

    qwen2_llm: Any | None = None  # Make qwen2_llm a class attribute for Pydantic
    tools_defs: list[dict[str, Any]] | None = None

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, qwen2_llm, **kwargs):
        """Initialize wrapper.

        Args:
            qwen2_llm: Qwen2TextLLM instance
            **kwargs: Additional parameters
        """
        super().__init__(qwen2_llm=qwen2_llm, **kwargs)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[
        PromptValue
        | str
        | Sequence[BaseMessage | list[str] | tuple[str, str] | str | dict[str, Any]],
        AIMessage,
    ]:
        """Bind tools to the model for tool calling.

        Args:
            tools: List of tool definitions
            tool_choice: Optional tool choice
            **kwargs: Additional parameters

        Returns:
            Model with tools bound (Runnable)
        """
        formatted_tools: list[dict[str, Any]] = []
        for tool in tools:
            if hasattr(tool, "args_schema"):
                # Handle LangChain tools
                t = cast(BaseTool, tool)
                # Check if it's a Pydantic model with schema method
                params: dict[str, Any] = {}
                if t.args_schema is not None:
                    if hasattr(t.args_schema, "schema"):
                        params = t.args_schema.schema()
                    elif hasattr(t.args_schema, "model_json_schema"):
                        params = t.args_schema.model_json_schema()

                formatted_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": params,
                        },
                    }
                )
            elif isinstance(tool, dict):
                # Assume already formatted or simple dict
                formatted_tools.append(tool)
            elif callable(tool):
                # Simple heuristic for callables
                formatted_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": getattr(tool, "__name__", "unknown"),
                            "description": getattr(tool, "__doc__", "") or "",
                            "parameters": {},  # Basic fallback
                        },
                    }
                )

        self.tools_defs = formatted_tools
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response (synchronous).

        Args:
            messages: List of messages
            stop: Stop sequences
            run_manager: Run manager
            **kwargs: Additional parameters

        Returns:
            ChatResult with generation
        """
        if self.qwen2_llm is None:
            raise ValueError("qwen2_llm is not initialized")

        # Extract system and user messages
        system_prompt = ""
        user_messages = []

        for m in messages:
            if m.type == "system":
                system_prompt = str(m.content)
            else:
                user_messages.append(m)

        # For tool calling, we need the raw list of messages to pass to the processor
        # But Qwen2TextLLM.generate expects a string prompt.
        # We'll adapt it to handle tools if they are bound.

        # Format tools if present
        tools_to_use = kwargs.get("tools") or self.tools_defs

        if tools_to_use:
            # We need to use the processor directly to handle tools correctly
            # Qwen2TextLLM encapsulates the model and tokenizer
            processor = self.qwen2_llm.tokenizer  # Actually the tokenizer in Qwen2TextLLM

            chat_messages = []
            if system_prompt:
                chat_messages.append({"role": "system", "content": system_prompt})

            for m in user_messages:
                role = "assistant" if m.type == "ai" else "user"
                chat_messages.append({"role": role, "content": str(m.content)})

            # Use chat template with tools
            prompt = processor.apply_chat_template(
                chat_messages, tools=tools_to_use, tokenize=False, add_generation_prompt=True
            )
        else:
            # Basic prompt concatenation
            prompt = "\n".join([str(m.content) for m in user_messages])

        response_text = self.qwen2_llm.generate(
            prompt,
            system_prompt=system_prompt if not tools_to_use else None,
            max_tokens=self.qwen2_llm.max_new_tokens,
        )

        # Parse tool calls from response
        tool_calls = QwenToolParser.parse_tool_calls(response_text)
        cleaned_content = QwenToolParser.clean_text(response_text)

        # Create AIMessage
        ai_message = AIMessage(content=cleaned_content, tool_calls=tool_calls)

        # Return ChatResult
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response (async).

        Args:
            messages: List of messages
            stop: Stop sequences
            run_manager: Run manager
            **kwargs: Additional parameters

        Returns:
            ChatResult with generation
        """
        import asyncio

        return await asyncio.get_event_loop().run_in_executor(
            None, self._generate, messages, stop, run_manager, **kwargs
        )

    @property
    def _llm_type(self) -> str:
        """Return LLM type identifier."""
        return "qwen2_langchain_wrapper"


class ModelSelector:
    """Selects appropriate LLM based on configuration and availability.

    Implements multi-provider strategy:
    1. Try primary provider (DeepSeek, OpenAI, Ollama, or local)
    2. Fallback through configured provider chain
    3. Final fallback to local Qwen3-8B
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
        """Get local Qwen3-8B model for fast, cost-free inference.

        Returns:
            BaseChatModel: Local Qwen3-8B model instance (LangChain-compatible)

        Raises:
            RuntimeError: If local model cannot be initialized
        """
        if self._local_llm is None:
            try:
                from src.utils.vision_embedding import get_qwen2_llm

                logger.info("Initializing local Qwen3-8B model...")
                qwen2_llm = get_qwen2_llm()
                # Wrap in LangChain-compatible interface
                self._local_llm = Qwen2LangChainWrapper(qwen2_llm)
                logger.info("✓ Local Qwen3-8B model initialized (LangChain wrapper)")
            except Exception as e:
                logger.error(f"Failed to initialize local LLM: {e}")
                raise RuntimeError(f"Local LLM initialization failed: {e}") from e

        return self._local_llm

    def get_llm_for_provider(self, provider_name: str) -> BaseChatModel:
        """Get LLM instance for specified provider using registry.

        Args:
            provider_name: Provider name (deepseek, openai, ollama, local, etc.)

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
        elif spec.name == "ollama":
            if not OLLAMA_AVAILABLE:
                raise RuntimeError("Ollama support (langchain-community) not installed")
            llm = ChatOllama(
                model=self.settings.ollama_model,
                base_url=self.settings.ollama_base_url,
                temperature=0.0,
            )
        else:
            # Handle API providers (OpenAI-compatible)
            api_key = getattr(self.settings, f"{spec.name}_api_key", None)
            if not api_key:
                # Fallback to direct env var check
                import os

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
            )

        logger.info(f"✓ {spec.label} LLM initialized")
        self._provider_cache[provider_name] = llm
        return cast(BaseChatModel, llm)

    def get_llm_with_fallback(self) -> BaseChatModel:
        """Get LLM with automatic fallback.

        Strategy:
        1. Use configured primary provider from settings
        2. Fallback through provider chain if available
        3. Final fallback to local Qwen3-8B

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
