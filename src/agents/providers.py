"""Provider Registry — single source of truth for LLM provider metadata.

Adapted from nanobot for ARK.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ProviderSpec:
    """One LLM provider's metadata."""
    name: str                       # config field name, e.g. "deepseek"
    keywords: Tuple[str, ...]       # model-name keywords for matching
    env_key: str                    # Env var for API key
    display_name: str = ""          # Display name
    litellm_prefix: str = ""        # Prefix for LiteLLM
    default_api_base: str = ""      # Default API base URL
    is_local: bool = False          # Local provider (Ollama, vLLM)

    @property
    def label(self) -> str:
        return self.display_name or self.name.title()


PROVIDERS: Tuple[ProviderSpec, ...] = (
    ProviderSpec(
        name="deepseek",
        keywords=("deepseek",),
        env_key="DEEPSEEK_API_KEY",
        display_name="DeepSeek",
        litellm_prefix="deepseek",
        default_api_base="https://api.deepseek.com",
    ),
    ProviderSpec(
        name="openai",
        keywords=("openai", "gpt"),
        env_key="OPENAI_API_KEY",
        display_name="OpenAI",
        litellm_prefix="",
    ),
    ProviderSpec(
        name="anthropic",
        keywords=("anthropic", "claude"),
        env_key="ANTHROPIC_API_KEY",
        display_name="Anthropic",
        litellm_prefix="",
    ),
    ProviderSpec(
        name="ollama",
        keywords=("ollama",),
        env_key="",
        display_name="Ollama",
        litellm_prefix="ollama",
        default_api_base="http://localhost:11434",
        is_local=True,
    ),
    ProviderSpec(
        name="local",
        keywords=("local",),
        env_key="",
        display_name="Local Qwen",
        litellm_prefix="",
        is_local=True,
    ),
)


def find_by_name(name: str) -> Optional[ProviderSpec]:
    """Find a provider spec by name."""
    for spec in PROVIDERS:
        if spec.name == name:
            return spec
    return None


def find_by_model(model: str) -> Optional[ProviderSpec]:
    """Match a provider by model-name keyword."""
    model_lower = model.lower()
    for spec in PROVIDERS:
        if any(kw in model_lower for kw in spec.keywords):
            return spec
    return None
