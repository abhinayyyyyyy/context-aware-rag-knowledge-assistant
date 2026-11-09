"""
LLM Provider Factory.

Instantiates and configures the appropriate BaseLLMProvider implementation
(OllamaProvider or CloudLLMProvider) based on environment configuration.
"""

from typing import Optional

from src.cloud_llm_provider import CloudLLMProvider
from src.llm_config import (
    get_cloud_llm_api_key,
    get_cloud_llm_base_url,
    get_cloud_llm_model,
    get_llm_provider,
    get_ollama_model,
)
from src.llm_provider_interface import BaseLLMProvider
from src.ollama_provider import OllamaProvider


def get_llm_provider_instance(
    provider_type: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> BaseLLMProvider:
    """
    Factory function to create a BaseLLMProvider based on configuration or parameters.

    Args:
        provider_type: Optional provider name ('ollama' or 'cloud').
                       Defaults to environment variable LLM_PROVIDER (which defaults to 'ollama').
        model_name: Optional model override.
        api_key: Optional API key override (for cloud provider).
        base_url: Optional base URL override (for cloud provider).

    Returns:
        Instance of BaseLLMProvider (OllamaProvider or CloudLLMProvider).

    Raises:
        ValueError: If an unknown provider type is specified.
    """
    selected_provider = (
        provider_type.strip().lower()
        if provider_type
        else get_llm_provider()
    )

    if selected_provider == "ollama":
        resolved_model = model_name or get_ollama_model()
        return OllamaProvider(model_name=resolved_model)
    elif selected_provider in ("cloud", "openai"):
        resolved_model = model_name or get_cloud_llm_model()
        resolved_key = api_key or get_cloud_llm_api_key()
        resolved_base_url = base_url or get_cloud_llm_base_url()
        return CloudLLMProvider(
            api_key=resolved_key,
            model_name=resolved_model,
            base_url=resolved_base_url,
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider '{selected_provider}'. "
            f"Supported providers are: 'ollama', 'cloud'."
        )
