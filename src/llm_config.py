"""
LLM Configuration Module.

Manages environment variables and configuration settings for LLM providers
(local Ollama and cloud providers).
"""

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

DEFAULT_API_KEY_VAR = "LLM_API_KEY"
DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_OLLAMA_MODEL = "llama3.2:latest"
DEFAULT_CLOUD_MODEL = "gpt-4o-mini"
DEFAULT_CLOUD_BASE_URL = "https://api.openai.com/v1"


def get_llm_provider() -> str:
    """
    Get the configured LLM provider type from environment.
    Defaults to 'ollama' for safe local development without cloud credentials.
    """
    return os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()


def get_ollama_model() -> str:
    """Get the configured Ollama model name."""
    return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()


def get_cloud_llm_model() -> str:
    """Get the configured Cloud LLM model name."""
    return os.getenv("CLOUD_LLM_MODEL", DEFAULT_CLOUD_MODEL).strip()


def get_cloud_llm_base_url() -> str:
    """Get the configured Cloud LLM base URL endpoint."""
    return os.getenv("CLOUD_LLM_BASE_URL", DEFAULT_CLOUD_BASE_URL).strip()


def get_llm_api_key(env_var_name: str = DEFAULT_API_KEY_VAR) -> str:
    """
    Retrieve the LLM API key from an environment variable.

    Args:
        env_var_name: Name of the environment variable containing the API key.

    Returns:
        The API key as a string.

    Raises:
        ValueError: If the API key is missing or empty.
    """
    api_key: Optional[str] = os.getenv(env_var_name)

    if not api_key or not api_key.strip():
        raise ValueError(
            f"Missing required API key: environment variable "
            f"'{env_var_name}' is not set or empty."
        )

    return api_key.strip()


def get_cloud_llm_api_key() -> str:
    """
    Retrieve the Cloud LLM API key, checking CLOUD_LLM_API_KEY then LLM_API_KEY.

    Returns:
        The API key as a string.

    Raises:
        ValueError: If neither environment variable is set or non-empty.
    """
    key = os.getenv("CLOUD_LLM_API_KEY") or os.getenv("LLM_API_KEY")
    if not key or not key.strip():
        raise ValueError(
            "Missing required Cloud LLM API key. Set the 'CLOUD_LLM_API_KEY' "
            "or 'LLM_API_KEY' environment variable."
        )
    return key.strip()


class LLMConfig:
    """Stores basic configuration for the LLM connection."""

    def __init__(
        self,
        api_key_env_var: str = DEFAULT_API_KEY_VAR,
        model_name: Optional[str] = None,
    ) -> None:
        self.api_key_env_var = api_key_env_var
        self.model_name = model_name

    def get_api_key(self) -> str:
        """Return the configured LLM API key."""
        return get_llm_api_key(self.api_key_env_var)
