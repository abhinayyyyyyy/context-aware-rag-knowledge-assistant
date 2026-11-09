"""
Cloud LLM Provider.

Implementation of BaseLLMProvider that communicates with OpenAI-compatible
chat completion endpoints (OpenAI, Groq, Together, DeepSeek, etc.).
"""

import json
import os
from typing import Any, Dict, Optional

import requests

from src.llm_provider_interface import BaseLLMProvider


class CloudLLMProvider(BaseLLMProvider):
    """
    Cloud LLM Provider implementing OpenAI-compatible Chat Completions API.
    Ensures safe key handling, configurable models/endpoints, and clean error masking.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        """
        Initialize the Cloud LLM provider.

        Args:
            api_key: Optional API key. If omitted, falls back to CLOUD_LLM_API_KEY
                     or LLM_API_KEY environment variables.
            model_name: Optional model name. If omitted, falls back to CLOUD_LLM_MODEL
                        or defaults to 'gpt-4o-mini'.
            base_url: Optional base endpoint URL. If omitted, falls back to CLOUD_LLM_BASE_URL
                      or defaults to 'https://api.openai.com/v1'.
            timeout: HTTP request timeout in seconds. Defaults to 60.0.

        Raises:
            ValueError: If no valid API key is found in arguments or environment.
        """
        resolved_key = api_key or os.getenv("CLOUD_LLM_API_KEY") or os.getenv("LLM_API_KEY")
        if not resolved_key or not resolved_key.strip():
            raise ValueError(
                "Missing required Cloud LLM API key. Set the 'CLOUD_LLM_API_KEY' "
                "environment variable or pass api_key directly to constructor."
            )

        self._api_key = resolved_key.strip()
        self.model_name = (
            model_name.strip()
            if model_name and model_name.strip()
            else os.getenv("CLOUD_LLM_MODEL", "gpt-4o-mini").strip()
        )
        raw_base_url = (
            base_url.strip()
            if base_url and base_url.strip()
            else os.getenv("CLOUD_LLM_BASE_URL", "https://api.openai.com/v1").strip()
        )
        self.base_url = raw_base_url.rstrip("/")
        self.timeout = float(timeout)

    @property
    def endpoint_url(self) -> str:
        """Construct the full chat completions endpoint URL."""
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _mask_secret(self, text: str) -> str:
        """Safely mask any appearance of the secret API key in strings."""
        if hasattr(self, "_api_key") and self._api_key:
            return text.replace(self._api_key, "[REDACTED]")
        return text

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text completion from prompt using cloud endpoint.

        Args:
            prompt: Text prompt to submit.
            temperature: Sampling temperature. Defaults to 0.0.
            max_tokens: Optional maximum completion tokens.

        Returns:
            Generated text string.

        Raises:
            ValueError: If prompt is empty or API response contains no content.
            RuntimeError: If HTTP communication fails or API returns error status.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        headers: Dict[str, str] = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "RAG-Knowledge-Assistant/1.0",
        }

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)

        try:
            response = requests.post(
                self.endpoint_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as exc:
            safe_msg = self._mask_secret(str(exc))
            raise RuntimeError(f"Cloud LLM request network failure: {safe_msg}") from None

        if response.status_code != 200:
            error_detail = response.text
            try:
                err_json = response.json()
                if "error" in err_json and isinstance(err_json["error"], dict):
                    error_detail = err_json["error"].get("message", error_detail)
            except Exception:
                pass
            safe_detail = self._mask_secret(str(error_detail))
            raise RuntimeError(
                f"Cloud LLM API error (HTTP {response.status_code}): {safe_detail}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError(f"Cloud LLM response parse error: {exc}") from None

        choices = data.get("choices")
        if not choices or not isinstance(choices, list):
            raise ValueError("Cloud LLM response missing 'choices' list.")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Cloud LLM returned an empty response.")

        return content.strip()

    def __repr__(self) -> str:
        return (
            f"CloudLLMProvider(model_name='{self.model_name}', "
            f"base_url='{self.base_url}', api_key='***')"
        )
