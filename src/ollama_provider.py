"""
Ollama LLM Provider.

Implementation of BaseLLMProvider that communicates with a local Ollama daemon.
"""

from typing import Any, Dict, Optional
import ollama

from src.llm_provider_interface import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """
    Local LLM provider wrapping the Ollama Python client.
    """

    def __init__(self, model_name: str = "llama3.2:latest") -> None:
        """
        Initialize the Ollama provider.

        Args:
            model_name: Name of the Ollama model to use. Defaults to 'llama3.2:latest'.
        """
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text using local Ollama service.

        Args:
            prompt: Text prompt to send to the model.
            temperature: Sampling temperature. Defaults to 0.0.
            max_tokens: Optional maximum token limit for generation.

        Returns:
            Generated text string.

        Raises:
            ValueError: If Ollama returns an empty response.
            Exception: If Ollama service is unreachable or encounters an error.
        """
        options: Dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options=options,
        )

        response_text = response.get("response", "").strip()
        if not response_text:
            raise ValueError(f"Ollama model '{self.model_name}' returned an empty response.")

        return response_text
