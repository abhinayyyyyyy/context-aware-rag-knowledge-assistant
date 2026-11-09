"""
LLM Provider Interface.

Defines the abstract base class for Large Language Model providers,
ensuring decoupling between RAG generation logic and model execution backends
(e.g., local Ollama, cloud LLM APIs).
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMProvider(ABC):
    """
    Abstract interface for LLM inference providers.
    Enables switching between local (Ollama) and cloud LLM APIs without modifying RAG logic.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Execute text generation for the given prompt.

        Args:
            prompt: Text prompt to send to the LLM.
            temperature: Sampling temperature for generation. Defaults to 0.0.
            max_tokens: Optional maximum token limit for the response.

        Returns:
            Generated response text as a non-empty string.

        Raises:
            Exception: If the underlying model service is unavailable or generation fails.
        """
        pass
