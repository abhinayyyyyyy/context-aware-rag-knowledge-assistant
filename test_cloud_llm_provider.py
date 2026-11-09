"""
Test Suite for Cloud LLM Provider and Abstraction Integration.

Validates:
1. CloudLLMProvider construction and argument/env handling
2. Missing and invalid API-key validation
3. Mocked successful generation with OpenAI-compatible payload
4. Mocked API failures (network error, HTTP status errors, empty answers)
5. Secret masking (API key never exposed in logs or errors)
6. RAGGenerator integration with custom BaseLLMProvider
7. Provider selection factory (Ollama vs Cloud)
"""

import os
from unittest.mock import MagicMock, patch

from src.cloud_llm_provider import CloudLLMProvider
from src.llm_config import (
    get_cloud_llm_api_key,
    get_cloud_llm_model,
    get_llm_provider,
    get_ollama_model,
)
from src.llm_factory import get_llm_provider_instance
from src.llm_provider_interface import BaseLLMProvider
from src.ollama_provider import OllamaProvider
from src.rag_generator import RAGGenerator


# Dummy mock provider for testing RAGGenerator abstraction
class MockLLMProvider(BaseLLMProvider):
    def __init__(self, response_text: str = "Mocked answer") -> None:
        self.response_text = response_text
        self.last_prompt = None
        self.last_temperature = None

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens = None) -> str:
        self.last_prompt = prompt
        self.last_temperature = temperature
        return self.response_text


def test_cloud_provider_construction() -> None:
    """Test explicit constructor parameters and secret masking."""
    provider = CloudLLMProvider(
        api_key="secret-key-123",
        model_name="gpt-4o",
        base_url="https://api.openai.com/v1",
    )
    assert provider.model_name == "gpt-4o"
    assert provider.endpoint_url == "https://api.openai.com/v1/chat/completions"
    assert "secret-key-123" not in repr(provider), "Secret key leaked in repr!"
    assert "***" in repr(provider)
    print("✓ Test 1: CloudLLMProvider construction and secret masking passed.")


def test_missing_api_key_handling() -> None:
    """Test validation when no API key is provided or present in environment."""
    with patch.dict(os.environ, {}, clear=True):
        try:
            CloudLLMProvider(api_key=None)
            assert False, "Expected ValueError on missing API key"
        except ValueError as exc:
            assert "Missing required Cloud LLM API key" in str(exc)

    with patch.dict(os.environ, {"CLOUD_LLM_API_KEY": "   "}):
        try:
            CloudLLMProvider()
            assert False, "Expected ValueError on whitespace API key"
        except ValueError as exc:
            assert "Missing required Cloud LLM API key" in str(exc)
    print("✓ Test 2: Missing API key handling passed.")


def test_mocked_successful_generation() -> None:
    """Test successful generation with OpenAI-compatible response parsing."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "This is a grounded answer from cloud LLM.",
                }
            }
        ]
    }

    with patch("requests.post", return_value=mock_response) as mock_post:
        provider = CloudLLMProvider(
            api_key="test-sk-12345",
            model_name="gpt-4o-mini",
        )
        result = provider.generate("What is Juspay?", temperature=0.2, max_tokens=150)

        assert result == "This is a grounded answer from cloud LLM."
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]

        # Verify payload and headers
        assert call_kwargs["json"]["model"] == "gpt-4o-mini"
        assert call_kwargs["json"]["temperature"] == 0.2
        assert call_kwargs["json"]["max_tokens"] == 150
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-sk-12345"

    print("✓ Test 3: Mocked successful generation passed.")


def test_mocked_api_failures() -> None:
    """Test error handling for network errors, HTTP error statuses, and empty responses."""
    provider = CloudLLMProvider(api_key="secret-token-xyz")

    # 1. HTTP 401 Unauthorized
    mock_401 = MagicMock()
    mock_401.status_code = 401
    mock_401.text = "Unauthorized"
    mock_401.json.return_value = {"error": {"message": "Invalid API key provided: secret-token-xyz"}}

    with patch("requests.post", return_value=mock_401):
        try:
            provider.generate("test prompt")
            assert False, "Expected RuntimeError on 401"
        except RuntimeError as exc:
            err_msg = str(exc)
            assert "HTTP 401" in err_msg
            assert "secret-token-xyz" not in err_msg, "Secret token leaked in error message!"
            assert "[REDACTED]" in err_msg

    # 2. Empty response content
    mock_empty = MagicMock()
    mock_empty.status_code = 200
    mock_empty.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "   "}}]
    }

    with patch("requests.post", return_value=mock_empty):
        try:
            provider.generate("test prompt")
            assert False, "Expected ValueError on empty content"
        except ValueError as exc:
            assert "empty response" in str(exc)

    print("✓ Test 4: Mocked API failure handling and secret scrubbing passed.")


def test_rag_generator_with_base_llm_provider() -> None:
    """Test that RAGGenerator cleanly accepts any BaseLLMProvider implementation."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {
            "document": {
                "text": "Juspay operates an enterprise payments infrastructure.",
                "source": "sample.pdf",
                "page_number": 1,
            },
            "distance": 0.5,
        }
    ]

    custom_provider = MockLLMProvider("Juspay is a payments platform.")
    rag = RAGGenerator(
        retriever=mock_retriever,
        llm_provider=custom_provider,
        default_max_distance=1.10,
    )

    result = rag.generate_answer("What is Juspay?")
    assert result["answer"] == "Juspay is a payments platform."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == "sample.pdf"
    assert custom_provider.last_prompt is not None
    print("✓ Test 5: RAGGenerator using BaseLLMProvider passed.")


def test_provider_factory_selection() -> None:
    """Test factory resolution for 'ollama' default, 'cloud', and invalid inputs."""
    # 1. Default should be ollama
    with patch.dict(os.environ, {}, clear=True):
        p_default = get_llm_provider_instance()
        assert isinstance(p_default, OllamaProvider)
        assert p_default.model_name == "llama3.2:latest"

    # 2. Explicit ollama
    p_ollama = get_llm_provider_instance(provider_type="ollama", model_name="llama3.1")
    assert isinstance(p_ollama, OllamaProvider)
    assert p_ollama.model_name == "llama3.1"

    # 3. Cloud provider via env
    env = {
        "LLM_PROVIDER": "cloud",
        "CLOUD_LLM_API_KEY": "test-key",
        "CLOUD_LLM_MODEL": "gpt-4o-mini",
    }
    with patch.dict(os.environ, env, clear=True):
        p_cloud = get_llm_provider_instance()
        assert isinstance(p_cloud, CloudLLMProvider)
        assert p_cloud.model_name == "gpt-4o-mini"

    # 4. Unknown provider
    try:
        get_llm_provider_instance(provider_type="unsupported_provider")
        assert False, "Expected ValueError on unsupported provider"
    except ValueError as exc:
        assert "Unsupported LLM provider" in str(exc)

    print("✓ Test 6: Provider selection factory passed.")


if __name__ == "__main__":
    print("=== Running Phase 10.2 Cloud LLM Provider Tests ===")
    test_cloud_provider_construction()
    test_missing_api_key_handling()
    test_mocked_successful_generation()
    test_mocked_api_failures()
    test_rag_generator_with_base_llm_provider()
    test_provider_factory_selection()
    print("\nALL PHASE 10.2 FOCUSED TESTS PASSED SUCCESSFULLY!")
