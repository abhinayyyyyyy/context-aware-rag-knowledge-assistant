import ollama


MODEL_NAME = "llama3.2:latest"
PROMPT = "Explain in one sentence what Juspay does."


def test_ollama() -> None:
    """Test communication with the local Ollama model."""

    print("Connecting to local Ollama service...")

    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=PROMPT,
        )

        response_text = response.get("response", "").strip()

        if not response_text:
            raise ValueError("Ollama returned an empty response.")

        print("\nConnection successful.")
        print(f"Model: {MODEL_NAME}")
        print("\nGenerated response:")
        print(response_text)

    except Exception as err:
        print("\nOllama connection test failed.")
        print(f"Error: {err}")
        print(
            "\nMake sure Ollama is running and "
            f"that the model '{MODEL_NAME}' is available."
        )


if __name__ == "__main__":
    test_ollama()
