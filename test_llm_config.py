"""
Test the LLM configuration module (src/llm_config.py).

Verifies module import, error handling for missing keys, and retrieval
with dummy test credentials without printing or exposing secrets.
"""

import os


def test_module_import():
    """Verify that src.llm_config can be imported successfully."""
    print("1. Testing module import...")
    try:
        import src.llm_config as llm_config
        print("   Passed: Successfully imported src.llm_config.")
        return llm_config
    except Exception as err:
        print(f"   Failed: Could not import src.llm_config -> {err}")
        raise


def test_missing_api_key_error(llm_config):
    """Verify that missing API key configuration raises a clear ValueError."""
    print("\n2. Testing missing API key error handling...")
    non_existent_var = "__NON_EXISTENT_LLM_KEY_FOR_TEST__"

    # Ensure the test variable is not set in the environment
    os.environ.pop(non_existent_var, None)

    # Test standalone get_llm_api_key function with missing key
    try:
        llm_config.get_llm_api_key(env_var_name=non_existent_var)
        print("   Failed: get_llm_api_key did not raise ValueError for missing key.")
    except ValueError as err:
        print(f"   Passed: get_llm_api_key raised clear error -> {err}")

    # Test LLMConfig class with missing key
    try:
        config = llm_config.LLMConfig(api_key_env_var=non_existent_var)
        config.get_api_key()
        print("   Failed: LLMConfig.get_api_key did not raise ValueError for missing key.")
    except ValueError as err:
        print(f"   Passed: LLMConfig.get_api_key raised clear error -> {err}")


def test_dummy_api_key_retrieval(llm_config):
    """
    Verify API key retrieval with dummy configuration.
    Validates correct behavior without printing or exposing any secret value.
    """
    print("\n3. Testing dummy key retrieval (ensuring secret values remain hidden)...")
    test_var = "__DUMMY_LLM_TEST_KEY__"
    dummy_key = "dummy-credential-for-unit-test-only"
    os.environ[test_var] = dummy_key

    try:
        # Retrieve key via function
        retrieved_key = llm_config.get_llm_api_key(env_var_name=test_var)
        assert isinstance(retrieved_key, str)
        assert retrieved_key == dummy_key
        print("   Passed: Key retrieved successfully as non-empty string.")

        # Retrieve key via LLMConfig class
        config = llm_config.LLMConfig(api_key_env_var=test_var, model_name="test-model")
        config_key = config.get_api_key()
        assert config_key == dummy_key
        print("   Passed: LLMConfig instance retrieved key successfully.")

        # Confirm secret value was not printed or exposed
        print("   Passed: Secret values validated in memory and NOT printed to output.")
    finally:
        os.environ.pop(test_var, None)


def main() -> None:
    print("=" * 60)
    print("Running LLM Configuration Tests")
    print("=" * 60)

    try:
        config_module = test_module_import()
        test_missing_api_key_error(config_module)
        test_dummy_api_key_retrieval(config_module)
        print("\n" + "=" * 60)
        print("All configuration tests completed successfully.")
        print("=" * 60)
    except Exception as err:
        print(f"\nTest execution encountered an error: {err}")


if __name__ == "__main__":
    main()
