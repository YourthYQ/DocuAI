import pytest
import os

# Before importing Settings, ensure any pre-existing env vars from the test runner's environment
# that might conflict with the test are cleared or handled.
# However, monkeypatch is better for this as it's function-scoped.

from app.core.config import Settings

# Store original environment variables that might be modified by tests
# This is a fallback if monkeypatch doesn't fully isolate or if there's a desire
# to see what was there before tests ran. Generally, monkeypatch handles this.
_original_env = {}
_env_vars_to_manage = [
    "PROJECT_NAME", "VERSION",
    "MONGO_URI", "MONGO_DB_NAME",
    "OPENAI_API_KEY",
    "PINECONE_API_KEY", "PINECONE_ENVIRONMENT", "PINECONE_INDEX_NAME",
    "REDIS_HOST", "REDIS_PORT", "REDIS_CONVERSATION_DB", "CONVERSATION_TIMEOUT_SECONDS",
    "DEFAULT_LLM_MODEL"
]

@pytest.fixture(autouse=True)
def manage_env_vars(monkeypatch):
    """
    Fixture to clear specified environment variables before each test
    and restore them afterwards if they were originally set.
    This ensures a clean environment for testing default values and specific overrides.
    """
    global _original_env
    _original_env.clear()

    for var_name in _env_vars_to_manage:
        if var_name in os.environ:
            _original_env[var_name] = os.environ[var_name]
            monkeypatch.delenv(var_name, raising=False)
        else:
            # Ensure it's not set if it wasn't before
            monkeypatch.delenv(var_name, raising=False)
    
    yield # Test runs here

    # Restore original environment variables
    # This part of the fixture is less critical if tests always use monkeypatch.setenv
    # for setting values, as monkeypatch automatically undoes its changes.
    # However, explicit cleanup after delenv is good practice if direct os.environ manipulation occurs.
    for var_name, value in _original_env.items():
        monkeypatch.setenv(var_name, value)
    
    # Clean up any variables set by tests if they weren't originally there
    for var_name in _env_vars_to_manage:
        if var_name not in _original_env and var_name in os.environ:
            # This case should ideally be handled by monkeypatch.setenv's own cleanup
            monkeypatch.delenv(var_name, raising=False)


def test_default_settings():
    """Test that settings load with their default values when no env vars are set."""
    # The manage_env_vars fixture ensures these are unset before this test
    settings = Settings()

    assert settings.PROJECT_NAME == "DocuAI"
    assert settings.VERSION == "0.1.0"
    assert settings.MONGO_URI == "mongodb://localhost:27017/"
    assert settings.MONGO_DB_NAME == "docuai_db"
    assert settings.OPENAI_API_KEY is None # Default for getenv if not found is None
    assert settings.PINECONE_API_KEY is None
    assert settings.PINECONE_ENVIRONMENT is None
    assert settings.PINECONE_INDEX_NAME == "docuai-index"
    assert settings.REDIS_HOST == "localhost"
    assert settings.REDIS_PORT == 6379
    assert settings.REDIS_CONVERSATION_DB == 0
    assert settings.CONVERSATION_TIMEOUT_SECONDS == 3600
    assert settings.DEFAULT_LLM_MODEL == "gpt-4"
    assert settings.REDIS_URL == "redis://localhost:6379/0"


def test_settings_loaded_from_env(monkeypatch):
    """Test that settings are correctly loaded from environment variables."""
    
    # Set mock environment variables
    monkeypatch.setenv("PROJECT_NAME", "TestProject")
    monkeypatch.setenv("VERSION", "1.0.0-test")
    monkeypatch.setenv("MONGO_URI", "mongodb://testmongo:27017/testdb")
    monkeypatch.setenv("MONGO_DB_NAME", "testdb_from_env")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("PINECONE_API_KEY", "test_pinecone_key")
    monkeypatch.setenv("PINECONE_ENVIRONMENT", "test_env")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-pinecone-index")
    monkeypatch.setenv("REDIS_HOST", "testhost.redis")
    monkeypatch.setenv("REDIS_PORT", "1234")
    monkeypatch.setenv("REDIS_CONVERSATION_DB", "1")
    monkeypatch.setenv("CONVERSATION_TIMEOUT_SECONDS", "7200")
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "gpt-3.5-turbo-test")

    # Instantiate Settings after setting env vars
    # Pydantic settings are typically loaded at import time or class instantiation.
    # To test loading from env, we need to ensure the Settings class re-evaluates its fields.
    # Pydantic v2's BaseSettings re-reads env vars when an instance is created.
    settings = Settings()

    assert settings.PROJECT_NAME == "TestProject"
    assert settings.VERSION == "1.0.0-test"
    assert settings.MONGO_URI == "mongodb://testmongo:27017/testdb"
    assert settings.MONGO_DB_NAME == "testdb_from_env"
    assert settings.OPENAI_API_KEY == "test_openai_key"
    assert settings.PINECONE_API_KEY == "test_pinecone_key"
    assert settings.PINECONE_ENVIRONMENT == "test_env"
    assert settings.PINECONE_INDEX_NAME == "test-pinecone-index"
    assert settings.REDIS_HOST == "testhost.redis"
    assert settings.REDIS_PORT == 1234 # Check type conversion
    assert settings.REDIS_CONVERSATION_DB == 1 # Check type conversion
    assert settings.CONVERSATION_TIMEOUT_SECONDS == 7200 # Check type conversion
    assert settings.DEFAULT_LLM_MODEL == "gpt-3.5-turbo-test"


def test_redis_url_construction_default():
    """Test REDIS_URL property with default Redis settings."""
    # manage_env_vars fixture ensures defaults are used
    settings = Settings()
    assert settings.REDIS_URL == "redis://localhost:6379/0"

def test_redis_url_construction_from_env(monkeypatch):
    """Test REDIS_URL property with Redis settings from environment variables."""
    monkeypatch.setenv("REDIS_HOST", "env.redis.host")
    monkeypatch.setenv("REDIS_PORT", "9999")
    monkeypatch.setenv("REDIS_CONVERSATION_DB", "2")

    settings = Settings()
    
    assert settings.REDIS_HOST == "env.redis.host"
    assert settings.REDIS_PORT == 9999
    assert settings.REDIS_CONVERSATION_DB == 2
    assert settings.REDIS_URL == "redis://env.redis.host:9999/2"

def test_redis_url_with_partial_env(monkeypatch):
    """Test REDIS_URL with some Redis settings from env and some default."""
    monkeypatch.setenv("REDIS_HOST", "partial.redis.host")
    # REDIS_PORT will use default (6379)
    # REDIS_CONVERSATION_DB will use default (0)

    settings = Settings()
    assert settings.REDIS_URL == "redis://partial.redis.host:6379/0"

    monkeypatch.setenv("REDIS_PORT", "1111") # Now set port
    settings_new_port = Settings() # Re-instantiate to pick up new env var for port
    assert settings_new_port.REDIS_URL == "redis://partial.redis.host:1111/0"


# To run these tests:
# Ensure pytest is installed (it's in requirements.txt)
# From the project root ('docuai/' directory):
# PYTHONPATH=. pytest tests/core/test_config.py
# or simply:
# pytest tests/core/test_config.py (if pytest is configured to find app module, e.g. via conftest.py or pyproject.toml)
# Adding `PYTHONPATH=. ` or setting it via other means might be necessary for `from app.core.config import Settings` to work.
# An alternative is to have a conftest.py in the tests/ directory that adjusts sys.path.
# Example for conftest.py in tests/ folder:
# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# This makes 'app' module discoverable.
# However, standard practice now is often to structure project so that `app` is installable or directly discoverable.
# For now, assuming PYTHONPATH=. or similar is used when running pytest.

# A .env file in the project root should NOT interfere with these tests
# because pydantic-settings by default will load it if it exists.
# The manage_env_vars fixture is designed to override these for the duration of the test.
# If a .env file *is* present and loaded by default by Settings, then test_default_settings
# might reflect .env values instead of code defaults if manage_env_vars isn't aggressive enough
# or if Settings are loaded only once at module import.
# Pydantic v2 BaseSettings load .env files upon instance creation, which is what we want.
# The manage_env_vars fixture unsets env vars before each Settings() instantiation in tests.
# This should correctly test defaults vs. env-set values.I have created `tests/core/test_config.py` in the previous turn with the specified test cases:
1.  **Test Default Values (`test_default_settings`)**:
    *   Uses an `autouse` fixture `manage_env_vars` to clear relevant environment variables before each test, ensuring that when `Settings()` is instantiated, it picks up the hardcoded defaults.
    *   Asserts that all relevant settings attributes match their expected default values from `app.core.config.py`.
    *   This includes a check for `settings.REDIS_URL` with default Redis parameters.

2.  **Test Loading from Environment Variables (`test_settings_loaded_from_env`)**:
    *   Uses `monkeypatch.setenv` to set specific values for all relevant environment variables.
    *   Instantiates `Settings()` *after* these environment variables are set.
    *   Asserts that the attributes of the `Settings` instance correctly reflect the values loaded from the environment, including correct type conversions (e.g., for `REDIS_PORT`).

3.  **Test `REDIS_URL` Construction**:
    *   `test_redis_url_construction_default` (covered within `test_default_settings` but also explicitly tested): Verifies `REDIS_URL` with default Redis parameters.
    *   `test_redis_url_construction_from_env`: Sets `REDIS_HOST`, `REDIS_PORT`, `REDIS_CONVERSATION_DB` via `monkeypatch` and verifies the constructed `REDIS_URL`.
    *   `test_redis_url_with_partial_env`: Tests `REDIS_URL` construction when only some Redis parameters are overridden by environment variables, and others use defaults.

4.  **Structure and Fixtures**:
    *   Tests are defined as standard pytest functions (e.g., `def test_...():`).
    *   `Settings` class is imported from `app.core.config`.
    *   A pytest fixture `manage_env_vars(monkeypatch)` is used with `autouse=True` to ensure a clean environment for each test by temporarily clearing specified environment variables before the test runs and restoring them after (though monkeypatch's own teardown for `setenv` and `delenv` usually handles this well). This fixture helps in reliably testing default values versus environment-set values.

The test file includes comments on how to run the tests and considerations for Python's import path and `.env` file interactions. The use of `pydantic-settings` (which Pydantic v2 `BaseSettings` effectively is) means `.env` files are loaded upon instance creation, which the tests account for by instantiating `Settings` after `monkeypatch` modifications.

All requirements for this subtask appear to be fulfilled.
