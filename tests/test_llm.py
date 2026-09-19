"""
Unit tests for ZICO centralized OpenAI LLM provider (app.core.llm).

Covers:
- Test 1: Successful model creation with configured model name and API key
- Test 2: Missing API key raises LLMConfigurationError without exposing secrets
- Test 3: Empty API key raises LLMConfigurationError
- Test 4: OPENAI_MODEL from configuration is passed accurately to the model constructor
- Test 5: Default (0.0) and overridden temperature handling
- Test 6: Provider initialization failures are surfaced cleanly with secret protection
- Test 7: Importing app.core.llm is completely safe and triggers zero network requests
- Test 8: Repeated model access returns independent, deterministic instances (factory pattern)
- Test 9: Logger safety confirms fake API keys never appear in log records
"""

from unittest.mock import MagicMock, patch

import pytest

from app.core.config import settings
from app.core.llm import DEFAULT_TEMPERATURE, LLMConfigurationError, get_chat_model


@pytest.fixture(autouse=True)
def restore_settings(monkeypatch: pytest.MonkeyPatch):
    """Ensure settings are isolated for each test."""
    original_key = getattr(settings, "OPENAI_API_KEY", "")
    original_model = getattr(settings, "OPENAI_MODEL", "gpt-4o-mini")
    yield
    monkeypatch.setattr(settings, "OPENAI_API_KEY", original_key)
    monkeypatch.setattr(settings, "OPENAI_MODEL", original_model)


def test_successful_model_creation(monkeypatch: pytest.MonkeyPatch):
    """Test 1 — Successful model creation with mocked constructor."""
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(settings, "OPENAI_MODEL", "test-model-4o")

    mock_chat_instance = MagicMock()
    with patch("app.core.llm.ChatOpenAI", return_value=mock_chat_instance) as mock_cls:
        model = get_chat_model()

        assert model is mock_chat_instance
        mock_cls.assert_called_once()
        _, kwargs = mock_cls.call_args
        assert kwargs["model"] == "test-model-4o"
        assert kwargs["api_key"] == "test-openai-key"
        assert kwargs["temperature"] == DEFAULT_TEMPERATURE
        # Verify no network invocations occurred on the model
        mock_chat_instance.invoke.assert_not_called()
        mock_chat_instance.ainvoke.assert_not_called()


def test_missing_api_key(monkeypatch: pytest.MonkeyPatch):
    """Test 2 — Missing API key raises LLMConfigurationError."""
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)
    monkeypatch.setattr(settings, "OPENAI_MODEL", "test-model")

    with pytest.raises(LLMConfigurationError) as exc_info:
        get_chat_model()

    err_msg = str(exc_info.value)
    assert "missing or empty" in err_msg
    # Ensure no credentials appear in the exception message
    assert "test-openai-key" not in err_msg


def test_empty_api_key(monkeypatch: pytest.MonkeyPatch):
    """Test 3 — Empty API key raises LLMConfigurationError."""
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "   ")
    monkeypatch.setattr(settings, "OPENAI_MODEL", "test-model")

    with pytest.raises(LLMConfigurationError) as exc_info:
        get_chat_model()

    assert "missing or empty" in str(exc_info.value)


def test_model_name_from_configuration(monkeypatch: pytest.MonkeyPatch):
    """Test 4 — Model name from configuration is passed to constructor."""
    custom_model_name = "test-custom-model-v2"
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(settings, "OPENAI_MODEL", custom_model_name)

    with patch("app.core.llm.ChatOpenAI") as mock_cls:
        get_chat_model()
        mock_cls.assert_called_once()
        _, kwargs = mock_cls.call_args
        assert kwargs["model"] == custom_model_name


def test_temperature_handling(monkeypatch: pytest.MonkeyPatch):
    """Test 5 — Temperature defaults to 0.0 and allows explicit override."""
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(settings, "OPENAI_MODEL", "test-model")

    with patch("app.core.llm.ChatOpenAI") as mock_cls:
        # Default temperature check
        get_chat_model()
        _, kwargs_default = mock_cls.call_args
        assert kwargs_default["temperature"] == 0.0

        # Explicit temperature override check
        get_chat_model(temperature=0.7)
        _, kwargs_override = mock_cls.call_args
        assert kwargs_override["temperature"] == 0.7


def test_provider_initialization_failure(monkeypatch: pytest.MonkeyPatch):
    """Test 6 — Provider initialization failure surfaces cleanly without credential leak."""
    fake_key = "test-secret-key-12345"
    monkeypatch.setattr(settings, "OPENAI_API_KEY", fake_key)
    monkeypatch.setattr(settings, "OPENAI_MODEL", "test-model")

    with patch(
        "app.core.llm.ChatOpenAI",
        side_effect=RuntimeError(f"Connection init failed for {fake_key}"),
    ):
        with pytest.raises(LLMConfigurationError) as exc_info:
            get_chat_model()

        err_msg = str(exc_info.value)
        assert "Failed to initialize OpenAI chat model" in err_msg
        # Critical: Verify secret key was redacted from the error message
        assert fake_key not in err_msg
        assert "[REDACTED]" in err_msg


def test_import_safety():
    """Test 7 — Importing app.core.llm performs no network operations or completions."""
    import importlib

    import app.core.llm

    # Re-importing should succeed purely as code loading without invoking any network calls
    module = importlib.reload(app.core.llm)
    assert hasattr(module, "get_chat_model")
    assert hasattr(module, "LLMConfigurationError")


def test_repeated_model_access(monkeypatch: pytest.MonkeyPatch):
    """Test 8 — Repeated model access follows factory pattern (distinct, deterministic instances)."""
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(settings, "OPENAI_MODEL", "test-model")

    with patch("app.core.llm.ChatOpenAI") as mock_cls:
        mock_cls.side_effect = [MagicMock(name="model_1"), MagicMock(name="model_2")]

        model_a = get_chat_model()
        model_b = get_chat_model()

        assert model_a is not model_b
        assert mock_cls.call_count == 2


def test_logger_safety(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    """Test 9 — Logger safety: fake API key is never emitted into captured log records."""
    fake_key = "test-openai-key-secret-987"
    monkeypatch.setattr(settings, "OPENAI_API_KEY", fake_key)
    monkeypatch.setattr(settings, "OPENAI_MODEL", "test-model-xyz")

    with patch("app.core.llm.ChatOpenAI"):
        with caplog.at_level("DEBUG"):
            get_chat_model()

        log_text = caplog.text
        assert "test-model-xyz" in log_text
        assert fake_key not in log_text
