import pytest

from magic_assistant.agent.llm import AgentConfigurationError, create_openai_model_from_environment


def test_live_model_configuration_is_lazy_and_explicit(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MAGIC_CHAT_MODEL", raising=False)

    with pytest.raises(AgentConfigurationError, match="MAGIC_CHAT_MODEL"):
        create_openai_model_from_environment()


def test_live_model_requires_api_key_after_model_name(monkeypatch):
    monkeypatch.setenv("MAGIC_CHAT_MODEL", "configured-by-user")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(AgentConfigurationError, match="OPENAI_API_KEY"):
        create_openai_model_from_environment()
