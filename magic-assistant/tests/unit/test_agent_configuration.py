import pytest

from magic_assistant.agent.llm import (
    AgentConfigurationError,
    OpenAICustomCardGenerator,
    OpenAIGroundedAnswerGenerator,
    OpenAIRequestPlanner,
    create_openai_model_from_environment,
)
from magic_assistant.agent.schemas import CustomCardDraft, GroundedAnswerDraft, RequestPlan


class SpyModel:
    def __init__(self) -> None:
        self.calls = []

    def with_structured_output(self, schema, **kwargs):
        self.calls.append((schema, kwargs))
        return object()


def test_openai_adapters_use_pydantic_json_schema_without_incompatible_strict_mode():
    model = SpyModel()

    OpenAIRequestPlanner(model)
    OpenAIGroundedAnswerGenerator(model)
    OpenAICustomCardGenerator(model)

    assert model.calls == [
        (RequestPlan, {"method": "json_schema", "strict": False}),
        (GroundedAnswerDraft, {"method": "json_schema", "strict": False}),
        (CustomCardDraft, {"method": "json_schema", "strict": False}),
    ]


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
