import os
from collections.abc import Sequence
from typing import Protocol, cast

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from magic_assistant.agent.schemas import (
    CustomCardDraft,
    CustomCardRequest,
    GroundedAnswerDraft,
    RequestPlan,
)
from magic_assistant.cards.models import Card
from magic_assistant.rules.retrieval import RuleEvidence


class AgentConfigurationError(ValueError):
    pass


class RequestPlanner(Protocol):
    def plan(self, messages: Sequence[BaseMessage]) -> RequestPlan: ...


class GroundedAnswerGenerator(Protocol):
    def generate(
        self,
        *,
        question: str,
        messages: Sequence[BaseMessage],
        response_language: str,
        rules_evidence: Sequence[RuleEvidence],
        cards: Sequence[Card],
        validation_feedback: str | None,
    ) -> GroundedAnswerDraft: ...


class CustomCardGenerator(Protocol):
    def generate(
        self,
        *,
        request: CustomCardRequest,
        original_question: str,
        rules_evidence: Sequence[RuleEvidence],
        response_language: str,
        validation_feedback: str | None,
    ) -> CustomCardDraft: ...


class OpenAIRequestPlanner:
    def __init__(self, model: ChatOpenAI) -> None:
        self._planner = cast(
            Runnable,
            model.with_structured_output(RequestPlan, method="json_schema", strict=False),
        )

    def plan(self, messages: Sequence[BaseMessage]) -> RequestPlan:
        prompt = SystemMessage(
            content=(
                "Classify and extract the user's Magic request. Do not answer it. "
                "Use domain-level card filters only; never create URLs or API parameters. "
                "Use card_interaction when named cards and rules must be reasoned about together. "
                "Use out_of_scope for requests unrelated to Magic rules, cards, or fan-made cards."
            )
        )
        return cast(RequestPlan, self._planner.invoke([prompt, *messages]))


class OpenAIGroundedAnswerGenerator:
    def __init__(self, model: ChatOpenAI) -> None:
        self._generator = cast(
            Runnable,
            model.with_structured_output(GroundedAnswerDraft, method="json_schema", strict=False),
        )

    def generate(
        self,
        *,
        question: str,
        messages: Sequence[BaseMessage],
        response_language: str,
        rules_evidence: Sequence[RuleEvidence],
        cards: Sequence[Card],
        validation_feedback: str | None,
    ) -> GroundedAnswerDraft:
        rules = "\n".join(evidence.model_dump_json() for evidence in rules_evidence)
        card_data = "\n".join(card.model_dump_json() for card in cards)
        conversation = "\n".join(f"{message.type}: {message.content}" for message in messages[-8:])
        feedback = validation_feedback or "None"
        prompt = SystemMessage(
            content=(
                "Answer only from the supplied rule and card evidence. Distinguish rules from "
                "Oracle text, be concise, and answer in the requested language. Do not write "
                "source notation. Select used IDs only from the supplied evidence. If evidence "
                "is insufficient, state that rather than inventing facts."
            )
        )
        request = HumanMessage(
            content=(
                f"Question: {question}\nLanguage: {response_language}\n"
                f"Conversation:\n{conversation}\nValidation feedback: {feedback}\n"
                f"Rule evidence:\n{rules}\nCards:\n{card_data}"
            )
        )
        return cast(GroundedAnswerDraft, self._generator.invoke([prompt, request]))


class OpenAICustomCardGenerator:
    def __init__(self, model: ChatOpenAI) -> None:
        self._generator = cast(
            Runnable,
            model.with_structured_output(CustomCardDraft, method="json_schema", strict=False),
        )

    def generate(
        self,
        *,
        request: CustomCardRequest,
        original_question: str,
        rules_evidence: Sequence[RuleEvidence],
        response_language: str,
        validation_feedback: str | None,
    ) -> CustomCardDraft:
        mechanics = "\n".join(evidence.model_dump_json() for evidence in rules_evidence)
        feedback = validation_feedback or "None"
        prompt = SystemMessage(
            content=(
                "Create a balanced fan-made Magic card using the structured request and supplied "
                "mechanic rules. Use standard Oracle-style wording where evidence supports it. "
                "Select used rule chunk IDs only from the supplied mechanic evidence."
            )
        )
        user = HumanMessage(
            content=(
                f"Original request: {original_question}\nLanguage: {response_language}\n"
                f"Validation feedback: {feedback}\n"
                f"Extracted request: {request.model_dump_json()}\nMechanic evidence:\n{mechanics}"
            )
        )
        return cast(CustomCardDraft, self._generator.invoke([prompt, user]))


def create_openai_model_from_environment() -> ChatOpenAI:
    model_name = os.environ.get("MAGIC_CHAT_MODEL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not model_name:
        raise AgentConfigurationError("MAGIC_CHAT_MODEL is required for a live model run")
    if not api_key:
        raise AgentConfigurationError("OPENAI_API_KEY is required for a live model run")
    base_url = os.environ.get("OPENAI_BASE_URL")
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )
