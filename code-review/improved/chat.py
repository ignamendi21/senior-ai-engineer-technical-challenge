from collections.abc import Sequence
from typing import Protocol

from openai import OpenAI, OpenAIError

from improved.errors import GenerationError
from improved.models import GeneratedAnswer, PromptMessage


class ChatProvider(Protocol):
    def generate(self, messages: Sequence[PromptMessage]) -> GeneratedAnswer: ...


class OpenAIChatProvider:
    def __init__(self, client: OpenAI, model_name: str) -> None:
        self._client = client
        self._model_name = model_name

    def generate(self, messages: Sequence[PromptMessage]) -> GeneratedAnswer:
        if not messages or messages[0].role != "system":
            raise ValueError("A fixed system message is required")
        try:
            response = self._client.responses.parse(
                model=self._model_name,
                instructions=messages[0].content,
                input=[message.model_dump() for message in messages[1:]],
                text_format=GeneratedAnswer,
            )
        except OpenAIError as error:
            raise GenerationError("Chat provider request failed") from error
        if response.output_parsed is None:
            raise GenerationError("Chat provider did not return a structured answer")
        return response.output_parsed
