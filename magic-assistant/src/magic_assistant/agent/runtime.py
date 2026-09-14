import os
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from magic_assistant.agent.graph import (
    AssistantDependencies,
    MagicAssistant,
    build_assistant_graph,
)
from magic_assistant.agent.llm import (
    OpenAICustomCardGenerator,
    OpenAIGroundedAnswerGenerator,
    OpenAIRequestPlanner,
    create_openai_model_from_environment,
)
from magic_assistant.agent.state import AssistantState
from magic_assistant.cards.client import MtgApiClient
from magic_assistant.cards.service import CardSearchService
from magic_assistant.rules.search import load_knowledge_base


class RuntimeInitializationError(RuntimeError):
    pass


class RuntimeSettings(BaseModel):
    rules_pdf: Path = Path("data/MagicCompRules 20260417.pdf")
    index_directory: Path = Path("data/index")

    @classmethod
    def from_environment(cls) -> "RuntimeSettings":
        return cls(
            rules_pdf=Path(os.environ.get("MAGIC_RULES_PDF", "data/MagicCompRules 20260417.pdf")),
            index_directory=Path(os.environ.get("MAGIC_INDEX_DIR", "data/index")),
        )


class AssistantRuntime(Protocol):
    @property
    def ready(self) -> bool: ...

    def invoke(self, message: str, *, thread_id: str) -> AssistantState: ...

    def close(self) -> None: ...


class LiveAssistantRuntime:
    def __init__(self, assistant: MagicAssistant, card_client: MtgApiClient) -> None:
        self._assistant = assistant
        self._card_client = card_client
        self._closed = False

    @property
    def ready(self) -> bool:
        return not self._closed

    @classmethod
    def create(cls, settings: RuntimeSettings | None = None) -> "LiveAssistantRuntime":
        settings = settings or RuntimeSettings.from_environment()
        if not settings.rules_pdf.is_file():
            location = settings.rules_pdf
            raise RuntimeInitializationError(
                f"Rules PDF not found at {location}. Place the PDF or set MAGIC_RULES_PDF."
            )
        if not (settings.index_directory / "manifest.json").is_file():
            raise RuntimeInitializationError(
                "Rules index is missing. Run `python -m magic_assistant.rules.index` first."
            )

        card_client = MtgApiClient()
        try:
            model = create_openai_model_from_environment()
            knowledge_base = load_knowledge_base(
                settings.rules_pdf,
                settings.index_directory,
            )
            assistant = MagicAssistant(
                build_assistant_graph(
                    AssistantDependencies(
                        planner=OpenAIRequestPlanner(model),
                        answer_generator=OpenAIGroundedAnswerGenerator(model),
                        custom_card_generator=OpenAICustomCardGenerator(model),
                        rules_retriever=knowledge_base,
                        card_searcher=CardSearchService(card_client),
                    )
                )
            )
        except Exception as error:
            card_client.close()
            if isinstance(error, RuntimeInitializationError):
                raise
            raise RuntimeInitializationError(
                "Could not initialize the assistant runtime. Check model and index configuration."
            ) from error
        return cls(assistant, card_client)

    def invoke(self, message: str, *, thread_id: str) -> AssistantState:
        if self._closed:
            raise RuntimeInitializationError("Assistant runtime is closed")
        return self._assistant.invoke(message, thread_id=thread_id)

    def close(self) -> None:
        if not self._closed:
            self._card_client.close()
            self._closed = True

    def __enter__(self) -> "LiveAssistantRuntime":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
