from urllib.parse import urlparse
from uuid import uuid4

import httpx
from pydantic import BaseModel

from magic_assistant.agent.schemas import RequestIntent
from magic_assistant.api.schemas import (
    CardSource,
    ChatResponse,
    GlossarySource,
    PublicSource,
    RuleSource,
)

_ALLOWED_IMAGE_HOSTS = {"gatherer.wizards.com"}


class DemoApiError(RuntimeError):
    pass


class ResponsePresentation(BaseModel):
    answer_text: str | None
    card_heading: str | None
    show_custom_card: bool


class DemoApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout)

    def ready(self) -> bool:
        try:
            response = self._client.get(f"{self._base_url}/ready")
            return response.status_code == 200 and response.json().get("status") == "ready"
        except (httpx.RequestError, ValueError):
            return False

    def chat(self, message: str, thread_id: str) -> ChatResponse:
        try:
            response = self._client.post(
                f"{self._base_url}/api/chat",
                json={"message": message, "thread_id": thread_id},
            )
            response.raise_for_status()
            return ChatResponse.model_validate(response.json())
        except (httpx.HTTPError, ValueError) as error:
            raise DemoApiError("The Magic Assistant API request failed") from error

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


def new_thread_id() -> str:
    return str(uuid4())


def valid_image_url(value: str | None) -> bool:
    if not value:
        return False
    parsed = urlparse(value)
    try:
        valid_port = parsed.port in {None, 80, 443}
    except ValueError:
        return False
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname in _ALLOWED_IMAGE_HOSTS
        and parsed.username is None
        and parsed.password is None
        and valid_port
    )


def response_presentation(response: ChatResponse) -> ResponsePresentation:
    if response.custom_card is not None:
        return ResponsePresentation(
            answer_text=None,
            card_heading=None,
            show_custom_card=True,
        )
    if response.intent == RequestIntent.CARD_SEARCH and response.cards:
        return ResponsePresentation(
            answer_text=None,
            card_heading="Cards found",
            show_custom_card=False,
        )
    answer = strip_generated_source_block(response.answer, response.sources)
    return ResponsePresentation(
        answer_text=answer,
        card_heading=None,
        show_custom_card=False,
    )


def strip_generated_source_block(answer: str, sources: list[PublicSource]) -> str:
    if not sources:
        return answer
    expected_block = "\n\nSources:\n" + "\n".join(
        f"- {_rendered_answer_source_label(source)}" for source in sources
    )
    return answer.removesuffix(expected_block).rstrip()


def source_label(source: PublicSource) -> str:
    if isinstance(source, RuleSource):
        pages = _pages(source.page_start, source.page_end)
        return (
            f"Magic Comprehensive Rules {source.version} — rule {source.rule_id} — PDF p. {pages}"
        )
    if isinstance(source, GlossarySource):
        pages = _pages(source.page_start, source.page_end)
        prefix = f'Magic Comprehensive Rules {source.version} — glossary "{source.term}"'
        return f"{prefix} — PDF p. {pages}"
    if isinstance(source, CardSource):
        return f"{source.name} — {source.provider}"
    raise TypeError("Unsupported public source")


def technical_details(response: ChatResponse) -> dict[str, object]:
    return {
        "intent": response.intent.value,
        "route": " → ".join(response.route_trace),
        "sources": [source_label(source) for source in response.sources],
        "request_id": response.request_id,
    }


def _rendered_answer_source_label(source: PublicSource) -> str:
    if isinstance(source, CardSource):
        return f"{source.name} — MTG card API"
    return source_label(source)


def _pages(page_start: int, page_end: int) -> str:
    return str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
