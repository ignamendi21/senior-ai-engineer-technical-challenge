import httpx
import pytest

from magic_assistant.api.schemas import ChatResponse, RuleSource
from magic_assistant.ui.client import (
    DemoApiClient,
    DemoApiError,
    source_label,
    technical_details,
    valid_image_url,
)


def response_payload() -> dict:
    return {
        "request_id": "request-1",
        "thread_id": "thread-1",
        "answer": "Answer",
        "intent": "rules_question",
        "route_trace": ["plan_request", "retrieve_rules", "render_answer"],
        "sources": [
            {
                "type": "rule",
                "document": "Magic Comprehensive Rules",
                "version": "2026-04-17",
                "rule_id": "509.1h",
                "page_start": 111,
                "page_end": 112,
            }
        ],
        "cards": [],
        "custom_card": None,
    }


def test_response_parsing_and_technical_details_are_pure():
    response = ChatResponse.model_validate(response_payload())

    details = technical_details(response)

    assert details == {
        "intent": "rules_question",
        "route": "plan_request → retrieve_rules → render_answer",
        "sources": ["Magic Comprehensive Rules 2026-04-17 — rule 509.1h — PDF p. 111-112"],
        "request_id": "request-1",
    }


def test_source_label_and_image_validation():
    source = RuleSource(
        version="2026-04-17",
        rule_id="702.49",
        page_start=160,
        page_end=160,
    )

    assert source_label(source).endswith("rule 702.49 — PDF p. 160")
    assert valid_image_url("https://example.test/card.jpg")
    assert not valid_image_url("javascript:alert(1)")
    assert not valid_image_url(None)


def test_demo_client_uses_api_contract_without_live_backend():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        assert request.url.path == "/api/chat"
        assert request.read() == b'{"message":"question","thread_id":"thread-1"}'
        return httpx.Response(200, json=response_payload())

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = DemoApiClient("http://test", client=http_client)

    assert client.ready()
    assert client.chat("question", "thread-1").intent.value == "rules_question"
    http_client.close()


def test_demo_client_maps_backend_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = DemoApiClient("http://test", client=http_client)

    with pytest.raises(DemoApiError, match="API request failed"):
        client.chat("question", "thread-1")
    http_client.close()
