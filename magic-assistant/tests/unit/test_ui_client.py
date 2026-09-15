import httpx
import pytest

from magic_assistant.api.schemas import ChatResponse, RuleSource
from magic_assistant.ui.client import (
    DemoApiClient,
    DemoApiError,
    response_presentation,
    source_label,
    strip_generated_source_block,
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


def test_grounded_answer_strips_only_matching_generated_trailing_sources():
    response = ChatResponse.model_validate(response_payload())
    answer = (
        "Grounded body.\n\nSources:\n"
        "- Magic Comprehensive Rules 2026-04-17 — rule 509.1h — PDF p. 111-112"
    )

    assert strip_generated_source_block(answer, response.sources) == "Grounded body."


def test_ordinary_sources_word_is_not_truncated():
    response = ChatResponse.model_validate(response_payload())
    answer = "Sources of mana are discussed here, but this is ordinary prose."

    assert strip_generated_source_block(answer, response.sources) == answer


def test_structured_card_search_suppresses_textual_enumeration():
    payload = response_payload()
    payload["intent"] = "card_search"
    payload["answer"] = "Cards found:\n- Duplicate textual card"
    payload["sources"] = []
    payload["cards"] = [
        {
            "id": "card-1",
            "name": "Test Warrior",
            "mana_cost": "{W}",
            "mana_value": 1,
            "colors": ["W"],
            "type_line": "Creature — Human Warrior",
            "oracle_text": "Vigilance",
            "set_code": "TST",
            "image_url": None,
        }
    ]
    response = ChatResponse.model_validate(payload)

    presentation = response_presentation(response)

    assert presentation.answer_text is None
    assert presentation.card_heading == "Cards found"
    assert response.cards[0].name == "Test Warrior"


def test_empty_card_search_preserves_no_results_message():
    payload = response_payload()
    payload.update(
        intent="card_search",
        answer="No cards matched those filters.",
        sources=[],
        cards=[],
    )
    response = ChatResponse.model_validate(payload)

    presentation = response_presentation(response)

    assert presentation.answer_text == "No cards matched those filters."
    assert presentation.card_heading is None


def test_custom_card_uses_one_structured_warning_and_preserves_fields():
    payload = response_payload()
    payload["intent"] = "custom_card"
    payload["answer"] = "CUSTOM / FAN-MADE — NOT AN OFFICIAL MAGIC CARD\n\nDuplicate text"
    payload["sources"] = []
    payload["custom_card"] = {
        "name": "Han Solo, Daring Captain",
        "mana_cost": "{1}{R}{W}",
        "colors": ["R", "W"],
        "type_line": "Legendary Creature — Human Rogue",
        "oracle_text": "First strike",
        "power": "3",
        "toughness": "2",
        "flavor_text": "Never tell me the odds.",
    }
    response = ChatResponse.model_validate(payload)

    presentation = response_presentation(response)

    assert presentation.show_custom_card
    assert presentation.answer_text is None
    assert response.custom_card.name == "Han Solo, Daring Captain"
    assert response.custom_card.oracle_text == "First strike"
    assert response.custom_card.flavor_text == "Never tell me the odds."


def test_source_label_and_image_validation():
    source = RuleSource(
        version="2026-04-17",
        rule_id="702.49",
        page_start=160,
        page_end=160,
    )

    assert source_label(source).endswith("rule 702.49 — PDF p. 160")
    assert valid_image_url("https://gatherer.wizards.com/Handlers/Image.ashx?id=1")
    assert not valid_image_url("http://127.0.0.1/card.jpg")
    assert not valid_image_url("http://169.254.169.254/latest/meta-data")
    assert not valid_image_url("https://user:password@gatherer.wizards.com/card.jpg")
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
