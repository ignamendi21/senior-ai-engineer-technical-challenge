import httpx
import pytest

from magic_assistant.cards.client import (
    CardApiConnectionError,
    CardApiNotFoundError,
    CardApiPayloadError,
    CardApiRateLimitError,
    CardApiRequestError,
    CardApiServerError,
    MtgApiClient,
)
from magic_assistant.cards.models import MagicColor


def card_payload() -> dict:
    return {
        "cards": [
            {
                "id": "card-1",
                "name": "Honored Crop-Captain",
                "manaCost": "{R}{W}",
                "cmc": 2,
                "colors": ["White", "Red"],
                "colorIdentity": ["W", "R"],
                "type": "Creature — Human Warrior",
                "types": ["Creature"],
                "subtypes": ["Human", "Warrior"],
                "text": "Whenever this creature attacks, other attacking creatures get +1/+0.",
                "imageUrl": "https://example.test/card.jpg",
                "set": "AKH",
                "rulings": [{"date": "2017-04-18", "text": "A ruling."}],
            }
        ]
    }


def make_client(handler, *, max_retries: int = 0, sleep=lambda _: None):
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    api_client = MtgApiClient(client=http_client, max_retries=max_retries, sleep=sleep)
    return api_client, http_client


def test_builds_pagination_parameters_and_normalizes_response():
    captured_request = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        return httpx.Response(
            200,
            json=card_payload(),
            headers={
                "Total-Count": "1",
                "Page-Size": "100",
                "Count": "1",
                "Ratelimit-Limit": "5000",
                "Ratelimit-Remaining": "4999",
            },
        )

    client, http_client = make_client(handler)
    page = client.fetch_cards({"name": "Crop-Captain"}, page=2, page_size=100)
    http_client.close()

    assert captured_request.url.params["name"] == "Crop-Captain"
    assert captured_request.url.params["page"] == "2"
    assert captured_request.url.params["pageSize"] == "100"
    card = page.cards[0]
    assert card.mana_cost == "{R}{W}"
    assert card.mana_value == 2
    assert card.colors == [MagicColor.WHITE, MagicColor.RED]
    assert card.oracle_text.startswith("Whenever this creature attacks")
    assert card.rulings[0].text == "A ruling."
    assert page.rate_limit.limit == 5000
    assert page.rate_limit.remaining == 4999


@pytest.mark.parametrize(
    ("status", "error_type"),
    [(400, CardApiRequestError), (404, CardApiNotFoundError)],
)
def test_does_not_retry_deterministic_client_errors(status, error_type):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, request=request)

    client, http_client = make_client(handler, max_retries=2)
    with pytest.raises(error_type):
        client.fetch_cards({}, page=1)
    http_client.close()

    assert calls == 1


def test_surfaces_rate_limit_metadata_on_403():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            request=request,
            headers={"Ratelimit-Limit": "5000", "Ratelimit-Remaining": "0"},
        )

    client, http_client = make_client(handler)
    with pytest.raises(CardApiRateLimitError) as captured:
        client.fetch_cards({}, page=1)
    http_client.close()

    assert captured.value.rate_limit.limit == 5000
    assert captured.value.rate_limit.remaining == 0


def test_retries_transient_server_failure():
    calls = 0
    sleeps = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(200, request=request, json={"cards": []})

    client, http_client = make_client(handler, max_retries=1, sleep=sleeps.append)
    page = client.fetch_cards({}, page=1)
    http_client.close()

    assert page.cards == []
    assert calls == 2
    assert sleeps == [0.25]


def test_retries_network_failure_then_raises():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out", request=request)

    client, http_client = make_client(handler, max_retries=1)
    with pytest.raises(CardApiConnectionError, match="connect"):
        client.fetch_cards({}, page=1)
    http_client.close()

    assert calls == 2


def test_exhausted_server_retries_raise_clear_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, request=request)

    client, http_client = make_client(handler, max_retries=1)
    with pytest.raises(CardApiServerError, match="after retries"):
        client.fetch_cards({}, page=1)
    http_client.close()


def test_malformed_payload_raises_domain_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, json={"unexpected": []})

    client, http_client = make_client(handler)
    with pytest.raises(CardApiPayloadError, match="malformed"):
        client.fetch_cards({}, page=1)
    http_client.close()
