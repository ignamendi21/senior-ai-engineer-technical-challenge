import time
from collections.abc import Callable, Mapping
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from magic_assistant.cards.models import Card

DEFAULT_BASE_URL = "https://api.magicthegathering.io/v1"
DEFAULT_PAGE_SIZE = 100


class CardApiError(RuntimeError):
    pass


class CardApiConnectionError(CardApiError):
    pass


class CardApiRequestError(CardApiError):
    pass


class CardApiNotFoundError(CardApiError):
    pass


class CardApiServerError(CardApiError):
    pass


class CardApiPayloadError(CardApiError):
    pass


class CardApiRateLimitError(CardApiError):
    def __init__(self, message: str, rate_limit: "RateLimitInfo") -> None:
        super().__init__(message)
        self.rate_limit = rate_limit


class RateLimitInfo(BaseModel):
    limit: int | None = None
    remaining: int | None = None
    reset: str | None = None


class CardPage(BaseModel):
    cards: list[Card]
    total_count: int | None = None
    page_size: int | None = None
    count: int | None = None
    rate_limit: RateLimitInfo


class MtgApiClient:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 10.0,
        max_retries: int = 2,
        backoff_seconds: float = 0.25,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._sleep = sleep
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(timeout),
            headers={"User-Agent": "magic-assistant/0.1"},
        )

    def fetch_cards(
        self,
        filters: Mapping[str, str],
        *,
        page: int,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> CardPage:
        params = {**filters, "page": str(page), "pageSize": str(page_size)}
        response = self._request("/cards", params)
        rate_limit = self._rate_limit(response.headers)
        try:
            payload: Any = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("cards"), list):
                raise ValueError("payload must contain a cards list")
            cards = [Card.model_validate(card) for card in payload["cards"]]
        except (ValueError, ValidationError) as error:
            raise CardApiPayloadError("MTG API returned a malformed cards payload") from error
        return CardPage(
            cards=cards,
            total_count=self._integer_header(response.headers, "total-count"),
            page_size=self._integer_header(response.headers, "page-size"),
            count=self._integer_header(response.headers, "count"),
            rate_limit=rate_limit,
        )

    def _request(self, path: str, params: Mapping[str, str]) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.get(f"{self._base_url}{path}", params=params)
            except httpx.RequestError as error:
                if attempt < self._max_retries:
                    self._sleep(self._backoff_seconds * (2**attempt))
                    continue
                raise CardApiConnectionError("Could not connect to the MTG API") from error

            if response.status_code >= 500:
                if attempt < self._max_retries:
                    self._sleep(self._backoff_seconds * (2**attempt))
                    continue
                raise CardApiServerError(
                    f"MTG API failed after retries with status {response.status_code}"
                )
            if response.status_code == 400:
                raise CardApiRequestError("MTG API rejected the card search request")
            if response.status_code == 403:
                raise CardApiRateLimitError(
                    "MTG API denied the request or rate limit was exceeded",
                    self._rate_limit(response.headers),
                )
            if response.status_code == 404:
                raise CardApiNotFoundError("MTG API cards endpoint was not found")
            if response.status_code >= 400:
                raise CardApiRequestError(
                    f"MTG API card search failed with status {response.status_code}"
                )
            return response
        raise CardApiServerError("MTG API request exhausted retries")

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "MtgApiClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @staticmethod
    def _integer_header(headers: httpx.Headers, name: str) -> int | None:
        value = headers.get(name)
        try:
            return int(value) if value is not None else None
        except ValueError:
            return None

    @classmethod
    def _rate_limit(cls, headers: httpx.Headers) -> RateLimitInfo:
        return RateLimitInfo(
            limit=cls._integer_header(headers, "ratelimit-limit"),
            remaining=cls._integer_header(headers, "ratelimit-remaining"),
            reset=headers.get("ratelimit-reset"),
        )
