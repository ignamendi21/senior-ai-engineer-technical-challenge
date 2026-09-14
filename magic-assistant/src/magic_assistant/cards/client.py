import math
import os
import ssl
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx
import truststore
from pydantic import BaseModel, ValidationError

from magic_assistant.cards.models import Card

DEFAULT_BASE_URL = "https://api.magicthegathering.io/v1"
DEFAULT_PAGE_SIZE = 100
CA_BUNDLE_ENVIRONMENT_VARIABLE = "MTG_API_CA_BUNDLE"


class CardApiError(RuntimeError):
    pass


class CardApiConfigurationError(CardApiError):
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


def create_tls_context(ca_bundle: Path | str | None = None) -> ssl.SSLContext:
    configured_bundle = ca_bundle or os.environ.get(CA_BUNDLE_ENVIRONMENT_VARIABLE)
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    if configured_bundle:
        bundle_path = Path(configured_bundle).expanduser()
        if not bundle_path.is_file():
            raise CardApiConfigurationError(f"MTG API CA bundle not found: {bundle_path}")
        try:
            context.load_verify_locations(cafile=str(bundle_path))
        except (OSError, ssl.SSLError) as error:
            raise CardApiConfigurationError(
                f"Could not load MTG API CA bundle: {bundle_path}"
            ) from error
    return context


class MtgApiClient:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 10.0,
        max_retries: int = 2,
        backoff_seconds: float = 0.25,
        client: httpx.Client | None = None,
        tls_context: ssl.SSLContext | None = None,
        ca_bundle: Path | str | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if not math.isfinite(backoff_seconds) or backoff_seconds < 0:
            raise ValueError("backoff_seconds must be nonnegative and finite")
        if tls_context is not None and ca_bundle is not None:
            raise ValueError("Provide either tls_context or ca_bundle, not both")
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._sleep = sleep
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(timeout),
            headers={"User-Agent": "magic-assistant/0.1"},
            verify=tls_context or create_tls_context(ca_bundle),
        )

    def fetch_cards(
        self,
        filters: Mapping[str, str],
        *,
        page: int,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> CardPage:
        if page < 1:
            raise ValueError("page must be positive")
        if not 1 <= page_size <= DEFAULT_PAGE_SIZE:
            raise ValueError(f"page_size must be between 1 and {DEFAULT_PAGE_SIZE}")
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
