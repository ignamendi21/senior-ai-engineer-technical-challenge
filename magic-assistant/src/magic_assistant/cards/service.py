from collections.abc import Mapping

from magic_assistant.cards.client import DEFAULT_PAGE_SIZE, MtgApiClient
from magic_assistant.cards.models import Card, CardSearchFilters


class CardSearchService:
    def __init__(self, client: MtgApiClient) -> None:
        self._client = client

    def search(self, filters: CardSearchFilters) -> list[Card]:
        api_filters = build_api_filters(filters)
        unique_cards: dict[str, Card] = {}
        for page_number in range(1, filters.max_pages + 1):
            page = self._client.fetch_cards(
                api_filters,
                page=page_number,
                page_size=DEFAULT_PAGE_SIZE,
            )
            for card in page.cards:
                if not matches_filters(card, filters):
                    continue
                identity = normalize_card_name(card.name)
                current = unique_cards.get(identity)
                if current is None or card_quality(card) > card_quality(current):
                    unique_cards[identity] = card
            if len(unique_cards) >= filters.result_limit:
                break
            returned_count = page.count if page.count is not None else len(page.cards)
            if returned_count < DEFAULT_PAGE_SIZE or not page.cards:
                break
        return list(unique_cards.values())[: filters.result_limit]


def build_api_filters(filters: CardSearchFilters) -> dict[str, str]:
    params: dict[str, str] = {}
    direct_values: Mapping[str, str | None] = {
        "name": filters.name,
        "text": filters.text,
        "rarity": filters.rarity,
        "set": filters.set_code,
    }
    params.update({key: value for key, value in direct_values.items() if value})
    if filters.colors:
        params["colors"] = ",".join(color.value for color in filters.colors)
    if filters.color_identity:
        params["colorIdentity"] = ",".join(color.value for color in filters.color_identity)
    if filters.types:
        params["types"] = ",".join(filters.types)
    if filters.subtypes:
        params["subtypes"] = ",".join(filters.subtypes)
    if filters.mana_value is not None:
        params["cmc"] = format(filters.mana_value, "g")
    return params


def matches_filters(card: Card, filters: CardSearchFilters) -> bool:
    if filters.name and filters.name.casefold() not in card.name.casefold():
        return False
    if filters.colors and not set(filters.colors).issubset(card.colors):
        return False
    if filters.color_identity and not set(filters.color_identity).issubset(card.color_identity):
        return False
    if filters.types and not _contains_all(card.types, filters.types):
        return False
    if filters.subtypes and not _contains_all(card.subtypes, filters.subtypes):
        return False
    if filters.text and (
        card.oracle_text is None or filters.text.casefold() not in card.oracle_text.casefold()
    ):
        return False
    if filters.rarity and (
        card.rarity is None or filters.rarity.casefold() != card.rarity.casefold()
    ):
        return False
    if filters.set_code and (
        card.set_code is None or filters.set_code.casefold() != card.set_code.casefold()
    ):
        return False
    if filters.mana_value is not None and card.mana_value != filters.mana_value:
        return False
    if filters.min_mana_value is not None and (
        card.mana_value is None or card.mana_value < filters.min_mana_value
    ):
        return False
    return not (
        filters.max_mana_value_exclusive is not None
        and (card.mana_value is None or card.mana_value >= filters.max_mana_value_exclusive)
    )


def normalize_card_name(name: str) -> str:
    return " ".join(name.casefold().split())


def card_quality(card: Card) -> tuple[int, int, int, str, str]:
    return (
        int(bool(card.oracle_text)),
        int(bool(card.image_url)),
        len(card.rulings),
        (card.set_code or "").casefold(),
        card.id,
    )


def _contains_all(actual: list[str], expected: list[str]) -> bool:
    normalized = {value.casefold() for value in actual}
    return all(value.casefold() in normalized for value in expected)
