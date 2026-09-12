import pytest

from magic_assistant.cards.client import CardPage, RateLimitInfo
from magic_assistant.cards.models import Card, CardRuling, CardSearchFilters, MagicColor
from magic_assistant.cards.service import (
    CardSearchService,
    build_api_filters,
    matches_filters,
)


def make_card(
    identifier: str,
    name: str,
    mana_value: float,
    *,
    colors: list[MagicColor] | None = None,
    types: list[str] | None = None,
    subtypes: list[str] | None = None,
    oracle_text: str | None = None,
    image_url: str | None = None,
    rulings: list[CardRuling] | None = None,
) -> Card:
    return Card(
        id=identifier,
        name=name,
        mana_value=mana_value,
        colors=colors or [],
        color_identity=colors or [],
        type_line="Creature — Human Warrior",
        types=types or ["Creature"],
        subtypes=subtypes or ["Human", "Warrior"],
        oracle_text=oracle_text,
        image_url=image_url,
        rulings=rulings or [],
    )


class FakeCardClient:
    def __init__(self, pages: list[CardPage]) -> None:
        self.pages = pages
        self.calls = []

    def fetch_cards(self, filters, *, page, page_size):
        self.calls.append((filters, page, page_size))
        return self.pages[page - 1]


def card_page(cards: list[Card], *, count: int | None = None) -> CardPage:
    return CardPage(
        cards=cards,
        count=len(cards) if count is None else count,
        rate_limit=RateLimitInfo(),
    )


def warrior_filters(**updates) -> CardSearchFilters:
    values = {
        "colors": [MagicColor.WHITE],
        "types": ["Creature"],
        "subtypes": ["Warrior"],
        "max_mana_value_exclusive": 2,
        "result_limit": 10,
    }
    values.update(updates)
    return CardSearchFilters(**values)


def test_translates_supported_filters_to_api_parameters():
    filters = CardSearchFilters(
        name="guard",
        colors=[MagicColor.WHITE, MagicColor.BLUE],
        color_identity=[MagicColor.WHITE],
        types=["Creature"],
        subtypes=["Warrior"],
        text="vigilance",
        rarity="rare",
        set_code="khm",
        mana_value=2,
    )

    assert build_api_filters(filters) == {
        "name": "guard",
        "colors": "W,U",
        "colorIdentity": "W",
        "types": "Creature",
        "subtypes": "Warrior",
        "text": "vigilance",
        "rarity": "rare",
        "set": "khm",
        "cmc": "2",
    }


def test_rejects_contradictory_or_unknown_filters():
    with pytest.raises(ValueError, match="configured mana-value range"):
        CardSearchFilters(mana_value=5, max_mana_value_exclusive=2)
    with pytest.raises(ValueError, match="Extra inputs"):
        CardSearchFilters(max_mana_value=2)


def test_client_side_filter_enforces_exclusive_mana_range_and_color():
    filters = warrior_filters()
    matching = make_card("1", "One-Mana Warrior", 1, colors=[MagicColor.WHITE])
    too_expensive = make_card("2", "Two-Mana Warrior", 2, colors=[MagicColor.WHITE])
    wrong_color = make_card("3", "Black Warrior", 1, colors=[MagicColor.BLACK])

    assert matches_filters(matching, filters)
    assert not matches_filters(too_expensive, filters)
    assert not matches_filters(wrong_color, filters)


def test_deduplicates_printings_and_keeps_richer_representative():
    sparse = make_card("old", "Warrior Example", 1, colors=[MagicColor.WHITE])
    rich = make_card(
        "new",
        "  warrior   example ",
        1,
        colors=[MagicColor.WHITE],
        oracle_text="Vigilance",
        image_url="https://example.test/image.jpg",
        rulings=[CardRuling(date="2020-01-01", text="A ruling")],
    )
    client = FakeCardClient([card_page([sparse, rich])])

    cards = CardSearchService(client).search(warrior_filters())

    assert cards == [rich]


def test_paginates_until_enough_unique_matches_are_found():
    nonmatching = make_card("0", "Wrong Color", 1, colors=[MagicColor.BLACK])
    first = make_card("1", "First Warrior", 1, colors=[MagicColor.WHITE])
    second = make_card("2", "Second Warrior", 1, colors=[MagicColor.WHITE])
    client = FakeCardClient(
        [
            card_page([nonmatching], count=100),
            card_page([first], count=100),
            card_page([second], count=1),
        ]
    )

    cards = CardSearchService(client).search(warrior_filters(result_limit=2, max_pages=3))

    assert [card.name for card in cards] == ["First Warrior", "Second Warrior"]
    assert [call[1] for call in client.calls] == [1, 2, 3]


def test_uses_total_count_to_avoid_an_extra_page():
    nonmatching = make_card("0", "Wrong Color", 1, colors=[MagicColor.BLACK])
    page = card_page([nonmatching], count=100)
    page.total_count = 100
    page.page_size = 100
    client = FakeCardClient([page])

    cards = CardSearchService(client).search(warrior_filters(max_pages=2))

    assert cards == []
    assert len(client.calls) == 1


def test_stops_at_configured_maximum_pages():
    nonmatching = make_card("0", "Wrong Color", 1, colors=[MagicColor.BLACK])
    client = FakeCardClient([card_page([nonmatching], count=100)] * 2)

    cards = CardSearchService(client).search(warrior_filters(max_pages=2))

    assert cards == []
    assert len(client.calls) == 2
