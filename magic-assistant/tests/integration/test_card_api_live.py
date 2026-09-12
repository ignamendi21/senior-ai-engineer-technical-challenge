import os

import pytest

from magic_assistant.cards.client import MtgApiClient
from magic_assistant.cards.models import CardSearchFilters
from magic_assistant.cards.service import CardSearchService

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_MTG_API_TESTS") != "1",
    reason="Set RUN_LIVE_MTG_API_TESTS=1 to call the external MTG API",
)


def test_live_card_search_smoke():
    with MtgApiClient() as client:
        cards = CardSearchService(client).search(
            CardSearchFilters(name="Black Lotus", result_limit=1, max_pages=1)
        )

    assert cards
    assert cards[0].name == "Black Lotus"
