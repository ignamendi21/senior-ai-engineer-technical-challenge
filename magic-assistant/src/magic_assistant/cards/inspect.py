import argparse
from collections.abc import Sequence

from magic_assistant.cards.client import MtgApiClient
from magic_assistant.cards.models import CardSearchFilters, MagicColor
from magic_assistant.cards.service import CardSearchService


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect deterministic MTG card search")
    parser.add_argument("--name")
    parser.add_argument("--color", action="append", choices=[color.value for color in MagicColor])
    parser.add_argument("--type", dest="types", action="append")
    parser.add_argument("--subtype", dest="subtypes", action="append")
    parser.add_argument("--text")
    parser.add_argument("--cmc", type=float)
    parser.add_argument("--min-cmc", type=float)
    parser.add_argument("--max-cmc", type=float)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-pages", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    filters = CardSearchFilters(
        name=arguments.name,
        colors=[MagicColor(color) for color in arguments.color or []],
        types=arguments.types or [],
        subtypes=arguments.subtypes or [],
        text=arguments.text,
        mana_value=arguments.cmc,
        min_mana_value=arguments.min_cmc,
        max_mana_value_exclusive=arguments.max_cmc,
        result_limit=arguments.limit,
        max_pages=arguments.max_pages,
    )
    with MtgApiClient() as client:
        cards = CardSearchService(client).search(filters)
    for index, card in enumerate(cards, 1):
        colors = "".join(color.value for color in card.colors) or "colorless"
        print(
            f"{index}. {card.name} | mana value={card.mana_value} | "
            f"colors={colors} | {card.type_line or 'Unknown type'}"
        )
        if card.oracle_text:
            print(f"   {' '.join(card.oracle_text.split())[:240]}")
        if card.image_url:
            print(f"   image={card.image_url}")
    print(f"Unique cards: {len(cards)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
