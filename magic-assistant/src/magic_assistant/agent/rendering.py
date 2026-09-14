from collections.abc import Sequence

from magic_assistant.agent.schemas import CustomCard, GroundedAnswerDraft
from magic_assistant.cards.models import Card
from magic_assistant.rules.retrieval import RuleEvidence

CUSTOM_CARD_NOTICE = "CUSTOM / FAN-MADE — NOT AN OFFICIAL MAGIC CARD"
SCOPE_RESPONSE = (
    "I can help with Magic: The Gathering rules, card searches, card interactions, "
    "or custom card concepts."
)
GROUNDING_FALLBACK = (
    "I couldn't produce an answer whose sources could be validated. "
    "Please rephrase the Magic question or provide more detail."
)


def render_grounded_answer(
    draft: GroundedAnswerDraft,
    rules_evidence: Sequence[RuleEvidence],
    cards: Sequence[Card],
) -> str:
    rules_by_id = {evidence.chunk_id: evidence for evidence in rules_evidence}
    cards_by_id = {card.id: card for card in cards}
    sources = []
    seen_rule_sources: set[tuple[str, str | None]] = set()
    for source in draft.rule_sources:
        source_key = (source.chunk_id, source.rule_id)
        if source_key in seen_rule_sources:
            continue
        seen_rule_sources.add(source_key)
        evidence = rules_by_id[source.chunk_id]
        if source.rule_id:
            label = f"rule {source.rule_id}"
        elif evidence.term:
            label = f'glossary "{evidence.term}"'
        elif evidence.root_rule_id:
            label = f"rule {evidence.root_rule_id}"
        else:
            label = "rules evidence"
        pages = (
            str(evidence.page_start)
            if evidence.page_start == evidence.page_end
            else f"{evidence.page_start}-{evidence.page_end}"
        )
        sources.append(
            f"- Magic Comprehensive Rules {evidence.rules_version} — {label} — PDF p. {pages}"
        )
    for card_id in dict.fromkeys(draft.used_card_ids):
        sources.append(f"- {cards_by_id[card_id].name} — MTG card API")
    source_block = "\n".join(sources)
    return f"{draft.text.strip()}\n\nSources:\n{source_block}"


def render_card_search(cards: Sequence[Card], response_language: str) -> str:
    spanish = response_language.casefold().startswith("es")
    if not cards:
        return (
            "No encontré cartas que coincidan con esos filtros."
            if spanish
            else ("I found no cards matching those filters.")
        )
    heading = "Cartas encontradas:" if spanish else "Cards found:"
    lines = [heading]
    for card in cards:
        colors = "".join(color.value for color in card.colors) or "Colorless"
        lines.append(
            f"- {card.name} — {card.mana_cost or 'no mana cost'}; mana value "
            f"{card.mana_value}; {colors}; {card.type_line or 'Unknown type'}"
        )
        if card.oracle_text:
            lines.append(f"  {card.oracle_text}")
        if card.set_code:
            lines.append(f"  Set: {card.set_code}")
        if card.image_url:
            lines.append(f"  Image: {card.image_url}")
    return "\n".join(lines)


def render_custom_card(card: CustomCard) -> str:
    colors = "".join(color.value for color in card.colors) or "Colorless"
    lines = [
        CUSTOM_CARD_NOTICE,
        "",
        card.name,
        f"Mana cost: {card.mana_cost or 'None'}",
        f"Colors: {colors}",
        f"Type: {card.type_line}",
        card.oracle_text,
    ]
    if card.power is not None or card.toughness is not None:
        lines.append(f"Power/Toughness: {card.power or '-'}/{card.toughness or '-'}")
    if card.flavor_text:
        lines.append(f'Flavor text: "{card.flavor_text}"')
    return "\n".join(lines)
