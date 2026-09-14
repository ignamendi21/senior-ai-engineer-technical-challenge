from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from magic_assistant.agent.schemas import CustomCard, RequestIntent, RuleSourceRef
from magic_assistant.agent.state import AssistantState
from magic_assistant.cards.models import Card, MagicColor
from magic_assistant.rules.retrieval import RuleEvidence


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=10_000)
    thread_id: str = Field(min_length=1, max_length=128)

    @field_validator("message", "thread_id")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class RuleSource(BaseModel):
    type: Literal["rule"] = "rule"
    document: Literal["Magic Comprehensive Rules"] = "Magic Comprehensive Rules"
    version: str
    rule_id: str
    page_start: int
    page_end: int


class GlossarySource(BaseModel):
    type: Literal["glossary"] = "glossary"
    term: str
    version: str
    page_start: int
    page_end: int


class CardSource(BaseModel):
    type: Literal["card"] = "card"
    card_id: str
    name: str
    provider: Literal["MTG API"] = "MTG API"


PublicSource = Annotated[RuleSource | GlossarySource | CardSource, Field(discriminator="type")]


class PublicCard(BaseModel):
    id: str
    name: str
    mana_cost: str | None
    mana_value: float | None
    colors: list[MagicColor]
    type_line: str | None
    oracle_text: str | None
    set_code: str | None
    image_url: str | None

    @classmethod
    def from_card(cls, card: Card) -> "PublicCard":
        return cls(
            id=card.id,
            name=card.name,
            mana_cost=card.mana_cost,
            mana_value=card.mana_value,
            colors=card.colors,
            type_line=card.type_line,
            oracle_text=card.oracle_text,
            set_code=card.set_code,
            image_url=card.image_url,
        )


class ChatResponse(BaseModel):
    request_id: str
    thread_id: str
    answer: str
    intent: RequestIntent
    route_trace: list[str]
    sources: list[PublicSource]
    cards: list[PublicCard]
    custom_card: CustomCard | None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadinessResponse(BaseModel):
    status: Literal["ready", "unavailable"]
    detail: str | None = None


def build_chat_response(
    state: AssistantState,
    *,
    thread_id: str,
    request_id: str,
) -> ChatResponse:
    plan = state.get("request_plan")
    if plan is None:
        raise ValueError("Assistant response does not contain a request plan")
    rules_by_id = {evidence.chunk_id: evidence for evidence in state.get("rules_evidence", [])}
    cards_by_id = {card.id: card for card in state.get("cards", [])}
    rule_refs, used_card_ids = _selected_source_ids(state)
    sources: list[PublicSource] = []
    seen: set[tuple[str, str]] = set()
    for reference in rule_refs:
        evidence = rules_by_id.get(reference.chunk_id)
        if evidence is None:
            continue
        source = _rule_public_source(evidence, reference)
        key = (source.type, reference.chunk_id + (reference.rule_id or ""))
        if key not in seen:
            sources.append(source)
            seen.add(key)
    for card_id in used_card_ids:
        card = cards_by_id.get(card_id)
        if card and ("card", card.id) not in seen:
            sources.append(CardSource(card_id=card.id, name=card.name))
            seen.add(("card", card.id))
    return ChatResponse(
        request_id=request_id,
        thread_id=thread_id,
        answer=state["final_answer"],
        intent=plan.intent,
        route_trace=state.get("route_trace", []),
        sources=sources,
        cards=[PublicCard.from_card(card) for card in state.get("cards", [])],
        custom_card=state.get("custom_card"),
    )


def _selected_source_ids(state: AssistantState) -> tuple[list[RuleSourceRef], list[str]]:
    draft = state.get("draft_answer")
    if draft and state.get("source_validation_passed"):
        return draft.rule_sources, list(dict.fromkeys(draft.used_card_ids))
    custom_draft = state.get("custom_card_draft")
    if custom_draft and state.get("source_validation_passed"):
        return (
            [RuleSourceRef(chunk_id=chunk_id) for chunk_id in custom_draft.used_rule_chunk_ids],
            [],
        )
    plan = state.get("request_plan")
    if plan and plan.intent == RequestIntent.CARD_SEARCH:
        return [], [card.id for card in state.get("cards", [])]
    return [], []


def _rule_public_source(
    evidence: RuleEvidence, reference: RuleSourceRef
) -> RuleSource | GlossarySource:
    if reference.rule_id:
        return RuleSource(
            version=evidence.rules_version,
            rule_id=reference.rule_id,
            page_start=evidence.page_start,
            page_end=evidence.page_end,
        )
    if evidence.term:
        return GlossarySource(
            term=evidence.term,
            version=evidence.rules_version,
            page_start=evidence.page_start,
            page_end=evidence.page_end,
        )
    if not evidence.root_rule_id:
        raise ValueError("Rule evidence has no precise rule or root identifier")
    return RuleSource(
        version=evidence.rules_version,
        rule_id=evidence.root_rule_id,
        page_start=evidence.page_start,
        page_end=evidence.page_end,
    )
