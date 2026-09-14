from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from magic_assistant.cards.models import CardSearchFilters, MagicColor


class RequestIntent(StrEnum):
    RULES_QUESTION = "rules_question"
    CARD_SEARCH = "card_search"
    CARD_INTERACTION = "card_interaction"
    CUSTOM_CARD = "custom_card"
    OUT_OF_SCOPE = "out_of_scope"


class CustomCardRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested_name: str | None = None
    colors: list[MagicColor] = Field(default_factory=list)
    requested_mechanics: list[str] = Field(default_factory=list)
    theme: str | None = None


class RequestPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: RequestIntent
    rules_query: str | None = None
    card_names: list[str] = Field(default_factory=list)
    card_search_filters: CardSearchFilters | None = None
    custom_card_request: CustomCardRequest | None = None
    response_language: str = "en"


class GroundedAnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    used_rule_chunk_ids: list[str] = Field(default_factory=list)
    used_card_ids: list[str] = Field(default_factory=list)


class CustomCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    mana_cost: str | None = None
    colors: list[MagicColor] = Field(default_factory=list)
    type_line: str
    oracle_text: str
    power: str | None = None
    toughness: str | None = None
    flavor_text: str | None = None


class AgentError(BaseModel):
    category: str
    user_message: str
