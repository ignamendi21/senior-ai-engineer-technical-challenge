from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MagicColor(StrEnum):
    WHITE = "W"
    BLUE = "U"
    BLACK = "B"
    RED = "R"
    GREEN = "G"


_API_COLOR_CODES = {
    "white": MagicColor.WHITE,
    "blue": MagicColor.BLUE,
    "black": MagicColor.BLACK,
    "red": MagicColor.RED,
    "green": MagicColor.GREEN,
}
_API_COLOR_CODES.update({color.value.casefold(): color for color in MagicColor})


class CardRuling(BaseModel):
    date: str | None = None
    text: str


class CardLegality(BaseModel):
    format: str
    legality: str


class Card(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    mana_cost: str | None = Field(default=None, alias="manaCost")
    mana_value: float | None = Field(default=None, alias="cmc")
    colors: list[MagicColor] = Field(default_factory=list)
    color_identity: list[MagicColor] = Field(default_factory=list, alias="colorIdentity")
    type_line: str | None = Field(default=None, alias="type")
    supertypes: list[str] = Field(default_factory=list)
    types: list[str] = Field(default_factory=list)
    subtypes: list[str] = Field(default_factory=list)
    oracle_text: str | None = Field(default=None, alias="text")
    power: str | None = None
    toughness: str | None = None
    loyalty: str | None = None
    image_url: str | None = Field(default=None, alias="imageUrl")
    set_code: str | None = Field(default=None, alias="set")
    rarity: str | None = None
    rulings: list[CardRuling] = Field(default_factory=list)
    legalities: list[CardLegality] = Field(default_factory=list)

    @field_validator("colors", "color_identity", mode="before")
    @classmethod
    def normalize_colors(cls, values: Any) -> list[MagicColor]:
        if values is None:
            return []
        try:
            return [_API_COLOR_CODES[str(value).casefold()] for value in values]
        except KeyError as error:
            raise ValueError(f"Unknown Magic color: {error.args[0]}") from error


class CardSearchFilters(BaseModel):
    name: str | None = None
    colors: list[MagicColor] = Field(default_factory=list)
    color_identity: list[MagicColor] = Field(default_factory=list)
    types: list[str] = Field(default_factory=list)
    subtypes: list[str] = Field(default_factory=list)
    text: str | None = None
    rarity: str | None = None
    set_code: str | None = None
    mana_value: float | None = Field(default=None, ge=0)
    min_mana_value: float | None = Field(default=None, ge=0)
    max_mana_value_exclusive: float | None = Field(default=None, gt=0)
    result_limit: int = Field(default=20, ge=1, le=100)
    max_pages: int = Field(default=5, ge=1, le=20)

    @field_validator("colors", "color_identity")
    @classmethod
    def deduplicate_colors(cls, values: list[MagicColor]) -> list[MagicColor]:
        return list(dict.fromkeys(values))

    @model_validator(mode="after")
    def validate_mana_value_range(self) -> "CardSearchFilters":
        if (
            self.min_mana_value is not None
            and self.max_mana_value_exclusive is not None
            and self.min_mana_value >= self.max_mana_value_exclusive
        ):
            raise ValueError("min_mana_value must be lower than max_mana_value_exclusive")
        return self
