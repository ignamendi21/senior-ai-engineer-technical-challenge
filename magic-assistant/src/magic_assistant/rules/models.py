from typing import Literal

from pydantic import BaseModel, Field


class RuleDocument(BaseModel):
    rule_id: str
    parent_rule_id: str | None = None
    chapter_id: str | None = None
    chapter_title: str | None = None
    section_id: str | None = None
    section_title: str | None = None
    title: str | None = None
    text: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    rules_version: str
    related_rule_ids: list[str] = Field(default_factory=list)
    document_type: Literal["rule"] = "rule"


class GlossaryDocument(BaseModel):
    term: str
    definition: str
    related_rule_ids: list[str] = Field(default_factory=list)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    rules_version: str
    document_type: Literal["glossary"] = "glossary"


class ParsedRules(BaseModel):
    rules: list[RuleDocument] = Field(default_factory=list)
    glossary: list[GlossaryDocument] = Field(default_factory=list)

    def find_rule(self, rule_id: str) -> RuleDocument | None:
        return next((rule for rule in self.rules if rule.rule_id == rule_id), None)
