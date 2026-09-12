import math
from collections.abc import Sequence
from statistics import mean, median
from typing import Literal

from pydantic import BaseModel, Field

from magic_assistant.rules.models import GlossaryDocument, ParsedRules, RuleDocument

DEFAULT_MAX_CHARS = 3_000


class RuleChunk(BaseModel):
    chunk_id: str
    rule_ids: list[str] = Field(default_factory=list)
    parent_rule_id: str | None = None
    chapter_id: str | None = None
    chapter_title: str | None = None
    section_id: str | None = None
    section_title: str | None = None
    title: str | None = None
    term: str | None = None
    text: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    rules_version: str
    related_rule_ids: list[str] = Field(default_factory=list)
    document_type: Literal["rule", "glossary"]


class LengthStatistics(BaseModel):
    count: int
    average: float
    median: float
    p90: int
    p95: int
    maximum: int


class RulesChunker:
    def __init__(self, max_chars: int = DEFAULT_MAX_CHARS) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self.max_chars = max_chars

    def chunk(self, parsed: ParsedRules) -> list[RuleChunk]:
        return [*self.chunk_rules(parsed.rules), *self.chunk_glossary(parsed.glossary)]

    def chunk_rules(self, rules: Sequence[RuleDocument]) -> list[RuleChunk]:
        chunks: list[RuleChunk] = []
        for group in self._semantic_groups(rules):
            partitions = self._partition_group(group)
            for index, partition in enumerate(partitions, 1):
                chunks.append(self._build_rule_chunk(group, partition, index))
        return chunks

    @staticmethod
    def chunk_glossary(glossary: Sequence[GlossaryDocument]) -> list[RuleChunk]:
        return [
            RuleChunk(
                chunk_id=f"glossary:{index:04d}",
                term=document.term,
                text=f"{document.term}\n{document.definition}",
                page_start=document.page_start,
                page_end=document.page_end,
                rules_version=document.rules_version,
                related_rule_ids=document.related_rule_ids,
                document_type="glossary",
            )
            for index, document in enumerate(glossary, 1)
        ]

    @staticmethod
    def length_statistics(rules: Sequence[RuleDocument]) -> LengthStatistics:
        lengths = sorted(len(rule.text) for rule in rules)
        if not lengths:
            return LengthStatistics(count=0, average=0, median=0, p90=0, p95=0, maximum=0)
        return LengthStatistics(
            count=len(lengths),
            average=round(mean(lengths), 2),
            median=median(lengths),
            p90=RulesChunker._percentile(lengths, 0.90),
            p95=RulesChunker._percentile(lengths, 0.95),
            maximum=lengths[-1],
        )

    @staticmethod
    def _semantic_groups(rules: Sequence[RuleDocument]) -> list[list[RuleDocument]]:
        groups: list[list[RuleDocument]] = []
        group_keys: list[str] = []
        for rule in rules:
            key = rule.rule_id[:-1] if rule.rule_id[-1].isalpha() else rule.rule_id
            if group_keys and group_keys[-1] == key:
                groups[-1].append(rule)
            else:
                group_keys.append(key)
                groups.append([rule])
        return groups

    def _partition_group(self, group: Sequence[RuleDocument]) -> list[list[RuleDocument]]:
        partitions: list[list[RuleDocument]] = []
        current: list[RuleDocument] = []
        current_length = 0
        for rule in group:
            separator_length = 2 if current else 0
            candidate_length = current_length + separator_length + len(rule.text)
            if current and candidate_length > self.max_chars:
                partitions.append(current)
                current = [rule]
                current_length = len(rule.text)
            else:
                current.append(rule)
                current_length = candidate_length
        if current:
            partitions.append(current)
        return partitions

    @staticmethod
    def _build_rule_chunk(
        complete_group: Sequence[RuleDocument],
        partition: Sequence[RuleDocument],
        index: int,
    ) -> RuleChunk:
        group_root = complete_group[0]
        contained_ids = {rule.rule_id for rule in partition}
        related_ids = dict.fromkeys(
            reference
            for rule in partition
            for reference in rule.related_rule_ids
            if reference not in contained_ids
        )
        return RuleChunk(
            chunk_id=f"rule:{group_root.rule_id}:{index}",
            rule_ids=[rule.rule_id for rule in partition],
            parent_rule_id=partition[0].parent_rule_id,
            chapter_id=group_root.chapter_id,
            chapter_title=group_root.chapter_title,
            section_id=group_root.section_id,
            section_title=group_root.section_title,
            title=group_root.title,
            text="\n\n".join(rule.text for rule in partition),
            page_start=min(rule.page_start for rule in partition),
            page_end=max(rule.page_end for rule in partition),
            rules_version=group_root.rules_version,
            related_rule_ids=list(related_ids),
            document_type="rule",
        )

    @staticmethod
    def _percentile(sorted_values: Sequence[int], percentile: float) -> int:
        index = max(0, math.ceil(percentile * len(sorted_values)) - 1)
        return sorted_values[index]
