from magic_assistant.rules.chunker import RulesChunker
from magic_assistant.rules.models import GlossaryDocument, ParsedRules, RuleDocument


def make_rule(
    rule_id: str,
    text: str,
    *,
    parent_rule_id: str | None,
    page_start: int = 10,
    related_rule_ids: list[str] | None = None,
) -> RuleDocument:
    return RuleDocument(
        rule_id=rule_id,
        parent_rule_id=parent_rule_id,
        chapter_id="7",
        chapter_title="Additional Rules",
        section_id="702",
        section_title="Keyword Abilities",
        title="Ninjutsu" if rule_id == "702.49" else None,
        text=text,
        page_start=page_start,
        page_end=page_start,
        rules_version="2026-04-17",
        related_rule_ids=related_rule_ids or [],
    )


def ninjutsu_group() -> list[RuleDocument]:
    return [
        make_rule("702.49", "702.49. Ninjutsu", parent_rule_id="702"),
        make_rule("702.49a", "702.49a First semantic subrule.", parent_rule_id="702.49"),
        make_rule(
            "702.49b",
            "702.49b Second semantic subrule. See rule 510.",
            parent_rule_id="702.49",
            page_start=11,
            related_rule_ids=["510"],
        ),
    ]


def test_keeps_small_semantic_rule_group_together():
    chunks = RulesChunker(max_chars=500).chunk_rules(ninjutsu_group())

    assert len(chunks) == 1
    assert chunks[0].rule_ids == ["702.49", "702.49a", "702.49b"]
    assert chunks[0].text == "\n\n".join(rule.text for rule in ninjutsu_group())


def test_large_group_splits_only_at_subrule_boundaries():
    rules = ninjutsu_group()
    chunks = RulesChunker(max_chars=70).chunk_rules(rules)

    assert len(chunks) > 1
    chunk_documents = [document for chunk in chunks for document in chunk.text.split("\n\n")]
    assert chunk_documents == [rule.text for rule in rules]
    assert all(
        rule.text in chunk.text
        for rule in rules
        for chunk in chunks
        if rule.rule_id in chunk.rule_ids
    )
    assert {chunk.root_rule_id for chunk in chunks} == {"702.49"}
    assert chunks[0].parent_rule_id == "702"
    assert chunks[-1].parent_rule_id == "702.49"


def test_single_oversized_subrule_is_not_split_mid_text():
    oversized = make_rule("702.49a", "702.49a " + "content " * 30, parent_rule_id="702.49")

    chunk = RulesChunker(max_chars=50).chunk_rules([oversized])[0]

    assert chunk.text == oversized.text
    assert len(chunk.text) > 50


def test_metadata_survives_chunking():
    chunk = RulesChunker(max_chars=500).chunk_rules(ninjutsu_group())[0]

    assert chunk.root_rule_id == "702.49"
    assert chunk.parent_rule_id == "702"
    assert chunk.chapter_id == "7"
    assert chunk.chapter_title == "Additional Rules"
    assert chunk.section_id == "702"
    assert chunk.section_title == "Keyword Abilities"
    assert chunk.title == "Ninjutsu"
    assert chunk.page_start == 10
    assert chunk.page_end == 11
    assert chunk.rules_version == "2026-04-17"
    assert chunk.related_rule_ids == ["510"]
    assert chunk.document_type == "rule"


def test_glossary_entry_becomes_its_own_semantic_chunk():
    glossary = GlossaryDocument(
        term="Ninjutsu",
        definition="A keyword ability. See rule 702.49.",
        related_rule_ids=["702.49"],
        page_start=286,
        page_end=286,
        rules_version="2026-04-17",
    )

    chunks = RulesChunker().chunk(ParsedRules(glossary=[glossary]))

    assert len(chunks) == 1
    assert chunks[0].root_rule_id is None
    assert chunks[0].term == "Ninjutsu"
    assert chunks[0].text.startswith("Ninjutsu\n")
    assert chunks[0].related_rule_ids == ["702.49"]
    assert chunks[0].document_type == "glossary"


def test_reports_rule_length_statistics():
    statistics = RulesChunker.length_statistics(ninjutsu_group())

    assert statistics.count == 3
    assert statistics.maximum == len(ninjutsu_group()[2].text)
    assert statistics.p90 == statistics.maximum
