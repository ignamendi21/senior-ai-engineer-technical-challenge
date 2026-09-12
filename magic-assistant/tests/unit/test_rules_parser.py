import os
from pathlib import Path

import pymupdf
import pytest

from magic_assistant.rules.parser import (
    RULES_VERSION,
    ComprehensiveRulesParser,
    RulesParseError,
)


@pytest.fixture
def parsed_rules():
    pages = [
        """Contents
7. Additional Rules
702. Keyword Abilities
""",
        """7. Additional Rules
702. Keyword Abilities
702.49. Ninjutsu
702.49a Ninjutsu is an activated ability that functions from a player's hand.
Return an unblocked attacking creature you control to its owner's hand. See rule 510.
Example: A creature returned this way remains an attacking creature.
702.49b The card with ninjutsu remains revealed. See rule 702.7 and section 8.
""",
        """Glossary
Ninjutsu
A keyword ability that lets a creature suddenly enter combat.
See rule 702.49, "Ninjutsu."
Credits
""",
    ]
    return ComprehensiveRulesParser().parse_pages(pages)


def test_parses_basic_numbered_rule(parsed_rules):
    rule = parsed_rules.find_rule("702.49")

    assert rule is not None
    assert rule.title == "Ninjutsu"
    assert rule.chapter_id == "7"
    assert rule.chapter_title == "Additional Rules"
    assert rule.section_id == "702"
    assert rule.section_title == "Keyword Abilities"
    assert rule.page_start == 2
    assert rule.page_end == 2


def test_parses_subrules_and_parent_relationship(parsed_rules):
    subrule_ids = [rule.rule_id for rule in parsed_rules.rules if rule.rule_id.startswith("702.49")]

    assert subrule_ids == ["702.49", "702.49a", "702.49b"]
    assert parsed_rules.find_rule("702.49a").parent_rule_id == "702.49"
    assert parsed_rules.find_rule("702.49b").parent_rule_id == "702.49"


def test_example_remains_attached_to_owning_rule(parsed_rules):
    first_subrule = parsed_rules.find_rule("702.49a")
    next_subrule = parsed_rules.find_rule("702.49b")

    assert "\nExample: A creature returned this way" in first_subrule.text
    assert "Example:" not in next_subrule.text


def test_extracts_rule_and_section_cross_references(parsed_rules):
    assert parsed_rules.find_rule("702.49a").related_rule_ids == ["510"]
    assert parsed_rules.find_rule("702.49b").related_rule_ids == ["702.7", "8"]


def test_does_not_treat_bare_quantities_as_rule_references():
    text = "A Commander deck contains exactly 100 cards. See rule 903.5."

    assert ComprehensiveRulesParser.extract_references(text) == ["903.5"]


def test_parses_glossary_as_first_class_document(parsed_rules):
    assert len(parsed_rules.glossary) == 1
    entry = parsed_rules.glossary[0]

    assert entry.term == "Ninjutsu"
    assert entry.definition.startswith("A keyword ability")
    assert entry.related_rule_ids == ["702.49"]
    assert entry.document_type == "glossary"
    assert entry.page_start == 3
    assert entry.page_end == 3


def test_applies_rules_version_to_every_document(parsed_rules):
    documents = [*parsed_rules.rules, *parsed_rules.glossary]

    assert documents
    assert {document.rules_version for document in documents} == {RULES_VERSION}


def test_missing_pdf_has_useful_error(tmp_path):
    missing_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError, match="Rules PDF not found"):
        ComprehensiveRulesParser().parse_pdf(missing_path)


def test_rejects_pdf_without_comprehensive_rules_structure(tmp_path):
    unrelated_pdf = tmp_path / "unrelated.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "This is a valid PDF, but it is not the rules document.")
        document.save(unrelated_pdf)

    with pytest.raises(RulesParseError, match="Could not locate the rules body"):
        ComprehensiveRulesParser().parse_pdf(unrelated_pdf)


def test_real_pdf_smoke_when_available():
    default_pdf = Path(__file__).parents[2] / "data" / "MagicCompRules 20260417.pdf"
    pdf_path = Path(os.environ.get("MAGIC_RULES_PDF", default_pdf))
    if not pdf_path.is_file():
        pytest.skip("Local Magic Comprehensive Rules PDF is not available")

    parsed = ComprehensiveRulesParser().parse_pdf(pdf_path)

    assert len(parsed.rules) == 3_285
    assert len({rule.rule_id for rule in parsed.rules}) == len(parsed.rules)
    assert len(parsed.glossary) == 730
    assert all(entry.definition for entry in parsed.glossary)
    assert parsed.find_rule("702.49").title == "Ninjutsu"
    assert next(
        entry for entry in parsed.glossary if entry.term == "Ninjutsu"
    ).related_rule_ids == ["702.49"]
