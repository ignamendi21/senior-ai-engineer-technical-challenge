import argparse
from collections.abc import Sequence
from pathlib import Path

from magic_assistant.rules.chunker import RulesChunker
from magic_assistant.rules.models import ParsedRules
from magic_assistant.rules.parser import ComprehensiveRulesParser

DEFAULT_PDF_PATH = Path("data/MagicCompRules 20260417.pdf")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect parsed Magic Comprehensive Rules")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF_PATH, help="path to the rules PDF")
    parser.add_argument("--rule", help="rule identifier to display, for example 702.49")
    parser.add_argument("--summary", action="store_true", help="display ingestion summary")
    return parser


def print_rule(parsed: ParsedRules, rule_id: str) -> bool:
    rule = parsed.find_rule(rule_id)
    if rule is None:
        return False

    documents = [
        rule,
        *(candidate for candidate in parsed.rules if candidate.parent_rule_id == rule_id),
    ]
    contained_ids = {document.rule_id for document in documents}
    related_ids = list(
        dict.fromkeys(
            reference
            for document in documents
            for reference in document.related_rule_ids
            if reference not in contained_ids
        )
    )
    heading = f"Rule: {rule.rule_id}"
    if rule.title:
        heading += f" — {rule.title}"
    print(heading)
    page_start = min(item.page_start for item in documents)
    page_end = max(item.page_end for item in documents)
    print(f"Pages: {page_start}-{page_end}")
    print(f"Version: {rule.rules_version}")
    print()
    print("\n\n".join(document.text for document in documents))
    print()
    print(f"Related rules: {', '.join(related_ids) if related_ids else 'None'}")
    return True


def print_summary(parsed: ParsedRules) -> None:
    chunks = RulesChunker().chunk(parsed)
    statistics = RulesChunker.length_statistics(parsed.rules)
    print(f"Parsed rules/sections: {len(parsed.rules)}")
    print(f"Glossary entries: {len(parsed.glossary)}")
    print(f"Semantic chunks: {len(chunks)}")
    print("Rule text lengths (characters):")
    print(f"  Average: {statistics.average:.2f}")
    print(f"  Median: {statistics.median:.1f}")
    print(f"  p90: {statistics.p90}")
    print(f"  p95: {statistics.p95}")
    print(f"  Maximum: {statistics.maximum}")


def main(argv: Sequence[str] | None = None) -> int:
    argument_parser = build_argument_parser()
    arguments = argument_parser.parse_args(argv)
    try:
        parsed = ComprehensiveRulesParser().parse_pdf(arguments.pdf)
    except (FileNotFoundError, ValueError) as error:
        argument_parser.error(str(error))

    if arguments.rule and not print_rule(parsed, arguments.rule):
        argument_parser.error(f"Rule {arguments.rule!r} was not found")
    if arguments.summary or not arguments.rule:
        print_summary(parsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
