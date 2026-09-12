import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from magic_assistant.rules.models import GlossaryDocument, ParsedRules, RuleDocument

RULES_VERSION = "2026-04-17"
_SECTION_MAX_INDENT = 100
_PARENT_RULE_MAX_INDENT = 115
_SUBRULE_MAX_INDENT = 130

_CHAPTER_RE = re.compile(r"^(?P<id>[1-9])\.\s+(?P<title>\S.*)$")
_SECTION_RE = re.compile(r"^(?P<id>\d{3})\.\s+(?P<title>\S.*)$")
_RULE_RE = re.compile(r"^(?P<id>\d{3}(?:\.\d+)+[a-z]?)\.?\s+(?P<text>\S.*)$")
_RULE_REFERENCE_RE = re.compile(
    r"\brules?\s+(?P<id>[1-9]\d{2}(?:\.\d+)*(?:[a-z])?)(?!\d|\.\d)",
    re.IGNORECASE,
)
_SECTION_REFERENCE_RE = re.compile(r"\bsections?\s+(?P<id>[1-9])\b", re.IGNORECASE)


class RulesParseError(ValueError):
    pass


@dataclass(frozen=True)
class _ExtractedLine:
    text: str
    page: int
    is_bold: bool | None = None
    indent: float | None = None


@dataclass
class _RuleBuilder:
    rule_id: str
    parent_rule_id: str | None
    chapter_id: str | None
    chapter_title: str | None
    section_id: str | None
    section_title: str | None
    title: str | None
    lines: list[str]
    page_start: int
    page_end: int


@dataclass
class _GlossaryBuilder:
    term: str
    lines: list[str] = field(default_factory=list)
    page_start: int = 1
    page_end: int = 1


class ComprehensiveRulesParser:
    def parse_pdf(self, path: Path) -> ParsedRules:
        if not path.is_file():
            raise FileNotFoundError(f"Rules PDF not found: {path}")

        try:
            with pymupdf.open(path) as document:
                pages = [
                    self._extract_pdf_page(page, number) for number, page in enumerate(document, 1)
                ]
        except pymupdf.FileDataError as error:
            raise RulesParseError(f"Could not read rules PDF: {path}") from error

        return self._parse_extracted_pages(pages, require_complete_document=True)

    def parse_pages(self, pages: Sequence[str]) -> ParsedRules:
        extracted_pages = [
            [
                _ExtractedLine(text=line.strip(), page=page_number)
                for line in page.splitlines()
                if line.strip()
            ]
            for page_number, page in enumerate(pages, 1)
        ]
        return self._parse_extracted_pages(extracted_pages, require_complete_document=False)

    @staticmethod
    def _extract_pdf_page(page: pymupdf.Page, page_number: int) -> list[_ExtractedLine]:
        extracted: list[_ExtractedLine] = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(span["text"] for span in spans).strip()
                if text:
                    extracted.append(
                        _ExtractedLine(
                            text=text,
                            page=page_number,
                            is_bold=bool(spans) and all(span["flags"] & 16 for span in spans),
                            indent=line["bbox"][0],
                        )
                    )
        return extracted

    def _parse_extracted_pages(
        self,
        pages: Sequence[Sequence[_ExtractedLine]],
        *,
        require_complete_document: bool,
    ) -> ParsedRules:
        first_rules_page = self._find_first_rules_page(pages)
        if first_rules_page is None:
            if require_complete_document:
                raise RulesParseError("Could not locate the rules body in the PDF")
            first_rules_page = 0
        rules: list[RuleDocument] = []
        glossary: list[GlossaryDocument] = []
        current_rule: _RuleBuilder | None = None
        current_glossary: _GlossaryBuilder | None = None
        chapter_id: str | None = None
        chapter_title: str | None = None
        section_id: str | None = None
        section_title: str | None = None
        in_glossary = False

        for page in pages[first_rules_page:]:
            for index, line in enumerate(page):
                if line.text == "Credits":
                    if current_glossary:
                        glossary.append(self._build_glossary(current_glossary))
                    return self._finish_parsing(rules, glossary, require_complete_document)

                if line.text == "Glossary":
                    if current_rule:
                        rules.append(self._build_rule(current_rule))
                        current_rule = None
                    in_glossary = True
                    continue

                if in_glossary:
                    next_text = page[index + 1].text if index + 1 < len(page) else None
                    if self._is_glossary_term(line, next_text):
                        if current_glossary:
                            glossary.append(self._build_glossary(current_glossary))
                        current_glossary = _GlossaryBuilder(
                            term=line.text,
                            page_start=line.page,
                            page_end=line.page,
                        )
                    elif current_glossary:
                        current_glossary.lines.append(line.text)
                        current_glossary.page_end = line.page
                    continue

                chapter_match = _CHAPTER_RE.fullmatch(line.text)
                if chapter_match and line.is_bold is not False:
                    if current_rule:
                        rules.append(self._build_rule(current_rule))
                        current_rule = None
                    chapter_id = chapter_match["id"]
                    chapter_title = chapter_match["title"]
                    section_id = None
                    section_title = None
                    continue

                section_match = _SECTION_RE.fullmatch(line.text)
                if (
                    section_match
                    and chapter_id == section_match["id"][0]
                    and (line.indent is None or line.indent < _SECTION_MAX_INDENT)
                ):
                    if current_rule:
                        rules.append(self._build_rule(current_rule))
                    section_id = section_match["id"]
                    section_title = section_match["title"]
                    current_rule = _RuleBuilder(
                        rule_id=section_id,
                        parent_rule_id=None,
                        chapter_id=chapter_id,
                        chapter_title=chapter_title,
                        section_id=section_id,
                        section_title=section_title,
                        title=section_title,
                        lines=[line.text],
                        page_start=line.page,
                        page_end=line.page,
                    )
                    continue

                rule_match = _RULE_RE.fullmatch(line.text)
                if rule_match and self._is_rule_start(rule_match["id"], line.indent):
                    if current_rule:
                        rules.append(self._build_rule(current_rule))
                    rule_id = rule_match["id"]
                    body = rule_match["text"] or ""
                    current_rule = _RuleBuilder(
                        rule_id=rule_id,
                        parent_rule_id=self._parent_rule_id(rule_id),
                        chapter_id=chapter_id,
                        chapter_title=chapter_title,
                        section_id=section_id,
                        section_title=section_title,
                        title=body if self._looks_like_rule_title(body) else None,
                        lines=[line.text],
                        page_start=line.page,
                        page_end=line.page,
                    )
                    continue

                if current_rule:
                    current_rule.lines.append(line.text)
                    current_rule.page_end = line.page

        if current_rule:
            rules.append(self._build_rule(current_rule))
        if current_glossary:
            glossary.append(self._build_glossary(current_glossary))
        return self._finish_parsing(rules, glossary, require_complete_document)

    @staticmethod
    def _find_first_rules_page(pages: Sequence[Sequence[_ExtractedLine]]) -> int | None:
        for index, page in enumerate(pages):
            chapter_match = _CHAPTER_RE.fullmatch(page[0].text) if page else None
            if (
                chapter_match
                and page[0].is_bold is not False
                and any(
                    (section_match := _SECTION_RE.fullmatch(line.text))
                    and section_match["id"].startswith(chapter_match["id"])
                    and (line.indent is None or line.indent < _SECTION_MAX_INDENT)
                    for line in page[1:]
                )
            ):
                return index
        return None

    @staticmethod
    def _finish_parsing(
        rules: list[RuleDocument],
        glossary: list[GlossaryDocument],
        require_complete_document: bool,
    ) -> ParsedRules:
        parsed = ParsedRules(rules=rules, glossary=glossary)
        if not require_complete_document:
            return parsed

        rule_ids = [rule.rule_id for rule in rules]
        required_anchors = {"100", "100.1", "900"}
        if not required_anchors.issubset(rule_ids) or not glossary:
            raise RulesParseError("PDF does not contain the expected Comprehensive Rules structure")
        if len(rule_ids) != len(set(rule_ids)):
            raise RulesParseError("Parsed rule identifiers are not unique")
        return parsed

    @staticmethod
    def _is_rule_start(rule_id: str, indent: float | None) -> bool:
        if indent is None:
            return True
        maximum_indent = _SUBRULE_MAX_INDENT if rule_id[-1].isalpha() else _PARENT_RULE_MAX_INDENT
        return indent < maximum_indent

    @staticmethod
    def _parent_rule_id(rule_id: str) -> str:
        if rule_id[-1].isalpha():
            return rule_id[:-1]
        return rule_id.rsplit(".", 1)[0]

    @staticmethod
    def _looks_like_rule_title(text: str) -> bool:
        if not text or len(text) > 100 or text[-1] in ".:;?!":
            return False
        words = re.findall(r"[A-Za-z]+", text)
        return bool(words) and all(
            word[0].isupper() or word.lower() in {"a", "and", "of", "the"} for word in words
        )

    @staticmethod
    def _is_glossary_term(line: _ExtractedLine, next_text: str | None) -> bool:
        if line.is_bold is not None:
            return line.is_bold
        if not next_text or len(line.text) > 80 or line.text[-1] in ".:;?!":
            return False
        words = re.findall(r"[A-Za-z]+", line.text)
        return bool(words) and all(
            word[0].isupper() or word.lower() in {"a", "an", "and", "as", "of", "or", "the", "to"}
            for word in words
        )

    def _build_rule(self, builder: _RuleBuilder) -> RuleDocument:
        text = self._join_rule_lines(builder.lines)
        return RuleDocument(
            rule_id=builder.rule_id,
            parent_rule_id=builder.parent_rule_id,
            chapter_id=builder.chapter_id,
            chapter_title=builder.chapter_title,
            section_id=builder.section_id,
            section_title=builder.section_title,
            title=builder.title,
            text=text,
            page_start=builder.page_start,
            page_end=builder.page_end,
            rules_version=RULES_VERSION,
            related_rule_ids=self.extract_references(text, exclude={builder.rule_id}),
        )

    def _build_glossary(self, builder: _GlossaryBuilder) -> GlossaryDocument:
        definition = " ".join(builder.lines).strip()
        return GlossaryDocument(
            term=builder.term,
            definition=definition,
            related_rule_ids=self.extract_references(definition),
            page_start=builder.page_start,
            page_end=builder.page_end,
            rules_version=RULES_VERSION,
        )

    @staticmethod
    def _join_rule_lines(lines: Sequence[str]) -> str:
        result = ""
        for line in lines:
            separator = "\n" if line.startswith("Example:") and result else " " if result else ""
            result += separator + line
        return result

    @staticmethod
    def extract_references(text: str, exclude: set[str] | None = None) -> list[str]:
        excluded = exclude or set()
        references = [match["id"] for match in _RULE_REFERENCE_RE.finditer(text)]
        references.extend(match["id"] for match in _SECTION_REFERENCE_RE.finditer(text))
        return list(
            dict.fromkeys(reference for reference in references if reference not in excluded)
        )
