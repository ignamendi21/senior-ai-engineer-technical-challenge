# Magic Assistant — Phase 1

Phase 1 builds the knowledge-ingestion foundation for the Magic: The Gathering Comprehensive Rules effective April 17, 2026. It extracts rules and glossary entries into typed records, then creates chunks at semantic rule boundaries.

This phase deliberately excludes retrieval, embeddings, vector databases, LLMs, agents, chat, APIs, and user interfaces.

## Requirements and setup

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)

From this directory:

```console
uv python install 3.12
uv sync --locked
```

`uv` creates and manages the local environment. Runtime and development dependencies are locked in `uv.lock`.

## Source PDF

Place the separately supplied file at:

```text
data/MagicCompRules 20260417.pdf
```

The PDF is intentionally excluded from Git. See [`data/README.md`](data/README.md) for details.

## Tests

Run all unit tests:

```console
uv run pytest
```

The real-PDF smoke test runs when the PDF exists in `data/`; otherwise it skips gracefully. A different local path can be supplied with `MAGIC_RULES_PDF`. In PowerShell:

```powershell
$env:MAGIC_RULES_PDF = "C:\path\to\MagicCompRules 20260417.pdf"
uv run pytest
```

## Linting and formatting

```console
uv run ruff check .
uv run ruff format --check .
```

To apply Ruff formatting locally, run `uv run ruff format .`.

## Inspect parsed output

Inspect a rule and its direct subrules:

```console
uv run python -m magic_assistant.rules.inspect --pdf "data/MagicCompRules 20260417.pdf" --rule 702.49
```

Print corpus and rule-length statistics:

```console
uv run python -m magic_assistant.rules.inspect --pdf "data/MagicCompRules 20260417.pdf" --summary
```

## Parsing strategy

The parser is intentionally specific to the known Comprehensive Rules format:

1. PyMuPDF extracts each physical page while retaining 1-based page provenance.
2. Front matter is skipped by locating the first chapter/section page rather than parsing the table of contents.
3. Chapter headings, three-digit sections, numbered rules, and lettered subrules are recognized separately.
4. Wrapped lines and `Example:` blocks remain attached to the active rule.
5. The glossary is detected by its heading and parsed into separate `GlossaryDocument` records; PDF font metadata distinguishes terms from definitions.
6. Explicit `rule`/`rules` citations and section references are retained as relationships; bare quantities are not treated as citations.

`parse_pdf` detects the effective-date statement in the front matter and normalizes it to the ISO `rules_version` stored on every record; parsing fails if that provenance is missing. Text fixtures default to `2026-04-17` and may supply an explicit version. Section headings are represented as rule records because identifiers such as `702` are valid citation and hierarchy anchors.

## Structure-aware chunking

`RulesChunker` groups a numbered parent rule with its lettered subrules. Every partition retains a stable `root_rule_id` for that semantic group, while `parent_rule_id` describes the first record's immediate hierarchy. A group is emitted intact when it fits the configured maximum. Large groups are partitioned only between records, never in the middle of a rule or subrule. Any oversized atomic record remains intact, making `max_chars` a soft limit in that exceptional case. Glossary entries each form one semantic chunk.

The default is **3,000 characters**. Measurements from the supplied 309-page PDF were:

| Measurement | Individual rule records | Parent/subrule groups |
| --- | ---: | ---: |
| Count | 3,285 | 1,312 |
| Average | 249.50 | 627.72 |
| Median | 197 | 330 |
| p90 | 492 | 1,456 |
| p95 | 667 | 2,358 |
| Maximum | 2,827 | 10,127 |

At 3,000 characters, 96.88% of semantic groups remain intact. The outliers split at subrule boundaries. The complete corpus produced 1,361 rule chunks and 730 glossary chunks; the largest resulting rule chunk was 2,947 characters.

## Known limitations

- The parser targets this English Comprehensive Rules layout rather than arbitrary PDFs or future layouts with different typography.
- Reference extraction captures explicit rule and section citations; it does not attempt to interpret every shorthand range.
- Physical PDF page numbers are recorded, not any independently printed page labels.
- Any atomic rule or glossary entry that exceeds the configured maximum remains oversized rather than losing semantic integrity.
