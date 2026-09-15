# Part 2 — Code Review Reference Rewrite

This directory preserves the challenge's original RAG helper and provides a compact, testable rewrite. The full severity-based review and design reasoning are in [`../code_review.md`](../code_review.md).

## Layout

- `original.py`: the supplied code, preserved faithfully and intentionally not fixed. Do not run or import it.
- `improved/`: typed reference implementation with injected providers, persistent storage, bounded history/context, untrusted-reference prompt separation, and validated provenance.
- `tests/`: offline behavior tests and a local Chroma persistence test.
- `pyproject.toml` / `uv.lock`: isolated Python 3.12 project; it does not modify Magic Assistant dependencies.

## Setup and quality checks

Run from `code-review/`:

```console
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

`original.py` is excluded from Ruff because formatting it would violate the preservation requirement.

## Optional OpenAI configuration

Normal tests use fakes and require no credentials or Internet. To compose the real current OpenAI providers, export:

```text
OPENAI_API_KEY
RAG_CHAT_MODEL
RAG_EMBEDDING_MODEL
```

Optional persistent location:

```text
RAG_PERSIST_DIRECTORY=chroma_data
```

No model names are hard-coded because availability, price and context limits are deployment decisions.

## Minimal example

```python
from improved.models import SourceDocument
from improved.rag import build_openai_rag

rag = build_openai_rag()
rag.ingest_documents(
    [
        SourceDocument(
            source_id="handbook-v1",
            text="Reference text goes here.",
            metadata={"filename": "handbook.txt"},
        )
    ]
)
answer = rag.ask("What does the handbook say?", session_id="demo")
print(answer.text)
print(answer.sources)
```

The example uses local persistent Chroma for vectors and in-memory bounded session history. The rewrite is intentionally a small reference component, not a web service or production platform.
