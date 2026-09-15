# Magic Assistant Rules Sequence

## What you are looking at

One rules question—“¿Cuántas fases tiene un turno?”—from user input through planning, hybrid retrieval of rule 500.1, grounded generation, deterministic source validation, and final presentation.

## How to read it

Read top to bottom. Green/emphasized steps carry deterministic retrieval evidence, violet/cloud participants mark structured LLM work, and security-colored steps validate selected IDs before citation rendering.

## Interview points

- Planning classifies the request; it does not answer the question.
- Exact/BM25/E5/RRF retrieval returns rule 500.1 with version and page provenance.
- The generation model receives evidence rather than unrestricted corpus access.
- Source validation is deterministic and retries are bounded to two attempts.
- Streamlit displays the answer and structured source once while preserving `thread_id`.

`diagram.json` is the Archify source. `diagram.html` is the self-contained interactive export; `diagram.svg` is the static vector export; `diagram.png` is the high-resolution light export for documents.
