# Clinic Solution Architecture

## What you are looking at

The approved Part 3 proposal for local clinical decision support: shared protocol knowledge, authorized patient context, deterministic calculations, specialized ECG pre-assessment, one local LLM, and validated clinician-facing output.

## How to read it

Everything in the large clinic boundary remains local. Follow Doctor → authentication/RBAC → backend → evidence/tools → local LLM → validation → grounded answer. The separate optional-cloud boundary contains only code/CI, signed updates, and non-sensitive monitoring.

## Interview points

- No patient or clinical data, embeddings, prompts, responses, or inference leave the clinic.
- Patient records are fetched on demand after authorization and never copied into the protocol index.
- Dosage arithmetic is deterministic; the LLM explains rather than calculates critical values.
- ECG pre-assessment is specialized local clinical ML, not general-LLM diagnosis, and requires clinician review.
- PostgreSQL + pgvector and one local base LLM minimize operational burden for one administrator.

`diagram.json` is the Archify source. `diagram.html` is the self-contained interactive export; `diagram.svg` is the static vector export; `diagram.png` is the high-resolution light export for documents.
