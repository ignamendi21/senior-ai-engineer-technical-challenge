# Senior AI Engineer Technical Challenge

This repository contains the completed three-part submission: a working Magic: The Gathering AI assistant, a senior review and rewrite of a legacy RAG helper, and a concise healthcare-clinic solution proposal. Architecture visualizations provide interview-oriented views across all three parts.

## Submission overview

### Part 1 — Magic: The Gathering AI Assistant

**Status: Complete**

- Implementation: [`magic-assistant/`](magic-assistant/)
- Technical guide and demo instructions: [`magic-assistant/README.md`](magic-assistant/README.md)
- Visuals: [architecture](diagrams/magic-assistant-architecture/diagram.html) · [rules-question sequence](diagrams/magic-assistant-rules-sequence/diagram.html)

Part 1 includes structure-aware rules ingestion, hybrid multilingual retrieval, deterministic card API access, explicit LangGraph orchestration, grounded source validation, and FastAPI/Streamlit demo boundaries. The end-to-end flow passed its live acceptance test with a real LLM, local rules retrieval, the MTG API, and Streamlit UI.

### Part 2 — Code Review

**Status: Complete**

- Senior review: [`code_review.md`](code_review.md)
- Preserved original and reference rewrite: [`code-review/`](code-review/)
- Visual: [before/after architecture](diagrams/code-review-before-after/diagram.html)

The original challenge script is preserved unchanged. The deliverable provides a severity-based review, a compact dependency-injected RAG rewrite, persistent and compatible vector storage, bounded session/context handling, validated provenance, and offline tests.

### Part 3 — Solution Proposal

**Status: Complete**

- Proposal: [`solution_design.md`](solution_design.md)
- Scope note: [`solution-design/`](solution-design/)
- Visual: [clinic solution architecture](diagrams/clinic-solution-architecture/diagram.html)

The proposal defines a local clinical decision-support architecture with strict separation between patient data and shared protocols, deterministic clinical calculations, specialized local ECG pre-assessment, local LLM inference, source validation, and privacy-by-design controls.

## Architecture visualizations

- [Magic Assistant architecture](diagrams/magic-assistant-architecture/)
- [Magic rules-question sequence](diagrams/magic-assistant-rules-sequence/)
- [Code review before/after](diagrams/code-review-before-after/)
- [Clinic solution architecture](diagrams/clinic-solution-architecture/)

Each directory contains the Archify JSON source, a self-contained interactive HTML export, SVG and high-resolution PNG exports, and concise interview notes.

## Repository structure

```text
magic-assistant/       Part 1 implementation, tests, API and UI
code-review/           Part 2 preserved original, improved RAG and tests
solution-design/       Part 3 scope note
code_review.md         Required senior code-review deliverable
solution_design.md     Required clinic solution proposal
diagrams/              Architecture sources, exports and explainability notes
decisions.md           Part 1 architectural decision records
```

## Running the demo

See the Part 1 [Quick start and demo guide](magic-assistant/README.md#quick-start) for PDF placement, index creation, environment configuration, FastAPI, Streamlit, tests, and optional Docker Compose usage.

## Documentation

- [Part 1 architectural decisions](decisions.md)
- [Part 2 senior code review](code_review.md)
- [Part 3 clinic solution proposal](solution_design.md)
