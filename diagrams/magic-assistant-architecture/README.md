# Magic Assistant Architecture

## What you are looking at

The implemented Part 1 system from the thin Streamlit client through FastAPI, `LiveAssistantRuntime`, the explicit LangGraph graph, deterministic rules/card services, structured LLM adapters, source validation, rendering, and thread memory.

## How to read it

Start at Streamlit and follow the emphasized request path into the graph. The upper pipeline shows how the local Comprehensive Rules PDF becomes `RuleEvidence`; the lower branch shows typed MagicTheGathering.io card access. Use the HTML guided views to isolate each path.

## Interview points

- One explicit graph routes known capabilities; this is not a multi-agent design.
- Rules and card data cross deterministic typed boundaries before LLM generation.
- The LLM selects structured source references; code validates and renders citations.
- `thread_id` selects an in-process LangGraph checkpoint without persisting turn-local evidence.
- Heavy runtime resources are initialized once and reused by CLI/API requests.

`diagram.json` is the Archify source. `diagram.html` is the self-contained interactive export; `diagram.svg` is the static vector export; `diagram.png` is the high-resolution light export for documents.
