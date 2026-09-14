# Magic Assistant — Phases 1–3

The implementation has three independently testable layers:

1. **Phase 1:** structure-aware ingestion of the Magic: The Gathering Comprehensive Rules.
2. **Phase 2:** deterministic hybrid rules retrieval and Magic card API access.
3. **Phase 3:** one explicit LangGraph conversational orchestrator with grounded generation.

Phase 3 includes a development terminal chat, not a web API or final UI.

## Requirements and setup

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)

From this directory:

```console
uv python install 3.12
uv sync --locked
```

The first real retrieval run downloads `intfloat/multilingual-e5-small` through Sentence Transformers into the user's external Hugging Face cache. Model weights are not stored in this repository.

For a live agent run, configure environment variables from the root `.env.example`:

- `OPENAI_API_KEY`: required secret; never commit it.
- `MAGIC_CHAT_MODEL`: required model selected by the operator.
- `OPENAI_BASE_URL`: optional OpenAI-compatible base URL.
- `MAGIC_RULES_PDF` and `MAGIC_RULES_INDEX`: optional local smoke-test paths.

No model name or API key is hard-coded.

## Source PDF

Place the separately supplied file at:

```text
data/MagicCompRules 20260417.pdf
```

The PDF and generated index are intentionally excluded from Git. See [`data/README.md`](data/README.md).

## Tests and quality checks

```console
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The real-PDF smoke test runs when the PDF exists in `data/`; otherwise it skips. A different local path can be supplied with `MAGIC_RULES_PDF`:

```powershell
$env:MAGIC_RULES_PDF = "C:\path\to\MagicCompRules 20260417.pdf"
uv run pytest
```

Unit tests inject deterministic planners, generators, retrievers, embedding providers, and mocked HTTP transports. They do not call OpenAI, download a model, or use the Internet.

The optional live MTG API smoke test is disabled by default:

```powershell
$env:RUN_LIVE_MTG_API_TESTS = "1"
uv run pytest tests/integration/test_card_api_live.py
```

The optional OpenAI graph smoke test also requires the local PDF/index and explicit credentials:

```powershell
$env:RUN_LIVE_LLM_TESTS = "1"
$env:OPENAI_API_KEY = "..."
$env:MAGIC_CHAT_MODEL = "operator-selected-model"
uv run pytest tests/integration/test_agent_live.py
```

## Phase 1: rules ingestion

PyMuPDF extracts each physical page with 1-based page provenance. The parser skips front matter, recognizes chapters, sections, rules and lettered subrules, retains examples and cross-references, and parses the glossary separately. The effective-date statement supplies the normalized rules version.

`RulesChunker` groups a numbered parent with its lettered subrules. Groups under the configurable 3,000-character soft maximum remain intact; larger groups split only between atomic records. Every partition retains a stable `root_rule_id`.

Inspect parsing:

```console
uv run python -m magic_assistant.rules.inspect --pdf "data/MagicCompRules 20260417.pdf" --rule 702.49
uv run python -m magic_assistant.rules.inspect --pdf "data/MagicCompRules 20260417.pdf" --summary
```

The supplied PDF produces 3,285 unique rule/section records, 730 glossary entries, 1,361 rule chunks, and 730 glossary chunks.

## Phase 2: hybrid rules retrieval

`RulesKnowledgeBase` exposes two operations:

- `get_rule(rule_id)` performs indexed deterministic lookup without semantic search. A section ID returns chunks in that section, a root ID returns all of its partitions, and a subrule returns its containing chunk.
- `search(query, top_k=5)` returns structured `RuleEvidence` with source metadata, score, and retrieval methods.

Search combines:

1. Explicit rule-ID detection and controlled complete-token glossary terminology matching. Linked rule evidence precedes its supporting glossary entry; explicit rule IDs remain strongest.
2. Deterministic BM25 tokenization and ranking.
3. Normalized multilingual E5 query/document embeddings.
4. Reciprocal Rank Fusion with a configurable default constant of 60.
5. At most one glossary-to-explicit-related-rule expansion step, with duplicate evidence removed.

The multilingual model is necessary because the source is English while queries may be English or Spanish. E5 inputs use `query:` and `passage:` prefixes as required by the model.

### Build or validate the local index

```console
uv run python -m magic_assistant.rules.index --pdf "data/MagicCompRules 20260417.pdf"
```

If an index exists, this validates it. Rebuild explicitly after changing the corpus or model:

```console
uv run python -m magic_assistant.rules.index --pdf "data/MagicCompRules 20260417.pdf" --rebuild
```

The generated `data/index/` contains:

- `embeddings.npy`: normalized float matrix.
- `chunks.jsonl`: source chunk metadata.
- `manifest.json`: schema version, model, rules version, chunk count, dimension, deterministic corpus fingerprint, and embedding-matrix fingerprint.

A stale or incompatible index fails clearly rather than being used silently.

### Search rules

```console
uv run python -m magic_assistant.rules.search "¿Cómo funciona ninjutsu?" --pdf "data/MagicCompRules 20260417.pdf"
```

Output includes rank, rule IDs, semantic root, title or glossary term, pages, retrieval methods, score, and a text preview.

### Evaluate retrieval

```console
uv run python -m magic_assistant.rules.evaluation --pdf "data/MagicCompRules 20260417.pdf"
```

The committed 14-case English/Spanish benchmark reports per-query expected-rule ranks, Hit@1, Hit@3, Hit@5, and full-ranking MRR. It measures whether expected rules are retrieved and does not assert a fabricated quality threshold in tests. The Phase 2 baseline was Hit@1 0.214, Hit@3 1.000, Hit@5 1.000, and MRR 0.595. Controlled terminology anchoring produces Hit@1 0.714, Hit@3 0.929, Hit@5 1.000, and MRR 0.836 without changing expected answers or tuning RRF.

## Phase 2: Magic card API

`MtgApiClient` uses direct `httpx` access to `https://api.magicthegathering.io/v1/cards`. API dictionaries are normalized immediately into typed `Card`, `CardRuling`, and legality models. `CardSearchService` accepts typed `CardSearchFilters`; callers never construct query strings.

Directly supported filters are sent to the API. Color codes use `W`, `U`, `B`, `R`, and `G`. The service always validates returned records client-side, including exclusive/inclusive mana-value ranges that the API cannot express directly.

Search is bounded by a result limit and maximum page count. Results are deduplicated by normalized card name. Among printings seen, the representative with more useful Oracle text, image, and ruling data wins, with stable set/ID tie-breaking.

The client uses explicit timeouts, bounded exponential retries for network and 5xx failures, no blind retry for deterministic 4xx responses, typed failures, malformed-payload validation, and captured rate-limit headers. Internally created clients validate TLS with the operating system's native trust store through `truststore`, which supports public roots and enterprise-managed roots without disabling verification. An injected `httpx.Client` remains supported for tests.

If an additional PEM CA bundle is required, set `MTG_API_CA_BUNDLE` to its path. The bundle extends native trust; certificate verification remains enabled. See the root `.env.example`.

Inspect card search:

```console
uv run python -m magic_assistant.cards.inspect --color W --type Creature --subtype Warrior --max-cmc 2
```

`--max-cmc` is exclusive, so this example returns matching cards with mana value below 2.

## Phase 3: LangGraph assistant

The application uses one explicit `StateGraph`; it is not a multi-agent or unrestricted ReAct system. Supported planner intents are:

- `rules_question`
- `card_search`
- `card_interaction`
- `custom_card`
- `out_of_scope`

The planner only classifies and extracts typed domain data. Rules and card nodes call the approved deterministic services. Card searches are rendered directly from `Card` objects. Generative rules and interaction answers receive only retrieved `RuleEvidence` and resolved cards.

OpenAI calls use LangChain `with_structured_output` with Pydantic schemas and native `json_schema` output. Provider-level strict mode is disabled because the nested domain schemas intentionally contain defaults and numeric constraints unsupported by OpenAI strict schemas; LangChain still validates returned data into the declared Pydantic models.

### Graph overview

```text
START → plan_request
  ├─ rules → retrieve_rules → synthesize_grounded_answer
  ├─ card search → search_cards → render_card_search
  ├─ interaction → resolve_named_cards → retrieve_interaction_rules → synthesize_grounded_answer
  ├─ custom card → retrieve_custom_mechanics → generate_custom_card → render_custom_card
  └─ out of scope → scope_response

synthesize_grounded_answer → validate_sources
  ├─ valid → render_answer → END
  ├─ invalid, attempt remaining → synthesize_grounded_answer
  └─ invalid twice → grounding_fallback → END
```

The maximum is two total synthesis attempts. A deterministic validator requires every selected rule chunk/card ID to belong to evidence in graph state. Rules answers require rule provenance; interaction answers require both rule and card provenance. Source lines and PDF pages are rendered by code, never invented by the model.

Card interaction retrieval is enriched with the original question, planner mechanics query, resolved card names, and actual Oracle text. Named-card resolution prefers exact normalized names before a deterministic first candidate.

Custom cards use a distinct `CustomCard` model and are always rendered with `CUSTOM / FAN-MADE — NOT AN OFFICIAL MAGIC CARD`. They are never sent to or mixed with records from the MTG API.

Expected deterministic service failures become controlled user messages. Repeated invalid source selection becomes a safe grounding fallback rather than ungrounded prose.

### Conversation threads

The graph uses LangGraph `InMemorySaver`. Invoke it with a `thread_id`; messages accumulate only in that thread, so follow-up planning sees earlier user and assistant turns. Different IDs are isolated. Memory is in-process demo state and disappears on restart. A production deployment should use a durable PostgreSQL or Redis-backed checkpointer.

### Development chat

Build the rules index first, configure `OPENAI_API_KEY` and `MAGIC_CHAT_MODEL`, then run:

```console
uv run python -m magic_assistant.agent.chat --pdf "data/MagicCompRules 20260417.pdf" --thread-id demo
```

Use `exit` or `quit` to stop. Reusing the same thread ID within the process preserves follow-up context.

## Known limitations

- The rules parser targets the supplied English Comprehensive Rules layout.
- BM25 has no translation layer; Spanish-to-English matching comes from the multilingual dense model.
- RRF is a transparent baseline without a neural reranker or learned weighting.
- NumPy exact search is intentionally sized for the current 2,091 static chunks, not a large dynamic corpus.
- The card API exposes printings rather than a canonical Oracle database; name-based deduplication is intentionally simple.
- Card range filters may require multiple API pages because numeric comparisons are enforced client-side.
- Live model download and card API inspection require working external network and TLS trust configuration.
- The live conversational provider is OpenAI; offline tests use injected deterministic fakes.
- Grounding validation proves citation IDs came from supplied evidence; it does not perform claim-level semantic entailment verification.
- Demo conversation memory is process-local and not durable.
- This phase has no FastAPI, Streamlit, production deployment, or multi-agent system.
