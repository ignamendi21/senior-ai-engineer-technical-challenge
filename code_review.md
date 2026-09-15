# Part 2 — Senior Code Review of the Legacy RAG Helper

## Executive summary

The supplied code is a proof of concept, not a safe or dependable application component. Its most urgent defects are a credential embedded in source, untrusted retrieved text promoted into the system prompt, and global plaintext conversation persistence. It also uses retired OpenAI APIs and model assumptions, creates a non-persistent vector collection at import time, has no index compatibility contract, embeds whole documents one request at a time, and returns answers without validated provenance.

The reference rewrite under `code-review/improved/` keeps the example intentionally small while correcting the core boundaries: environment configuration, injected providers, deterministic chunking and IDs, batched embeddings, idempotent persistent storage, model/index compatibility, bounded retrieval context, session-scoped memory, untrusted-reference prompt separation, and deterministic citation validation.

If the key shown in the original were ever real, the first operational action would be revocation and rotation—not merely deleting it from the latest commit, because it may remain in Git history and downstream logs or clones.

## Prioritization

| Severity | Finding | Production impact | Primary fix |
| --- | --- | --- | --- |
| Critical | API secret embedded in source | Credential theft, unauthorized spend and data access | Revoke/rotate; load secrets from environment or secret manager |
| Critical | Retrieved text concatenated into the system prompt | Prompt injection can override intended behavior at the most privileged prompt level | Keep policy in system message; mark retrieval as untrusted user-side reference data |
| High | One shared plaintext `history.json` | Cross-user disclosure, corruption, races, unbounded retention | Session-scoped bounded memory; production encryption, isolation and retention controls |
| High | Obsolete OpenAI APIs/models | Runtime breakage, unsupported behavior, blocked upgrades | Current injected client and environment-selected models |
| High | Ephemeral collection created at import | Data loss on restart, duplicate-name failures, non-testable startup side effects | Persistent client, lazy construction and `get_or_create_collection` |
| High | No embedding/index compatibility contract | Queries can compare vectors from incompatible models/corpora | Persist and validate embedding model and index schema metadata |
| High | Weak positional IDs and non-idempotent `add` | Collisions and failed or duplicated re-ingestion | Content-derived stable IDs and `upsert` |
| High | No provenance or citation validation | Unsupported answers cannot be audited | Typed retrieved chunks and allowlisted deterministic citations |
| Medium | Whole-document, one-by-one embedding | Poor retrieval precision, excessive latency/cost and rate-limit pressure | Deterministic chunking with overlap and batched requests |
| Medium | Unbounded prompt/history/context | Token-limit failures, escalating cost and latency | Bound history, top-k and reference characters |
| Medium | Fixed top five treated as relevant | Irrelevant context can be presented as evidence | Validate `top_k`, expose distances, evaluate thresholds/ranking empirically |
| Medium | Tight globals and no dependency injection/tests | Fragile changeability and impossible offline verification | Small protocols, constructor injection and deterministic fakes |
| Medium | No timeout/retry/error taxonomy | Hangs, transient failure amplification and secret-bearing raw errors | Configured client timeout/retries and sanitized domain exceptions |
| Medium | No auth, rate limits, logging or quality gates | Abuse and poor incident diagnosis in a service deployment | Add at the service boundary in a production implementation |
| Low | Unclosed file and implicit encoding | Partial writes, platform inconsistency | Context managers and explicit encoding—or no default disk history |

## Detailed findings

### Hard-coded API secret — Critical

**Problem:** `API_KEY` is module-level source code and is passed repeatedly into calls.

**Risk:** Source control, code review systems, build logs, packages, screenshots and developer machines can expose the credential. Removing it later does not remove Git history. A key can enable unauthorized usage and cost.

**Recommendation:** Revoke and rotate any real exposed key. Load `OPENAI_API_KEY` at runtime, keep `.env` ignored, and use a managed secret store in production. Never log the value.

**Improved implementation:** `RagSettings.from_environment()` validates names and limits; `build_openai_rag()` creates the current SDK client from environment configuration.

### Retrieved data promoted to system instructions — Critical

**Problem:** `"Responde usando: " + context` places arbitrary retrieved text in the system message. Retrieved documents are joined with one space, losing source boundaries; an empty collection yields a misleading system instruction with no reference material.

**Risk:** A document containing “ignore previous instructions,” data-exfiltration requests, or misleading policy is elevated into the most trusted prompt channel. Retrieval is a trust boundary: indexed content is data, not policy.

**Recommendation:** Keep invariant behavior in a fixed system message. Put clearly delimited reference material in a user message and state that it may contain instructions that must not be followed.

**Improved implementation:** `RagService.build_messages()` creates a fixed system policy and a separate `<reference_material trust="untrusted">` block. This reduces privilege confusion but does not completely solve prompt injection.

**Defense in depth for production:** source allowlisting, ingestion scanning, retrieval filtering, least-privilege tools, output/source validation, monitoring, adversarial evaluations and incident controls.

### Global plaintext conversation persistence — High

**Problem:** Every call mutates a caller-owned list and overwrites one global `history.json` without a context manager, encoding, locking, atomic replacement or session identity.

**Risk:** Concurrent writes can corrupt the file; users can read each other's conversations; data grows indefinitely; sensitive content has no retention/deletion policy; a crash can leave a partial file. The file is never read back, so this is effectively a write-only debug snapshot rather than recoverable session persistence.

**Recommendation:** Scope history by session and bound retained turns. Do not persist by default in this small helper.

**Improved implementation:** `InMemorySessionHistory` isolates `session_id` values, bounds turns and uses a lock. A real service would require encrypted durable storage, tenant/user access control, explicit retention, deletion/export lifecycle and audit policy.

### Retired OpenAI interface and fixed model assumptions — High

**Problem:** `openai.Embedding.create` and `openai.ChatCompletion.create` are legacy module APIs. `text-embedding-ada-002` and `gpt-4` are embedded assumptions rather than deployment configuration.

**Risk:** SDK upgrades break the code. Model availability, context limits, cost and behavior differ by account and over time.

**Recommendation:** Use an injected current `OpenAI` client and select `RAG_CHAT_MODEL` and `RAG_EMBEDDING_MODEL` through configuration. Configure timeout and bounded SDK retries at client creation.

**Improved implementation:** `OpenAIEmbeddingProvider` uses `client.embeddings.create`; `OpenAIChatProvider` uses the Responses structured parse API. Provider protocols keep all tests offline. The original also blindly indexes `choices[0]` without checking whether choices/content exist or whether generation was truncated or filtered; structured parsing and explicit missing-output errors replace that assumption.

### Import-time global side effects and collection lifecycle — High

**Problem:** Importing the module creates clients and a collection. Reloading/re-executing it in the same interpreter can make `create_collection("docs")` fail because the collection already exists; a new process instead starts with a fresh, empty default in-memory client. There is no entry-point guard, and the credential, client and collection are global mutable module state.

**Risk:** Data vanishes on restart, imports become environment-dependent, tests interfere, and re-deployment/re-import behavior is undefined.

**Recommendation:** Construct dependencies explicitly. Use `PersistentClient`, configurable paths and `get_or_create_collection`.

**Improved implementation:** `ChromaVectorStore` is created by composition code, persists locally, uses an idempotent collection lifecycle, and validates stored index metadata.

### Missing embedding compatibility contract — High

**Problem:** Nothing records which embedding model produced stored vectors.

**Risk:** Changing a model while reusing a collection can produce equal-dimensional but semantically incompatible vectors. Persistence alone does not make an index valid.

**Recommendation:** Store embedding model, index schema and chunking configuration metadata and fail clearly on mismatch. In a larger system, include corpus/version fingerprints and rebuild workflows.

### Weak IDs and undefined re-ingestion — High

**Problem:** IDs restart at `"0"` for every call. `collection.add` fails on collisions or encourages ad hoc collection resets.

**Risk:** Different sources collide; retries are not idempotent; operators cannot safely resume ingestion.

**Recommendation:** Hash source identity, chunk position and chunk content. Use `upsert` so retrying the same ingestion converges to the same state.

### No source provenance — High

**Problem:** Results lose source identity, chunk position and metadata. The answer contains no citations.

**Risk:** Users cannot verify claims, debug retrieval or identify stale/incorrect source material. Asking an LLM to invent citations would not fix this.

**Recommendation:** Return typed `RetrievedChunk` records. Let generation select only provided chunk IDs, validate that allowlist deterministically, then derive `SourceCitation` values in code.

### Inefficient and low-precision ingestion — Medium

**Problem:** Each whole document is embedded with a separate network call.

**Risk:** Long documents dilute local concepts and can exceed input limits. Serial calls increase latency, cost and rate-limit exposure.

**Recommendation:** Split generic text deterministically with overlap, then embed chunks in batches. Overlap preserves context near boundaries but increases storage and duplicate evidence; chunk size must be evaluated for the actual domain rather than treated as universal.

### Retrieval and confidence assumptions — Medium

**Problem:** `n_results=5` is unconditional. Distances are discarded. There is no threshold, lexical complement, reranker or benchmark.

**Risk:** “Top five” means only the five nearest available items; it does not guarantee that any item is relevant. Raw vector distance and BM25 scores are model/corpus/query dependent, not calibrated confidence probabilities.

**Recommendation:** Validate and configure top-k, retain scores for diagnosis, cap context, and use a retrieval benchmark to choose ranking/threshold behavior. Hybrid search and reranking may be justified by evaluation, but are intentionally omitted from this compact rewrite.

### Conversation-unaware retrieval and unlimited prompts — Medium

**Problem:** Retrieval embeds only the latest question while all history is appended to generation indefinitely.

**Risk:** Follow-ups may retrieve the wrong subject, while generation eventually exceeds context limits and becomes slow and expensive.

**Recommendation:** Build a bounded retrieval query from recent user turns plus the current question; independently bound retained history, retrieved chunk count and reference characters.

### Coupling, typing and testability — Medium

**Problem:** Provider calls, storage, retrieval, prompting, generation and persistence are one global procedure. `history: list` has no shape contract.

**Risk:** Components cannot be tested or replaced independently; malformed turns fail late; operational changes require risky edits.

**Recommendation:** Use small protocols and typed domain models. Inject embedding, store, chat and history dependencies into one compact `RagService`.

### Reliability, observability and service controls — Medium

**Problem:** There are no useful exceptions, timeout/retry policy, structured logs, metrics, authentication or rate limiting.

**Risk:** Transient failures become raw crashes or hangs. Raw provider errors may reveal sensitive request details. If exposed as a service, anyone can consume paid resources.

**Recommendation:** Translate provider/store failures into sanitized domain exceptions while retaining exception chaining for operators. Configure SDK timeout/retries. A production service should add authenticated identities, quotas/rate limits, structured logs, metrics, tracing, alerts and request correlation without logging prompts or secrets by default.

### Missing engineering quality gates — Medium

There are no tests, lint/format configuration, type-oriented contracts, CI checks, retrieval evaluations or concurrency tests. A production repository should run unit/integration/security checks in CI/CD and gate index/model changes with evaluations.

## Improved design

```text
SourceDocument
      |
      v
Deterministic overlapping chunker
      |
      v
EmbeddingProvider.embed_documents (batched)
      |
      v
VectorStore.upsert (stable IDs + metadata + compatibility)

question + bounded session history
      |
      v
EmbeddingProvider.embed_query
      |
      v
VectorStore.search -> RetrievedChunk[]
      |
      v
fixed system policy + untrusted reference block + bounded history
      |
      v
ChatProvider.generate -> text + selected chunk IDs
      |
      v
deterministic ID validation -> RagAnswer + SourceCitation[]
```

### Dependency boundaries

- `EmbeddingProvider`: document/query vectors.
- `VectorStore`: idempotent upsert and structured retrieval.
- `ChatProvider`: structured answer draft using supplied messages.
- `SessionHistory`: isolated bounded turns.
- `RagService`: orchestration only.

Most tests use in-memory fakes. One focused test exercises real local Chroma persistence and compatibility behavior.

## Senior-level design decisions

### Why references do not belong in the system prompt

The system channel defines policy and trust. Retrieved documents can be stale, malicious or user-controlled. Mixing them gives data policy-level authority. Delimiting them in a lower-trust message preserves the distinction, though model-level prompt injection still requires defense in depth.

### Why persistence is insufficient

A persistent directory can faithfully retain invalid vectors. Correct loading also requires agreement on embedding model, schema and corpus/version. Model compatibility is part of data integrity.

### Why history is scoped and bounded

Session scope prevents accidental cross-user context. Bounding protects context windows, latency and cost. Production durability adds privacy obligations—encryption, authorization, retention and deletion—not just a database table.

### Why scores and top-k are not confidence

Ranking scores are relative to one index and algorithm. Their scale changes with model, normalization and corpus. `top_k=3` asks for three ranked candidates; it does not assert that three relevant documents exist. Evaluate retrieval and calibrate any rejection threshold empirically.

### Why idempotent ingestion matters

Jobs retry after timeouts and deployments restart. Stable IDs plus upsert make repeated work converge instead of duplicating records or requiring destructive resets. This reduces operational recovery risk.

### Why RAG quality must be measured

A functioning vector query says nothing about answer relevance. A representative benchmark should measure retrieval hits/ranks, grounded answer correctness, citation accuracy, prompt-injection resilience, latency and cost across expected languages and document types.

## Trade-offs and intentionally omitted work

This reference rewrite intentionally does **not** add:

- Web API or authentication
- Rate limiting
- Durable multi-user conversation storage
- Hybrid lexical retrieval
- Neural reranking
- Agent or graph framework
- Full observability platform
- Distributed deployment, Redis, PostgreSQL or Kubernetes
- Domain-specific parsers

Those capabilities may be necessary in a production product, but they would obscure the corrections to this small RAG helper. The rewrite is reference-quality and testable, not production-complete.

## Production follow-up

A real service would add a secret manager, authenticated tenant identity, authorization, quotas, encrypted durable session storage, retention/deletion workflows, content/source governance, structured metrics/logs/traces, offline and online RAG evaluations, safety tests, CI/CD, dependency scanning, index migration/rebuild tooling, backups and capacity/latency controls.
