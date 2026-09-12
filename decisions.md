# Architectural Decision Log

## ADR-001 — Structure-aware parsing and chunking of the Magic rules

**Status:** Accepted

### Context

The Magic: The Gathering Comprehensive Rules already exposes meaningful identifiers and hierarchy: chapters, sections, numbered rules, lettered subrules, examples, cross-references, and glossary entries. These identifiers are useful provenance for later citations and are more informative than arbitrary chunk numbers.

### Decision

Parse the document into typed rule and glossary records while preserving identifiers, hierarchy, text, physical page provenance, version, and obvious cross-references. Group each numbered parent rule with its lettered subrules when the group fits the configured size. If a group is too large, split only at rule/subrule boundaries. Keep an individually oversized subrule intact rather than splitting it arbitrarily.

The measured parent/subrule p95 is 2,358 characters, so the initial default maximum is 3,000 characters. This preserves 96.88% of measured semantic groups intact while handling the long-tail groups at explicit boundaries.

### Alternatives considered

Generic fixed-character splitting, including recursive character splitting with overlap.

### Why rejected

Fixed-character splitting can break a rule or example across chunks, obscure the relationship between a parent and its subrules, reduce citation granularity, duplicate context through overlap, and make generated answers less explainable.

### Consequences

The ingestion code is slightly more domain-specific and may need adjustment if the source PDF layout changes. In return, each chunk has defensible provenance, stable rule-level citation metadata, and boundaries aligned with the source's meaning. The maximum size is a soft limit when preserving a single unusually large subrule.

## ADR-002 — Hybrid lexical and multilingual semantic retrieval

**Status:** Accepted

### Context

Magic rules contain exact identifiers and domain terms, while users may express conceptual questions in English or Spanish. Lexical matching is strong for identifiers and names such as “Ninjutsu,” but cannot reliably connect Spanish questions to an English-only corpus. Dense retrieval handles cross-language meaning but can be less precise for exact identifiers.

### Decision

Rank every query independently with BM25 and normalized `intfloat/multilingual-e5-small` embeddings, then combine the rankings with configurable Reciprocal Rank Fusion. Use E5's required `query:` and `passage:` prefixes. Resolve explicit rule identifiers deterministically before hybrid ranking and permit only one-hop glossary-to-rule expansion.

`multilingual-e5-small` was selected because it supports the required English/Spanish retrieval, is small enough for a local technical demonstration, and avoids a proprietary embedding service. A provider protocol keeps model loading replaceable and allows deterministic unit tests.

### Alternatives considered

- Lexical-only retrieval: precise for exact terms but inadequate for cross-language and paraphrased questions.
- Dense-only retrieval: multilingual, but weaker for exact identifiers and rare domain strings.
- Proprietary embedding API: adds credentials, cost, network dependence, and external data processing.
- Neural reranker: may improve ranking but adds latency and complexity before a baseline demonstrates the need.

### Consequences

The application maintains two simple rankings and loads a local embedding model. RRF remains transparent and avoids score calibration between BM25 and cosine similarity. The initial bilingual benchmark is a baseline, not a mandatory performance threshold.

## ADR-003 — Lightweight local dense index instead of a vector database

**Status:** Accepted

### Context

The rules corpus contains 2,091 static semantic chunks. At this scale, exact NumPy similarity over one normalized embedding matrix is operationally simpler than running a vector database.

### Decision

Persist normalized embeddings in `embeddings.npy`, serialized chunk metadata in `chunks.jsonl`, and compatibility metadata in `manifest.json`. The manifest identifies the model, rules version, chunk count, embedding dimension, schema version, deterministic corpus fingerprint, and embedding-matrix fingerprint. Index loading validates finite unit vectors and fails clearly when artifacts, corpus, or model are incompatible; rebuilding is explicit.

Dense access is encapsulated behind `DenseRuleIndex` and `EmbeddingProvider`, allowing replacement if future scale or production requirements justify it.

### Alternatives considered

- Chroma: unnecessary local persistence and dependency complexity for approximately two thousand records.
- pgvector: requires a database service without a current operational need.
- External vector database: introduces network, credentials, cost, and lifecycle management.
- FAISS: fast, but NumPy exact search is sufficient at this corpus size and simpler to inspect.

### Consequences

Startup can reuse a validated local index without recomputing document embeddings. Generated index artifacts and model weights remain uncommitted. Search remains an exact linear scan over a small normalized matrix; this choice should be revisited if corpus size or latency requirements change materially.
