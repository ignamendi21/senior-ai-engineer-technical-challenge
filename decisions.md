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
