# Local data

Place the separately supplied Comprehensive Rules PDF here with this exact filename:

```text
MagicCompRules 20260417.pdf
```

From `magic-assistant/`, the resulting path is `data/MagicCompRules 20260417.pdf`.

The source PDF is ignored by Git and must not be committed. Tests use small inline fixtures by default. The optional smoke test parses this local file when present, or a file identified by `MAGIC_RULES_PDF`.

`retrieval_benchmark.json` is the committed bilingual retrieval benchmark. Running the index command creates `data/index/` with embeddings and chunk metadata. That generated directory is ignored by Git and should be rebuilt explicitly when the corpus or embedding model changes.
