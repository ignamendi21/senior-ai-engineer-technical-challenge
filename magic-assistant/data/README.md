# Local source data

Place the separately supplied Comprehensive Rules PDF here with this exact filename:

```text
MagicCompRules 20260417.pdf
```

From `magic-assistant/`, the resulting path is `data/MagicCompRules 20260417.pdf`.

The source PDF is intentionally ignored by Git and must not be committed. Tests use small inline fixtures by default. The optional smoke test parses this local file when present, or a file identified by the `MAGIC_RULES_PDF` environment variable.
