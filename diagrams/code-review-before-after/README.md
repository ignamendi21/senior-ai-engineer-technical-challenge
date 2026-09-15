# Code Review Before / After

## What you are looking at

A visual contrast between the supplied `original.py` global script and the compact improved RAG reference. The split represents change and test boundaries, not a claim that every small script requires many files.

## How to read it

Read the red “Before” boundary first: one import-time script owns the key, legacy provider calls, in-memory Chroma, privileged prompt construction, and global history file. Then read “After” around `RagService` and its four injected protocols.

## Interview points

- Stable chunks, batch embeddings, persistent upserts, and compatibility metadata make ingestion repeatable.
- Retrieved documents remain untrusted reference data outside the system policy.
- Provider/store/history protocols enable offline tests without framework complexity.
- Session history and context are bounded; sessions do not share turns.
- Selected chunk IDs are checked against the supplied context before citations are returned.

`diagram.json` is the Archify source. `diagram.html` is the self-contained interactive export; `diagram.svg` is the static vector export; `diagram.png` is the high-resolution light export for documents.
