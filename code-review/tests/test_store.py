import pytest

from improved.chunking import chunk_documents, stable_chunk_id
from improved.errors import IncompatibleIndexError
from improved.models import DocumentChunk, SourceDocument
from improved.store import ChromaVectorStore


def make_chunk(identifier: str, index: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=identifier,
        source_id="source-a",
        text=text,
        chunk_index=index,
        metadata={"filename": "guide.txt"},
    )


def test_chunk_ids_are_deterministic_and_content_sensitive():
    first = stable_chunk_id("source", 0, "content")

    assert first == stable_chunk_id("source", 0, "content")
    assert first != stable_chunk_id("source", 1, "content")
    assert first != stable_chunk_id("source", 0, "changed")


def test_chunking_is_deterministic_with_overlap():
    document = SourceDocument(source_id="source", text="abcdefghij", metadata={"kind": "test"})

    first = chunk_documents([document], chunk_size=6, overlap=2)
    second = chunk_documents([document], chunk_size=6, overlap=2)

    assert first == second
    assert [chunk.text for chunk in first] == ["abcdef", "efghij"]
    assert all(chunk.metadata == {"kind": "test"} for chunk in first)


def test_chroma_persists_idempotent_upserts_and_provenance(tmp_path):
    chunks = [
        make_chunk("chunk-a", 0, "alpha reference"),
        make_chunk("chunk-b", 1, "beta reference"),
    ]
    store = ChromaVectorStore(tmp_path, "docs", "embedding-v1")
    store.upsert(chunks, [[1.0, 0.0], [0.0, 1.0]])
    store.upsert(chunks, [[1.0, 0.0], [0.0, 1.0]])

    reopened = ChromaVectorStore(tmp_path, "docs", "embedding-v1")
    results = reopened.search([1.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "chunk-a"
    assert results[0].source_id == "source-a"
    assert results[0].metadata == {"filename": "guide.txt"}


def test_chroma_rejects_embedding_model_mismatch(tmp_path):
    chunk = make_chunk("chunk-a", 0, "alpha reference")
    ChromaVectorStore(tmp_path, "docs", "embedding-v1").upsert([chunk], [[1.0, 0.0]])

    with pytest.raises(IncompatibleIndexError, match="incompatible"):
        ChromaVectorStore(tmp_path, "docs", "embedding-v2")
