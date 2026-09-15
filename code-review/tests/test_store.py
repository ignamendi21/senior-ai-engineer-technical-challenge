import pytest

from improved.chunking import chunk_documents, stable_chunk_id
from improved.errors import IncompatibleIndexError
from improved.models import DocumentChunk, SourceDocument
from improved.store import ChromaVectorStore


def make_store(path, model: str = "embedding-v1", chunk_size: int = 100) -> ChromaVectorStore:
    return ChromaVectorStore(path, "docs", model, chunk_size, 10)


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
    store = make_store(tmp_path)
    store.upsert(chunks, [[1.0, 0.0], [0.0, 1.0]])
    store.upsert(chunks, [[1.0, 0.0], [0.0, 1.0]])

    reopened = make_store(tmp_path)
    results = reopened.search([1.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "chunk-a"
    assert results[0].source_id == "source-a"
    assert results[0].metadata == {"filename": "guide.txt"}


def test_chroma_rejects_embedding_model_mismatch(tmp_path):
    chunk = make_chunk("chunk-a", 0, "alpha reference")
    make_store(tmp_path).upsert([chunk], [[1.0, 0.0]])

    with pytest.raises(IncompatibleIndexError, match="incompatible"):
        make_store(tmp_path, model="embedding-v2")


def test_chroma_rejects_chunking_configuration_mismatch(tmp_path):
    chunk = make_chunk("chunk-a", 0, "alpha reference")
    make_store(tmp_path).upsert([chunk], [[1.0, 0.0]])

    with pytest.raises(IncompatibleIndexError, match="incompatible"):
        make_store(tmp_path, chunk_size=200)
