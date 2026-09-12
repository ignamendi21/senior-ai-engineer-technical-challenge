from collections.abc import Sequence

import numpy as np
import pytest

from magic_assistant.rules.chunker import RuleChunk
from magic_assistant.rules.embeddings import SentenceTransformerEmbeddingProvider
from magic_assistant.rules.index import (
    DenseRuleIndex,
    RuleIndexError,
    corpus_fingerprint,
)


class FakeEmbeddingProvider:
    model_name = "fake-embeddings"

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([[index + 1.0, 1.0] for index, _ in enumerate(texts)], dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


class RecordingModel:
    def __init__(self) -> None:
        self.inputs: list[str | list[str]] = []

    def encode(self, texts, **kwargs):
        self.inputs.append(texts)
        count = len(texts) if isinstance(texts, list) else 1
        values = np.ones((count, 2), dtype=np.float32)
        return values if isinstance(texts, list) else values[0]


def make_chunk(rule_id: str, text: str) -> RuleChunk:
    return RuleChunk(
        chunk_id=f"rule:{rule_id}:1",
        rule_ids=[rule_id],
        root_rule_id=rule_id,
        parent_rule_id=rule_id.split(".")[0],
        chapter_id="7",
        section_id="702",
        text=text,
        page_start=1,
        page_end=1,
        rules_version="2026-04-17",
        document_type="rule",
    )


def test_sentence_transformer_provider_uses_e5_prefixes():
    provider = SentenceTransformerEmbeddingProvider.__new__(SentenceTransformerEmbeddingProvider)
    provider._model_name = "fake-e5"
    provider._show_progress = False
    provider._model = RecordingModel()

    provider.encode_documents(["First strike", "Ninjutsu"])
    provider.encode_query("¿Cómo funciona ninjutsu?")

    assert provider._model.inputs == [
        ["passage: First strike", "passage: Ninjutsu"],
        "query: ¿Cómo funciona ninjutsu?",
    ]


def test_builds_normalized_dense_index():
    chunks = [make_chunk("702.7", "First strike"), make_chunk("702.49", "Ninjutsu")]

    index = DenseRuleIndex.build(chunks, FakeEmbeddingProvider())

    assert index.embeddings.shape == (2, 2)
    assert np.allclose(np.linalg.norm(index.embeddings, axis=1), 1.0)
    assert index.manifest.embedding_model == "fake-embeddings"
    assert index.manifest.rules_version == "2026-04-17"
    assert index.manifest.corpus_fingerprint == corpus_fingerprint(chunks)


def test_round_trips_persisted_index(tmp_path):
    chunks = [make_chunk("702.7", "First strike"), make_chunk("702.49", "Ninjutsu")]
    built = DenseRuleIndex.build(chunks, FakeEmbeddingProvider())
    built.save(tmp_path)

    loaded = DenseRuleIndex.load(
        tmp_path,
        expected_chunks=chunks,
        expected_model="fake-embeddings",
    )

    assert [chunk.chunk_id for chunk in loaded.chunks] == [chunk.chunk_id for chunk in chunks]
    assert np.array_equal(loaded.embeddings, built.embeddings)
    assert loaded.manifest == built.manifest


def test_rejects_stale_corpus(tmp_path):
    chunks = [make_chunk("702.7", "First strike")]
    DenseRuleIndex.build(chunks, FakeEmbeddingProvider()).save(tmp_path)
    changed = [make_chunk("702.7", "Changed content")]

    with pytest.raises(RuleIndexError, match="corpus fingerprint"):
        DenseRuleIndex.load(
            tmp_path,
            expected_chunks=changed,
            expected_model="fake-embeddings",
        )


def test_rejects_incompatible_model(tmp_path):
    chunks = [make_chunk("702.7", "First strike")]
    DenseRuleIndex.build(chunks, FakeEmbeddingProvider()).save(tmp_path)

    with pytest.raises(RuleIndexError, match="embedding model"):
        DenseRuleIndex.load(
            tmp_path,
            expected_chunks=chunks,
            expected_model="different-model",
        )


def test_fingerprint_is_deterministic_and_metadata_sensitive():
    chunk = make_chunk("702.49", "Ninjutsu")
    same = make_chunk("702.49", "Ninjutsu")
    changed = same.model_copy(update={"page_start": 2, "page_end": 2})

    assert corpus_fingerprint([chunk]) == corpus_fingerprint([same])
    assert corpus_fingerprint([chunk]) != corpus_fingerprint([changed])
