from collections.abc import Sequence

import numpy as np

from magic_assistant.rules.chunker import RuleChunk
from magic_assistant.rules.index import DenseRuleIndex
from magic_assistant.rules.retrieval import RulesKnowledgeBase


class FakeEmbeddingProvider:
    model_name = "fake-multilingual"

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        normalized = text.casefold()
        if "ninjutsu" in normalized or "sigilo" in normalized:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        if "first strike" in normalized or "dañar primero" in normalized:
            return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)


def make_chunk(
    chunk_id: str,
    *,
    rule_ids: list[str] | None = None,
    root_rule_id: str | None = None,
    title: str | None = None,
    term: str | None = None,
    text: str,
    related_rule_ids: list[str] | None = None,
    document_type: str = "rule",
    section_id: str = "702",
) -> RuleChunk:
    return RuleChunk(
        chunk_id=chunk_id,
        rule_ids=rule_ids or [],
        root_rule_id=root_rule_id,
        parent_rule_id=section_id if root_rule_id else None,
        chapter_id=section_id[0] if section_id else None,
        section_id=section_id or None,
        title=title,
        term=term,
        text=text,
        page_start=10,
        page_end=11,
        rules_version="2026-04-17",
        related_rule_ids=related_rule_ids or [],
        document_type=document_type,
    )


def build_knowledge_base() -> RulesKnowledgeBase:
    chunks = [
        make_chunk(
            "rule:702.49:1",
            rule_ids=["702.49", "702.49a"],
            root_rule_id="702.49",
            title="Ninjutsu",
            text="702.49. Ninjutsu. Return an unblocked attacking creature.",
        ),
        make_chunk(
            "rule:702.7:1",
            rule_ids=["702.7", "702.7a"],
            root_rule_id="702.7",
            title="First Strike",
            text="702.7. First strike creates an additional combat damage step.",
        ),
        make_chunk(
            "glossary:0001",
            term="Ninjutsu",
            text="Ninjutsu\nA keyword ability that lets a creature enter combat.",
            related_rule_ids=["702.49"],
            document_type="glossary",
            section_id="",
        ),
        make_chunk(
            "rule:510:1",
            rule_ids=["510"],
            root_rule_id="510",
            title="Combat Damage Step",
            text="510. Combat Damage Step.",
            section_id="510",
        ),
    ]
    provider = FakeEmbeddingProvider()
    return RulesKnowledgeBase(chunks, DenseRuleIndex.build(chunks, provider), provider)


def test_exact_subrule_lookup_uses_prebuilt_index():
    evidence = build_knowledge_base().get_rule("702.49a")

    assert [item.chunk_id for item in evidence] == ["rule:702.49:1"]
    assert evidence[0].retrieval_methods == ["exact"]


def test_root_lookup_returns_all_root_partitions():
    knowledge_base = build_knowledge_base()
    first = knowledge_base._chunks[0]
    continuation = first.model_copy(update={"chunk_id": "rule:702.49:2", "rule_ids": ["702.49b"]})
    chunks = [first, continuation, *knowledge_base._chunks[1:]]
    provider = FakeEmbeddingProvider()
    retriever = RulesKnowledgeBase(chunks, DenseRuleIndex.build(chunks, provider), provider)

    assert [item.chunk_id for item in retriever.get_rule("702.49")] == [
        "rule:702.49:1",
        "rule:702.49:2",
    ]


def test_explicit_rule_id_is_ranked_first():
    evidence = build_knowledge_base().search("What does rule 702.49a mean?", top_k=3)

    assert evidence[0].chunk_id == "rule:702.49:1"
    assert evidence[0].retrieval_methods == ["exact"]


def test_hybrid_retrieval_reports_contributing_methods():
    evidence = build_knowledge_base().search("first strike", top_k=2)
    first_strike = next(item for item in evidence if item.root_rule_id == "702.7")

    assert first_strike.retrieval_methods == ["lexical", "semantic", "hybrid"]


def test_fake_multilingual_semantic_retrieval_is_injectable():
    evidence = build_knowledge_base().search("ataque con sigilo", top_k=2)

    assert any(item.root_rule_id == "702.49" for item in evidence)
    assert "semantic" in evidence[0].retrieval_methods


def test_glossary_expands_one_hop_to_related_rule_without_duplicates():
    evidence = build_knowledge_base().search("Ninjutsu", top_k=3)

    assert evidence[0].document_type == "glossary"
    assert evidence[1].root_rule_id == "702.49"
    assert evidence[1].retrieval_methods == ["glossary_expansion"]
    assert len({item.chunk_id for item in evidence}) == len(evidence)


def test_evidence_preserves_provenance_metadata():
    evidence = build_knowledge_base().search("first strike", top_k=1)[0]

    assert evidence.rule_ids == ["702.7", "702.7a"]
    assert evidence.page_start == 10
    assert evidence.page_end == 11
    assert evidence.rules_version == "2026-04-17"
