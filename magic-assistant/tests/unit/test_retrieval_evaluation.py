import pytest

from magic_assistant.rules.evaluation import RetrievalCase, evaluate_retrieval
from magic_assistant.rules.retrieval import RuleEvidence


class StubKnowledgeBase:
    chunk_count = 3

    def search(self, query: str, top_k: int = 5) -> list[RuleEvidence]:
        identifiers = {
            "first": ["702.7", "702.49", "510"],
            "second": ["510", "702.49", "702.7"],
            "miss": ["510", "702.49", "117"],
        }[query]
        return [
            RuleEvidence(
                chunk_id=f"rule:{rule_id}:1",
                rule_ids=[rule_id],
                root_rule_id=rule_id,
                text=rule_id,
                page_start=1,
                page_end=1,
                rules_version="2026-04-17",
                score=1.0,
                retrieval_methods=["hybrid"],
                document_type="rule",
            )
            for rule_id in identifiers[:top_k]
        ]


def test_calculates_hit_rates_and_mrr():
    cases = [
        RetrievalCase(query="first", expected_rule_ids=["702.7"], language="en"),
        RetrievalCase(query="second", expected_rule_ids=["702.7"], language="en"),
        RetrievalCase(query="miss", expected_rule_ids=["999"], language="en"),
    ]

    metrics = evaluate_retrieval(StubKnowledgeBase(), cases)

    assert metrics.hit_at_1 == pytest.approx(1 / 3)
    assert metrics.hit_at_3 == pytest.approx(2 / 3)
    assert metrics.hit_at_5 == pytest.approx(2 / 3)
    assert metrics.mrr == pytest.approx((1 + 1 / 3) / 3)
