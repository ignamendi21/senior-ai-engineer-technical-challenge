import pytest

from magic_assistant.rules.retrieval import reciprocal_rank_fusion, tokenize


def test_tokenizer_preserves_rule_identifiers_and_magic_terms():
    assert tokenize("Rule 702.49a: First-Strike isn't ninjutsu") == [
        "rule",
        "702.49a",
        "first-strike",
        "isn't",
        "ninjutsu",
    ]


def test_rrf_combines_rankings_transparently():
    fused = reciprocal_rank_fusion(
        [["a", "b", "c"], ["b", "c", "d"]],
        constant=10,
    )

    assert [identifier for identifier, _ in fused] == ["b", "c", "a", "d"]
    assert fused[0][1] == pytest.approx(1 / 12 + 1 / 11)


def test_rrf_deduplicates_within_ranking_and_breaks_ties_by_identifier():
    fused = reciprocal_rank_fusion([["b", "b"], ["a"]], constant=60)

    assert fused == [("a", pytest.approx(1 / 61)), ("b", pytest.approx(1 / 61))]


def test_rrf_requires_positive_constant():
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion([["a"]], constant=0)
