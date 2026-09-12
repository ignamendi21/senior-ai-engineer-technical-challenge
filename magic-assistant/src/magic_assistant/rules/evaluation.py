import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from magic_assistant.rules.embeddings import DEFAULT_EMBEDDING_MODEL
from magic_assistant.rules.index import DEFAULT_INDEX_DIRECTORY
from magic_assistant.rules.retrieval import RuleEvidence, RulesKnowledgeBase
from magic_assistant.rules.search import DEFAULT_PDF_PATH, load_knowledge_base

DEFAULT_BENCHMARK_PATH = Path("data/retrieval_benchmark.json")


class RetrievalCase(BaseModel):
    query: str
    expected_rule_ids: list[str]
    language: str


class RetrievalMetrics(BaseModel):
    case_count: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float


def load_cases(path: Path) -> list[RetrievalCase]:
    return [RetrievalCase.model_validate(item) for item in json.loads(path.read_text("utf-8"))]


def evaluate_retrieval(
    knowledge_base: RulesKnowledgeBase,
    cases: Sequence[RetrievalCase],
) -> RetrievalMetrics:
    if not cases:
        raise ValueError("Retrieval benchmark must contain at least one case")
    ranks = []
    for case in cases:
        evidence = knowledge_base.search(case.query, top_k=knowledge_base.chunk_count)
        rank = next(
            (
                index
                for index, item in enumerate(evidence, 1)
                if _matches_expected(item, case.expected_rule_ids)
            ),
            None,
        )
        ranks.append(rank)
    count = len(ranks)
    return RetrievalMetrics(
        case_count=count,
        hit_at_1=sum(rank == 1 for rank in ranks) / count,
        hit_at_3=sum(rank is not None and rank <= 3 for rank in ranks) / count,
        hit_at_5=sum(rank is not None and rank <= 5 for rank in ranks) / count,
        mrr=sum(1 / rank for rank in ranks if rank is not None) / count,
    )


def _matches_expected(evidence: RuleEvidence, expected_rule_ids: Sequence[str]) -> bool:
    expected = set(expected_rule_ids)
    return evidence.root_rule_id in expected or bool(expected.intersection(evidence.rule_ids))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate rules retrieval against a small benchmark"
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF_PATH)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIRECTORY)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--cases", type=Path, default=DEFAULT_BENCHMARK_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    knowledge_base = load_knowledge_base(
        arguments.pdf,
        arguments.index_dir,
        model_name=arguments.model,
    )
    cases = load_cases(arguments.cases)
    metrics = evaluate_retrieval(knowledge_base, cases)
    print(f"Cases: {metrics.case_count}")
    print(f"Hit@1: {metrics.hit_at_1:.3f}")
    print(f"Hit@3: {metrics.hit_at_3:.3f}")
    print(f"Hit@5: {metrics.hit_at_5:.3f}")
    print(f"MRR: {metrics.mrr:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
