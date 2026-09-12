import argparse
from collections.abc import Sequence
from pathlib import Path

from magic_assistant.rules.chunker import RulesChunker
from magic_assistant.rules.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbeddingProvider,
)
from magic_assistant.rules.index import DEFAULT_INDEX_DIRECTORY, DenseRuleIndex
from magic_assistant.rules.parser import ComprehensiveRulesParser
from magic_assistant.rules.retrieval import RulesKnowledgeBase

DEFAULT_PDF_PATH = Path("data/MagicCompRules 20260417.pdf")


def load_knowledge_base(
    pdf_path: Path,
    index_directory: Path,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> RulesKnowledgeBase:
    parsed = ComprehensiveRulesParser().parse_pdf(pdf_path)
    chunks = RulesChunker().chunk(parsed)
    provider = SentenceTransformerEmbeddingProvider(model_name)
    index = DenseRuleIndex.load(
        index_directory,
        expected_chunks=chunks,
        expected_model=provider.model_name,
    )
    return RulesKnowledgeBase(chunks, index, provider)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search the local Magic rules index")
    parser.add_argument("query")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF_PATH)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIRECTORY)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    knowledge_base = load_knowledge_base(
        arguments.pdf,
        arguments.index_dir,
        model_name=arguments.model,
    )
    for rank, evidence in enumerate(
        knowledge_base.search(arguments.query, top_k=arguments.top_k), 1
    ):
        label = evidence.title or evidence.term or "Untitled"
        identifiers = ", ".join(evidence.rule_ids) or "glossary"
        methods = ", ".join(evidence.retrieval_methods)
        preview = " ".join(evidence.text.split())[:240]
        print(f"{rank}. {identifiers} | root={evidence.root_rule_id or '-'} | {label}")
        print(
            f"   pages={evidence.page_start}-{evidence.page_end} "
            f"methods={methods} score={evidence.score:.6f}"
        )
        print(f"   {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
