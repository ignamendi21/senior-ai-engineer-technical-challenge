import argparse
from collections.abc import Sequence
from pathlib import Path

from magic_assistant.agent.graph import (
    AssistantDependencies,
    MagicAssistant,
    build_assistant_graph,
)
from magic_assistant.agent.llm import (
    OpenAICustomCardGenerator,
    OpenAIGroundedAnswerGenerator,
    OpenAIRequestPlanner,
    create_openai_model_from_environment,
)
from magic_assistant.cards.client import MtgApiClient
from magic_assistant.cards.service import CardSearchService
from magic_assistant.rules.index import DEFAULT_INDEX_DIRECTORY
from magic_assistant.rules.search import DEFAULT_PDF_PATH, load_knowledge_base


def create_live_assistant(
    pdf_path: Path,
    index_directory: Path,
    card_client: MtgApiClient,
) -> MagicAssistant:
    model = create_openai_model_from_environment()
    knowledge_base = load_knowledge_base(pdf_path, index_directory)
    dependencies = AssistantDependencies(
        planner=OpenAIRequestPlanner(model),
        answer_generator=OpenAIGroundedAnswerGenerator(model),
        custom_card_generator=OpenAICustomCardGenerator(model),
        rules_retriever=knowledge_base,
        card_searcher=CardSearchService(card_client),
    )
    return MagicAssistant(build_assistant_graph(dependencies))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the development Magic assistant chat")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF_PATH)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIRECTORY)
    parser.add_argument("--thread-id", default="demo")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    with MtgApiClient() as card_client:
        assistant = create_live_assistant(arguments.pdf, arguments.index_dir, card_client)
        while True:
            try:
                question = input("You> ").strip()
            except EOFError:
                break
            if question.casefold() in {"exit", "quit"}:
                break
            if not question:
                continue
            print(f"Assistant> {assistant.ask(question, thread_id=arguments.thread_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
