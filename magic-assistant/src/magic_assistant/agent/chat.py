import argparse
from collections.abc import Sequence
from pathlib import Path

from magic_assistant.agent.runtime import LiveAssistantRuntime, RuntimeSettings


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the development Magic assistant chat")
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--thread-id", default="demo")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    environment = RuntimeSettings.from_environment()
    settings = RuntimeSettings(
        rules_pdf=arguments.pdf or environment.rules_pdf,
        index_directory=arguments.index_dir or environment.index_directory,
    )
    with LiveAssistantRuntime.create(settings) as runtime:
        while True:
            try:
                question = input("You> ").strip()
            except EOFError:
                break
            if question.casefold() in {"exit", "quit"}:
                break
            if not question:
                continue
            state = runtime.invoke(question, thread_id=arguments.thread_id)
            print(f"Assistant> {state['final_answer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
