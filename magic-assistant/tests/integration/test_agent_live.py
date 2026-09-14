import os
from pathlib import Path

import pytest

from magic_assistant.agent.chat import create_live_assistant
from magic_assistant.cards.client import MtgApiClient


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_LLM_TESTS") != "1",
    reason="Set RUN_LIVE_LLM_TESTS=1 to enable the OpenAI agent smoke test",
)
def test_live_rules_question():
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("MAGIC_CHAT_MODEL"):
        pytest.skip("OPENAI_API_KEY and MAGIC_CHAT_MODEL are required")
    pdf_path = Path(os.environ.get("MAGIC_RULES_PDF", "data/MagicCompRules 20260417.pdf"))
    index_directory = Path(os.environ.get("MAGIC_RULES_INDEX", "data/index"))
    if not pdf_path.is_file() or not (index_directory / "manifest.json").is_file():
        pytest.skip("The local rules PDF and built index are required")

    with MtgApiClient() as card_client:
        assistant = create_live_assistant(pdf_path, index_directory, card_client)
        answer = assistant.ask("What are the phases of a turn?", thread_id="live-smoke")

    assert answer
    assert "Sources:" in answer
