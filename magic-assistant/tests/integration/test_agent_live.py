import os
from pathlib import Path

import pytest

from magic_assistant.agent.runtime import LiveAssistantRuntime, RuntimeSettings


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_LLM_TESTS") != "1",
    reason="Set RUN_LIVE_LLM_TESTS=1 to enable the OpenAI agent smoke test",
)
def test_live_rules_question():
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("MAGIC_CHAT_MODEL"):
        pytest.skip("OPENAI_API_KEY and MAGIC_CHAT_MODEL are required")
    settings = RuntimeSettings(
        rules_pdf=Path(os.environ.get("MAGIC_RULES_PDF", "data/MagicCompRules 20260417.pdf")),
        index_directory=Path(os.environ.get("MAGIC_INDEX_DIR", "data/index")),
    )
    if (
        not settings.rules_pdf.is_file()
        or not (settings.index_directory / "manifest.json").is_file()
    ):
        pytest.skip("The local rules PDF and built index are required")

    with LiveAssistantRuntime.create(settings) as runtime:
        state = runtime.invoke("What are the phases of a turn?", thread_id="live-smoke")

    assert state["final_answer"]
    assert "Sources:" in state["final_answer"]
