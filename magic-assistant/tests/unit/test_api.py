from fastapi.testclient import TestClient

from magic_assistant.agent.schemas import (
    CustomCard,
    CustomCardDraft,
    GroundedAnswerDraft,
    RequestIntent,
    RequestPlan,
    RuleSourceRef,
)
from magic_assistant.api.app import create_app
from magic_assistant.cards.models import Card, MagicColor
from magic_assistant.rules.retrieval import RuleEvidence


class FakeRuntime:
    def __init__(self, state=None, *, fail: bool = False) -> None:
        self.state = state
        self.fail = fail
        self.calls = []
        self.closed = False

    @property
    def ready(self) -> bool:
        return not self.closed

    def invoke(self, message: str, *, thread_id: str):
        self.calls.append((message, thread_id))
        if self.fail:
            raise RuntimeError("secret internal details")
        return self.state

    def close(self) -> None:
        self.closed = True


def rule_evidence() -> RuleEvidence:
    return RuleEvidence(
        chunk_id="rule:509.1:2",
        rule_ids=["509.1d", "509.1e", "509.1h", "509.1i"],
        root_rule_id="509.1",
        title="Declare Blockers Step",
        text="509.1h An attacking creature becomes unblocked.",
        page_start=111,
        page_end=112,
        rules_version="2026-04-17",
        score=1.0,
        retrieval_methods=["exact"],
        document_type="rule",
    )


def sample_card() -> Card:
    return Card(
        id="card-1",
        name="Ninja of the Deep Hours",
        mana_cost="{3}{U}",
        mana_value=4,
        colors=[MagicColor.BLUE],
        color_identity=[MagicColor.BLUE],
        type_line="Creature — Human Ninja",
        types=["Creature"],
        subtypes=["Human", "Ninja"],
        oracle_text="Ninjutsu {1}{U}",
        set_code="BOK",
        image_url="https://example.test/ninja.jpg",
    )


def grounded_state():
    evidence = rule_evidence()
    card = sample_card()
    return {
        "request_plan": RequestPlan(intent=RequestIntent.CARD_INTERACTION),
        "rules_evidence": [evidence],
        "cards": [card],
        "draft_answer": GroundedAnswerDraft(
            text="The Ninja can deal damage if a combat damage step remains.",
            rule_sources=[RuleSourceRef(chunk_id=evidence.chunk_id, rule_id="509.1h")],
            used_card_ids=[card.id],
        ),
        "source_validation_passed": True,
        "final_answer": "Grounded answer with sources.",
        "route_trace": ["plan_request", "resolve_named_cards", "render_answer"],
        "errors": [],
    }


def custom_state():
    evidence = rule_evidence()
    custom = CustomCard(
        name="Han Solo, Daring Captain",
        mana_cost="{1}{R}{W}",
        colors=[MagicColor.RED, MagicColor.WHITE],
        type_line="Legendary Creature — Human Rogue",
        oracle_text="First strike",
        power="3",
        toughness="2",
    )
    return {
        "request_plan": RequestPlan(intent=RequestIntent.CUSTOM_CARD),
        "rules_evidence": [evidence],
        "cards": [],
        "custom_card": custom,
        "custom_card_draft": CustomCardDraft(
            card=custom,
            used_rule_chunk_ids=[evidence.chunk_id],
        ),
        "source_validation_passed": True,
        "final_answer": "CUSTOM / FAN-MADE — NOT AN OFFICIAL MAGIC CARD",
        "route_trace": ["plan_request", "generate_custom_card", "render_custom_card"],
        "errors": [],
    }


def test_health_readiness_and_lifecycle_close():
    runtime = FakeRuntime(grounded_state())
    app = create_app(lambda: runtime)

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json() == {"status": "ready", "detail": None}

    assert runtime.closed


def test_chat_passes_thread_and_returns_metadata_and_structured_sources():
    runtime = FakeRuntime(grounded_state())
    app = create_app(lambda: runtime)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={"message": "How do these cards interact?", "thread_id": "demo-123"},
            headers={"X-Request-ID": "request-123"},
        )

    assert response.status_code == 200
    body = response.json()
    assert runtime.calls == [("How do these cards interact?", "demo-123")]
    assert body["request_id"] == "request-123"
    assert response.headers["X-Request-ID"] == "request-123"
    assert body["intent"] == "card_interaction"
    assert body["route_trace"] == ["plan_request", "resolve_named_cards", "render_answer"]
    assert body["sources"][0] == {
        "type": "rule",
        "document": "Magic Comprehensive Rules",
        "version": "2026-04-17",
        "rule_id": "509.1h",
        "page_start": 111,
        "page_end": 112,
    }
    assert body["sources"][1] == {
        "type": "card",
        "card_id": "card-1",
        "name": "Ninja of the Deep Hours",
        "provider": "MTG API",
    }
    assert body["cards"][0]["oracle_text"] == "Ninjutsu {1}{U}"


def test_glossary_source_is_structured_without_rule_id():
    evidence = RuleEvidence(
        chunk_id="glossary:0001",
        term="Ninjutsu",
        text="A keyword ability.",
        page_start=286,
        page_end=286,
        rules_version="2026-04-17",
        score=1.0,
        retrieval_methods=["terminology"],
        document_type="glossary",
    )
    state = {
        "request_plan": RequestPlan(intent=RequestIntent.RULES_QUESTION),
        "rules_evidence": [evidence],
        "cards": [],
        "draft_answer": GroundedAnswerDraft(
            text="Ninjutsu is a keyword ability.",
            rule_sources=[RuleSourceRef(chunk_id=evidence.chunk_id)],
        ),
        "source_validation_passed": True,
        "final_answer": "Answer",
        "route_trace": ["plan_request", "render_answer"],
        "errors": [],
    }

    with TestClient(create_app(lambda: FakeRuntime(state))) as client:
        response = client.post(
            "/api/chat",
            json={"message": "What is ninjutsu?", "thread_id": "glossary"},
        )

    assert response.json()["sources"] == [
        {
            "type": "glossary",
            "term": "Ninjutsu",
            "version": "2026-04-17",
            "page_start": 286,
            "page_end": 286,
        }
    ]


def test_custom_card_response_is_structured():
    runtime = FakeRuntime(custom_state())

    with TestClient(create_app(lambda: runtime)) as client:
        response = client.post(
            "/api/chat",
            json={"message": "Create Han Solo", "thread_id": "custom"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "custom_card"
    assert body["custom_card"]["name"] == "Han Solo, Daring Captain"
    assert body["custom_card"]["oracle_text"] == "First strike"


def test_malformed_chat_input_returns_422_without_invoking_runtime():
    runtime = FakeRuntime(grounded_state())

    with TestClient(create_app(lambda: runtime)) as client:
        response = client.post("/api/chat", json={"message": "   ", "thread_id": ""})

    assert response.status_code == 422
    assert runtime.calls == []


def test_runtime_initialization_failure_makes_readiness_and_chat_unavailable():
    def unavailable_runtime():
        raise RuntimeError("do not expose this")

    with TestClient(create_app(unavailable_runtime)) as client:
        readiness = client.get("/ready")
        chat = client.post(
            "/api/chat",
            json={"message": "How does flying work?", "thread_id": "demo"},
        )

    assert readiness.status_code == 503
    assert readiness.json() == {
        "status": "unavailable",
        "detail": "Assistant runtime initialization failed.",
    }
    assert chat.status_code == 503
    assert "do not expose this" not in chat.text


def test_unexpected_chat_failure_is_sanitized():
    runtime = FakeRuntime(fail=True)

    with TestClient(create_app(lambda: runtime)) as client:
        response = client.post(
            "/api/chat",
            json={"message": "How does flying work?", "thread_id": "demo"},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "secret internal details" not in response.text


def test_invalid_request_id_is_replaced():
    runtime = FakeRuntime(grounded_state())

    with TestClient(create_app(lambda: runtime)) as client:
        response = client.post(
            "/api/chat",
            json={"message": "How does flying work?", "thread_id": "demo"},
            headers={"X-Request-ID": "invalid request id with spaces"},
        )

    generated = response.headers["X-Request-ID"]
    assert generated != "invalid request id with spaces"
    assert len(generated) == 36
