from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage

from magic_assistant.agent.graph import (
    AssistantDependencies,
    MagicAssistant,
    build_assistant_graph,
    resolve_named_card,
)
from magic_assistant.agent.rendering import CUSTOM_CARD_NOTICE, GROUNDING_FALLBACK, SCOPE_RESPONSE
from magic_assistant.agent.schemas import (
    CustomCard,
    CustomCardDraft,
    CustomCardRequest,
    GroundedAnswerDraft,
    RequestIntent,
    RequestPlan,
)
from magic_assistant.cards.client import CardApiConnectionError
from magic_assistant.cards.models import Card, CardSearchFilters, MagicColor
from magic_assistant.rules.retrieval import RuleEvidence


class FakePlanner:
    def __init__(self, plans: dict[str, RequestPlan]) -> None:
        self.plans = plans
        self.histories: list[list[BaseMessage]] = []

    def plan(self, messages: Sequence[BaseMessage]) -> RequestPlan:
        self.histories.append(list(messages))
        question = next(
            str(message.content)
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        )
        return self.plans[question]


class FakeRulesRetriever:
    def __init__(
        self,
        evidence: list[RuleEvidence],
        *,
        results_by_query: dict[str, list[RuleEvidence]] | None = None,
        fail: bool = False,
    ) -> None:
        self.evidence = evidence
        self.results_by_query = results_by_query or {}
        self.fail = fail
        self.queries: list[str] = []

    def search(self, query: str, top_k: int = 5) -> list[RuleEvidence]:
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("index unavailable")
        return self.results_by_query.get(query, self.evidence)[:top_k]


class FakeCardSearcher:
    def __init__(
        self,
        *,
        search_results: list[Card] | None = None,
        named_results: dict[str, list[Card]] | None = None,
        fail: bool = False,
    ) -> None:
        self.search_results = search_results or []
        self.named_results = named_results or {}
        self.fail = fail
        self.filters: list[CardSearchFilters] = []

    def search(self, filters: CardSearchFilters) -> list[Card]:
        self.filters.append(filters)
        if self.fail:
            raise CardApiConnectionError("network unavailable")
        if filters.name:
            return self.named_results.get(filters.name, [])
        return self.search_results


class FakeAnswerGenerator:
    def __init__(
        self,
        drafts: list[GroundedAnswerDraft] | None = None,
        *,
        fail: bool = False,
    ) -> None:
        self.drafts = list(drafts or [])
        self.fail = fail
        self.calls = []

    def generate(self, **kwargs) -> GroundedAnswerDraft:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("model unavailable")
        if self.drafts:
            return self.drafts.pop(0)
        rules = kwargs["rules_evidence"]
        cards = kwargs["cards"]
        return GroundedAnswerDraft(
            text="Grounded answer.",
            used_rule_chunk_ids=[rules[0].chunk_id] if rules else [],
            used_card_ids=[card.id for card in cards],
        )


class FakeCustomCardGenerator:
    def __init__(self, drafts: list[CustomCardDraft] | None = None) -> None:
        self.drafts = list(drafts or [])
        self.calls = []

    def generate(self, **kwargs) -> CustomCardDraft:
        self.calls.append(kwargs)
        if self.drafts:
            return self.drafts.pop(0)
        rules = kwargs["rules_evidence"]
        card = CustomCard(
            name="Han Solo, Daring Captain",
            mana_cost="{1}{R}{W}",
            colors=[MagicColor.RED, MagicColor.WHITE],
            type_line="Legendary Creature — Human Rogue",
            oracle_text="First strike",
            power="3",
            toughness="2",
            flavor_text="Never tell me the odds.",
        )
        return CustomCardDraft(
            card=card,
            used_rule_chunk_ids=[rules[0].chunk_id] if rules else [],
        )


def make_rule_evidence(
    chunk_id: str = "rule:500.1:1",
    root_rule_id: str = "500.1",
) -> RuleEvidence:
    return RuleEvidence(
        chunk_id=chunk_id,
        rule_ids=[root_rule_id],
        root_rule_id=root_rule_id,
        title="General",
        text="A turn consists of five phases.",
        page_start=80,
        page_end=80,
        rules_version="2026-04-17",
        score=1.0,
        retrieval_methods=["exact"],
        document_type="rule",
    )


def make_card(identifier: str, name: str, oracle_text: str) -> Card:
    return Card(
        id=identifier,
        name=name,
        mana_cost="{W}",
        mana_value=1,
        colors=[MagicColor.WHITE],
        color_identity=[MagicColor.WHITE],
        type_line="Creature — Bird",
        types=["Creature"],
        oracle_text=oracle_text,
        set_code="TST",
        image_url=f"https://example.test/{identifier}.jpg",
    )


def rules_plan(query: str) -> RequestPlan:
    return RequestPlan(intent=RequestIntent.RULES_QUESTION, rules_query=query)


def build_test_assistant(
    plans: dict[str, RequestPlan],
    *,
    rules: FakeRulesRetriever | None = None,
    cards: FakeCardSearcher | None = None,
    answers: FakeAnswerGenerator | None = None,
    custom: FakeCustomCardGenerator | None = None,
):
    planner = FakePlanner(plans)
    rules = rules or FakeRulesRetriever([make_rule_evidence()])
    cards = cards or FakeCardSearcher()
    answers = answers or FakeAnswerGenerator()
    custom = custom or FakeCustomCardGenerator()
    graph = build_assistant_graph(
        AssistantDependencies(
            planner=planner,
            answer_generator=answers,
            custom_card_generator=custom,
            rules_retriever=rules,
            card_searcher=cards,
        )
    )
    return MagicAssistant(graph), planner, rules, cards, answers, custom


def test_rules_route_retrieves_and_renders_grounded_source():
    question = "¿Cuántas fases tiene un turno?"
    assistant, _, rules, _, answers, _ = build_test_assistant(
        {
            question: RequestPlan(
                intent=RequestIntent.RULES_QUESTION,
                rules_query="phases of a turn",
                response_language="es",
            )
        }
    )

    state = assistant.invoke(question, thread_id="rules")

    assert rules.queries == ["phases of a turn"]
    assert len(answers.calls) == 1
    assert state["route_trace"] == [
        "plan_request",
        "retrieve_rules",
        "synthesize_grounded_answer",
        "validate_sources",
        "render_answer",
    ]
    assert "rule 500.1" in state["final_answer"]
    assert "PDF p. 80" in state["final_answer"]


def test_card_search_uses_typed_filters_and_deterministic_rendering():
    question = "Busca una criatura blanca guerrero de coste menor que 2"
    card = make_card("warrior-1", "Test Warrior", "Vigilance")
    card_searcher = FakeCardSearcher(search_results=[card])
    assistant, _, _, cards, answers, custom = build_test_assistant(
        {
            question: RequestPlan(
                intent=RequestIntent.CARD_SEARCH,
                card_search_filters=CardSearchFilters(
                    colors=[MagicColor.WHITE],
                    types=["Creature"],
                    subtypes=["Warrior"],
                    max_mana_value_exclusive=2,
                ),
                response_language="es",
            )
        },
        cards=card_searcher,
    )

    state = assistant.invoke(question, thread_id="cards")

    assert cards.filters[0].colors == [MagicColor.WHITE]
    assert cards.filters[0].max_mana_value_exclusive == 2
    assert "Test Warrior" in state["final_answer"]
    assert "Vigilance" in state["final_answer"]
    assert answers.calls == []
    assert custom.calls == []


def test_interaction_enriches_rules_query_and_renders_rule_and_card_sources():
    question = "My Raptor dealt first-strike damage. Can my Ninja deal damage after ninjutsu?"
    raptor = make_card("raptor", "Battlefield Raptor", "Flying, first strike")
    ninja = make_card("ninja", "Ninja of the Deep Hours", "Ninjutsu {1}{U}")
    card_searcher = FakeCardSearcher(
        named_results={"Battlefield Raptor": [raptor], "Ninja of the Deep Hours": [ninja]}
    )
    evidence = make_rule_evidence("rule:702.49:1", "702.49")
    rules = FakeRulesRetriever([evidence])
    assistant, _, _, _, answers, _ = build_test_assistant(
        {
            question: RequestPlan(
                intent=RequestIntent.CARD_INTERACTION,
                rules_query="first strike ninjutsu combat damage",
                card_names=["Battlefield Raptor", "Ninja of the Deep Hours"],
            )
        },
        rules=rules,
        cards=card_searcher,
    )

    state = assistant.invoke(question, thread_id="interaction")

    assert "Flying, first strike" in rules.queries[0]
    assert "Ninjutsu {1}{U}" in rules.queries[0]
    assert answers.calls[0]["cards"] == [raptor, ninja]
    assert answers.calls[0]["rules_evidence"] == [evidence]
    assert "rule 702.49" in state["final_answer"]
    assert "Battlefield Raptor — MTG card API" in state["final_answer"]
    assert "Ninja of the Deep Hours — MTG card API" in state["final_answer"]


def test_interaction_requires_sources_for_every_resolved_card():
    question = "How do these two cards interact?"
    first = make_card("first", "First Card", "First strike")
    second = make_card("second", "Second Card", "Ninjutsu {1}{U}")
    evidence = make_rule_evidence("rule:702.49:1", "702.49")
    answers = FakeAnswerGenerator(
        [
            GroundedAnswerDraft(
                text="Incomplete",
                used_rule_chunk_ids=[evidence.chunk_id],
                used_card_ids=[first.id],
            )
        ]
    )
    assistant, _, _, _, _, _ = build_test_assistant(
        {
            question: RequestPlan(
                intent=RequestIntent.CARD_INTERACTION,
                card_names=[first.name, second.name],
            )
        },
        rules=FakeRulesRetriever([evidence]),
        cards=FakeCardSearcher(named_results={first.name: [first], second.name: [second]}),
        answers=answers,
    )

    state = assistant.invoke(question, thread_id="all-card-sources")

    assert state["generation_attempts"] == 2
    assert "every resolved card" in answers.calls[1]["validation_feedback"]
    assert first.name in state["final_answer"]
    assert second.name in state["final_answer"]


def test_custom_card_route_retrieves_mechanics_and_marks_result():
    question = "Create a Han Solo card, white-red, with first strike."
    custom_request = CustomCardRequest(
        requested_name="Han Solo",
        colors=[MagicColor.WHITE, MagicColor.RED],
        requested_mechanics=["first strike"],
        theme="daring space captain",
    )
    assistant, _, rules, _, answers, custom = build_test_assistant(
        {
            question: RequestPlan(
                intent=RequestIntent.CUSTOM_CARD,
                custom_card_request=custom_request,
            )
        }
    )

    state = assistant.invoke(question, thread_id="custom")

    assert rules.queries == ["first strike"]
    assert custom.calls[0]["request"] == custom_request
    assert custom.calls[0]["original_question"] == question
    assert CUSTOM_CARD_NOTICE in state["final_answer"]
    assert "Han Solo, Daring Captain" in state["final_answer"]
    assert answers.calls == []


def test_requested_custom_mechanics_require_rules_evidence():
    question = "Create a card with first strike"
    custom = FakeCustomCardGenerator()
    assistant, _, _, _, _, _ = build_test_assistant(
        {
            question: RequestPlan(
                intent=RequestIntent.CUSTOM_CARD,
                custom_card_request=CustomCardRequest(requested_mechanics=["first strike"]),
            )
        },
        rules=FakeRulesRetriever([]),
        custom=custom,
    )

    state = assistant.invoke(question, thread_id="custom-no-evidence")

    assert state["final_answer"] == 'I could not verify the requested mechanic "first strike".'
    assert custom.calls == []


def test_each_requested_custom_mechanic_requires_qualified_evidence():
    question = "Create a card with first strike and starjump"
    exact = make_rule_evidence("rule:702.7:1", "702.7")
    semantic_only = make_rule_evidence("rule:999:1", "999")
    semantic_only.retrieval_methods = ["semantic"]
    rules = FakeRulesRetriever(
        [],
        results_by_query={"first strike": [exact], "starjump": [semantic_only]},
    )
    custom = FakeCustomCardGenerator()
    assistant, _, _, _, _, _ = build_test_assistant(
        {
            question: RequestPlan(
                intent=RequestIntent.CUSTOM_CARD,
                custom_card_request=CustomCardRequest(
                    requested_mechanics=["first strike", "starjump"]
                ),
            )
        },
        rules=rules,
        custom=custom,
    )

    state = assistant.invoke(question, thread_id="partial-mechanics")

    assert rules.queries == ["first strike", "starjump"]
    assert state["final_answer"] == 'I could not verify the requested mechanic "starjump".'
    assert custom.calls == []


def test_custom_card_source_validation_retries_once():
    question = "Create a card with first strike"
    evidence = make_rule_evidence("rule:702.7:1", "702.7")
    card = CustomCard(
        name="Test Knight",
        colors=[MagicColor.WHITE],
        type_line="Creature — Human Knight",
        oracle_text="First strike",
    )
    custom = FakeCustomCardGenerator(
        [
            CustomCardDraft(card=card, used_rule_chunk_ids=["unknown"]),
            CustomCardDraft(card=card, used_rule_chunk_ids=[evidence.chunk_id]),
        ]
    )
    assistant, _, _, _, _, _ = build_test_assistant(
        {
            question: RequestPlan(
                intent=RequestIntent.CUSTOM_CARD,
                custom_card_request=CustomCardRequest(requested_mechanics=["first strike"]),
            )
        },
        rules=FakeRulesRetriever([evidence]),
        custom=custom,
    )

    state = assistant.invoke(question, thread_id="custom-source-retry")

    assert state["generation_attempts"] == 2
    assert "Unknown rule chunk IDs" in custom.calls[1]["validation_feedback"]
    assert CUSTOM_CARD_NOTICE in state["final_answer"]


def test_repeated_invalid_custom_sources_return_controlled_fallback():
    question = "Create a card with first strike"
    evidence = make_rule_evidence("rule:702.7:1", "702.7")
    card = CustomCard(
        name="Test Knight",
        type_line="Creature — Human Knight",
        oracle_text="First strike",
    )
    invalid = CustomCardDraft(card=card, used_rule_chunk_ids=["unknown"])
    custom = FakeCustomCardGenerator([invalid, invalid])
    assistant, _, _, _, _, _ = build_test_assistant(
        {
            question: RequestPlan(
                intent=RequestIntent.CUSTOM_CARD,
                custom_card_request=CustomCardRequest(requested_mechanics=["first strike"]),
            )
        },
        rules=FakeRulesRetriever([evidence]),
        custom=custom,
    )

    state = assistant.invoke(question, thread_id="custom-source-fallback")

    assert state["generation_attempts"] == 2
    assert len(custom.calls) == 2
    assert state["final_answer"] == GROUNDING_FALLBACK
    assert state["route_trace"].count("generate_custom_card") == 2


def test_out_of_scope_route_calls_no_services():
    question = "What is the weather?"
    assistant, _, rules, cards, answers, custom = build_test_assistant(
        {question: RequestPlan(intent=RequestIntent.OUT_OF_SCOPE)}
    )

    state = assistant.invoke(question, thread_id="scope")

    assert state["final_answer"] == SCOPE_RESPONSE
    assert rules.queries == []
    assert cards.filters == []
    assert answers.calls == []
    assert custom.calls == []


def test_grounding_retries_once_then_renders_valid_draft():
    question = "How many phases are in a turn?"
    evidence = make_rule_evidence()
    answers = FakeAnswerGenerator(
        [
            GroundedAnswerDraft(text="Invalid", used_rule_chunk_ids=["unknown"]),
            GroundedAnswerDraft(text="Valid", used_rule_chunk_ids=[evidence.chunk_id]),
        ]
    )
    assistant, _, _, _, _, _ = build_test_assistant(
        {
            question: RequestPlan(
                intent=RequestIntent.RULES_QUESTION,
                rules_query=question,
            )
        },
        rules=FakeRulesRetriever([evidence]),
        answers=answers,
    )

    state = assistant.invoke(question, thread_id="retry")

    assert state["generation_attempts"] == 2
    assert len(answers.calls) == 2
    assert "Unknown rule chunk IDs" in answers.calls[1]["validation_feedback"]
    assert state["final_answer"].startswith("Valid")


def test_repeated_invalid_grounding_returns_controlled_fallback():
    question = "How many phases are in a turn?"
    invalid = GroundedAnswerDraft(text="Invalid", used_rule_chunk_ids=["unknown"])
    answers = FakeAnswerGenerator([invalid, invalid])
    assistant, _, _, _, _, _ = build_test_assistant(
        {
            question: RequestPlan(
                intent=RequestIntent.RULES_QUESTION,
                rules_query=question,
            )
        },
        answers=answers,
    )

    state = assistant.invoke(question, thread_id="fallback")

    assert state["generation_attempts"] == 2
    assert state["final_answer"] == GROUNDING_FALLBACK
    assert state["route_trace"].count("synthesize_grounded_answer") == 2


def test_answer_generator_failure_uses_service_error_not_grounding_fallback():
    question = "How many phases are in a turn?"
    assistant, _, _, _, _, _ = build_test_assistant(
        {question: rules_plan(question)},
        answers=FakeAnswerGenerator(fail=True),
    )

    state = assistant.invoke(question, thread_id="generation-failure")

    assert state["generation_attempts"] == 1
    assert state["final_answer"] == "The answer model could not produce a grounded response."
    assert state["final_answer"] != GROUNDING_FALLBACK


def test_same_thread_preserves_conversation_messages():
    first = "How does first strike work?"
    second = "What if I use ninjutsu after that?"
    assistant, planner, _, _, answers, _ = build_test_assistant(
        {first: rules_plan(first), second: rules_plan(second)}
    )

    assistant.ask(first, thread_id="conversation-a")
    assistant.ask(second, thread_id="conversation-a")

    second_history = [str(message.content) for message in planner.histories[1]]
    assert first in second_history
    assert "Grounded answer." in second_history[1]
    assert second in second_history
    generator_history = [str(message.content) for message in answers.calls[1]["messages"]]
    assert first in generator_history
    assert second in generator_history


def test_thread_ids_isolate_conversation_history():
    first = "How does first strike work?"
    isolated = "How does trample work?"
    assistant, planner, _, _, _, _ = build_test_assistant(
        {first: rules_plan(first), isolated: rules_plan(isolated)}
    )

    assistant.ask(first, thread_id="thread-a")
    assistant.ask(isolated, thread_id="thread-b")

    isolated_history = [str(message.content) for message in planner.histories[1]]
    assert isolated_history == [isolated]


def test_turn_local_state_is_reset_when_same_thread_changes_intent():
    first = "How does first strike work?"
    second = "What is the weather?"
    assistant, _, _, _, _, _ = build_test_assistant(
        {
            first: rules_plan(first),
            second: RequestPlan(intent=RequestIntent.OUT_OF_SCOPE),
        }
    )

    assistant.ask(first, thread_id="changing-intent")
    state = assistant.invoke(second, thread_id="changing-intent")

    assert state["request_plan"].intent == RequestIntent.OUT_OF_SCOPE
    assert state["rules_evidence"] == []
    assert state["cards"] == []
    assert state["generation_attempts"] == 0
    assert state["final_answer"] == SCOPE_RESPONSE


def test_card_service_failure_returns_controlled_response():
    question = "Find a white Warrior"
    assistant, _, _, _, answers, _ = build_test_assistant(
        {
            question: RequestPlan(
                intent=RequestIntent.CARD_SEARCH,
                card_search_filters=CardSearchFilters(colors=[MagicColor.WHITE]),
            )
        },
        cards=FakeCardSearcher(fail=True),
    )

    state = assistant.invoke(question, thread_id="failure")

    assert state["final_answer"] == "The Magic card service is currently unavailable."
    assert "Traceback" not in state["final_answer"]
    assert answers.calls == []


def test_card_name_resolution_prefers_exact_normalized_name():
    partial = make_card("partial", "Ninja of the Deep Hours Extended", "Ninjutsu")
    exact = make_card("exact", "  ninja OF THE deep HOURS ", "Ninjutsu {1}{U}")
    searcher = FakeCardSearcher(named_results={"Ninja of the Deep Hours": [partial, exact]})

    assert resolve_named_card("Ninja of the Deep Hours", searcher) == exact
