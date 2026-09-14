from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from magic_assistant.agent.llm import (
    CustomCardGenerator,
    GroundedAnswerGenerator,
    RequestPlanner,
)
from magic_assistant.agent.rendering import (
    GROUNDING_FALLBACK,
    SCOPE_RESPONSE,
    render_card_search,
    render_custom_card,
    render_grounded_answer,
)
from magic_assistant.agent.schemas import AgentError, RequestIntent, RequestPlan
from magic_assistant.agent.state import AssistantState
from magic_assistant.cards.models import Card, CardSearchFilters
from magic_assistant.cards.service import normalize_card_name
from magic_assistant.rules.retrieval import RuleEvidence

MAX_SYNTHESIS_ATTEMPTS = 2
MAX_DISPLAYED_CARDS = 10


class RulesRetriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[RuleEvidence]: ...


class CardSearcher(Protocol):
    def search(self, filters: CardSearchFilters) -> list[Card]: ...


@dataclass(frozen=True)
class AssistantDependencies:
    planner: RequestPlanner
    answer_generator: GroundedAnswerGenerator
    custom_card_generator: CustomCardGenerator
    rules_retriever: RulesRetriever
    card_searcher: CardSearcher


class AssistantNodes:
    def __init__(self, dependencies: AssistantDependencies) -> None:
        self._dependencies = dependencies

    def plan_request(self, state: AssistantState) -> dict[str, Any]:
        question = self._latest_user_question(state.get("messages", []))
        update: dict[str, Any] = {
            "current_user_query": question,
            "request_plan": None,
            "rules_evidence": [],
            "cards": [],
            "custom_card": None,
            "custom_card_draft": None,
            "draft_answer": None,
            "final_answer": "",
            "generation_attempts": 0,
            "source_validation_passed": False,
            "validation_feedback": None,
            "errors": [],
            "route_trace": ["plan_request"],
        }
        try:
            update["request_plan"] = self._dependencies.planner.plan(state.get("messages", []))
        except Exception:
            update["errors"] = [
                AgentError(
                    category="planning_failure",
                    user_message="I couldn't classify that Magic request. Please rephrase it.",
                )
            ]
        return update

    def retrieve_rules(self, state: AssistantState) -> dict[str, Any]:
        plan = self._require_plan(state)
        query = plan.rules_query or state["current_user_query"]
        try:
            evidence = self._dependencies.rules_retriever.search(query, top_k=5)
        except Exception:
            return self._service_error(
                state,
                "rules_retrieval_failure",
                "The Magic rules index is currently unavailable.",
                "retrieve_rules",
            )
        if not evidence:
            return self._service_error(
                state,
                "insufficient_rules_evidence",
                "I couldn't find enough rules evidence to answer that question.",
                "retrieve_rules",
            )
        return {"rules_evidence": evidence, "route_trace": self._trace(state, "retrieve_rules")}

    def search_cards(self, state: AssistantState) -> dict[str, Any]:
        plan = self._require_plan(state)
        if plan.card_search_filters is None:
            return self._service_error(
                state,
                "missing_card_filters",
                "I need more card-search details to run that search.",
                "search_cards",
            )
        filters = plan.card_search_filters.model_copy(
            update={"result_limit": min(plan.card_search_filters.result_limit, MAX_DISPLAYED_CARDS)}
        )
        try:
            cards = self._dependencies.card_searcher.search(filters)
        except Exception:
            return self._service_error(
                state,
                "card_service_failure",
                "The Magic card service is currently unavailable.",
                "search_cards",
            )
        return {"cards": cards, "route_trace": self._trace(state, "search_cards")}

    def resolve_named_cards(self, state: AssistantState) -> dict[str, Any]:
        plan = self._require_plan(state)
        if not plan.card_names:
            return self._service_error(
                state,
                "missing_card_names",
                "I need the card names involved in that interaction.",
                "resolve_named_cards",
            )
        resolved = []
        try:
            for name in plan.card_names:
                card = resolve_named_card(name, self._dependencies.card_searcher)
                if card is None:
                    return self._service_error(
                        state,
                        "card_not_found",
                        f'I could not resolve the card "{name}".',
                        "resolve_named_cards",
                    )
                if card.id not in {candidate.id for candidate in resolved}:
                    resolved.append(card)
        except Exception:
            return self._service_error(
                state,
                "card_service_failure",
                "The Magic card service is currently unavailable.",
                "resolve_named_cards",
            )
        return {"cards": resolved, "route_trace": self._trace(state, "resolve_named_cards")}

    def retrieve_interaction_rules(self, state: AssistantState) -> dict[str, Any]:
        plan = self._require_plan(state)
        card_context = "\n".join(
            f"{card.name}: {card.oracle_text or 'No Oracle text available.'}"
            for card in state.get("cards", [])
        )
        query_parts = [state["current_user_query"]]
        if plan.rules_query:
            query_parts.append(plan.rules_query)
        query_parts.append(card_context)
        enriched_query = "\n".join(query_parts)
        try:
            evidence = self._dependencies.rules_retriever.search(enriched_query, top_k=8)
        except Exception:
            return self._service_error(
                state,
                "rules_retrieval_failure",
                "The Magic rules index is currently unavailable.",
                "retrieve_interaction_rules",
            )
        if not evidence:
            return self._service_error(
                state,
                "insufficient_rules_evidence",
                "I couldn't find enough rules evidence for that card interaction.",
                "retrieve_interaction_rules",
            )
        return {
            "rules_evidence": evidence,
            "route_trace": self._trace(state, "retrieve_interaction_rules"),
        }

    def retrieve_custom_mechanics(self, state: AssistantState) -> dict[str, Any]:
        plan = self._require_plan(state)
        request = plan.custom_card_request
        if request is None:
            return self._service_error(
                state,
                "missing_custom_card_request",
                "I need more detail about the custom card concept.",
                "retrieve_custom_mechanics",
            )
        evidence: list[RuleEvidence] = []
        try:
            if request.requested_mechanics:
                for mechanic in request.requested_mechanics:
                    retrieved = self._dependencies.rules_retriever.search(mechanic, top_k=5)
                    qualified = [
                        item
                        for item in retrieved
                        if {"exact", "terminology"}.intersection(item.retrieval_methods)
                    ]
                    if not qualified:
                        return self._service_error(
                            state,
                            "insufficient_mechanics_evidence",
                            f'I could not verify the requested mechanic "{mechanic}".',
                            "retrieve_custom_mechanics",
                        )
                    for item in qualified:
                        if item.chunk_id not in {existing.chunk_id for existing in evidence}:
                            evidence.append(item)
            else:
                evidence = self._dependencies.rules_retriever.search(
                    state["current_user_query"], top_k=5
                )
        except Exception:
            return self._service_error(
                state,
                "rules_retrieval_failure",
                "The Magic rules index is currently unavailable.",
                "retrieve_custom_mechanics",
            )
        return {
            "rules_evidence": evidence,
            "route_trace": self._trace(state, "retrieve_custom_mechanics"),
        }

    def synthesize_grounded_answer(self, state: AssistantState) -> dict[str, Any]:
        plan = self._require_plan(state)
        attempts = state.get("generation_attempts", 0) + 1
        try:
            draft = self._dependencies.answer_generator.generate(
                question=state["current_user_query"],
                messages=state.get("messages", []),
                response_language=plan.response_language,
                rules_evidence=state.get("rules_evidence", []),
                cards=state.get("cards", []),
                validation_feedback=state.get("validation_feedback"),
            )
            errors: list[AgentError] = []
        except Exception:
            draft = None
            errors = [
                AgentError(
                    category="answer_generation_failure",
                    user_message="The answer model could not produce a grounded response.",
                )
            ]
        return {
            "draft_answer": draft,
            "generation_attempts": attempts,
            "errors": errors,
            "route_trace": self._trace(state, "synthesize_grounded_answer"),
        }

    def validate_sources(self, state: AssistantState) -> dict[str, Any]:
        draft = state.get("draft_answer")
        plan = self._require_plan(state)
        feedback = []
        if draft is None:
            feedback.append("No structured answer draft was produced.")
        else:
            allowed_rules = {item.chunk_id for item in state.get("rules_evidence", [])}
            allowed_cards = {card.id for card in state.get("cards", [])}
            unknown_rules = set(draft.used_rule_chunk_ids) - allowed_rules
            unknown_cards = set(draft.used_card_ids) - allowed_cards
            if unknown_rules:
                feedback.append(f"Unknown rule chunk IDs: {sorted(unknown_rules)}")
            if unknown_cards:
                feedback.append(f"Unknown card IDs: {sorted(unknown_cards)}")
            if plan.intent == RequestIntent.RULES_QUESTION and not draft.used_rule_chunk_ids:
                feedback.append("A rules answer must use at least one retrieved rule source.")
            if plan.intent == RequestIntent.CARD_INTERACTION:
                if not draft.used_rule_chunk_ids:
                    feedback.append("An interaction answer must use rule evidence.")
                missing_cards = allowed_cards - set(draft.used_card_ids)
                if missing_cards:
                    missing = sorted(missing_cards)
                    feedback.append(
                        f"An interaction answer must use every resolved card: {missing}"
                    )
        return {
            "source_validation_passed": not feedback,
            "validation_feedback": " ".join(feedback) or None,
            "route_trace": self._trace(state, "validate_sources"),
        }

    def generate_custom_card(self, state: AssistantState) -> dict[str, Any]:
        plan = self._require_plan(state)
        request = plan.custom_card_request
        if request is None:
            return self._service_error(
                state,
                "missing_custom_card_request",
                "I need more detail about the custom card concept.",
                "generate_custom_card",
            )
        attempts = state.get("generation_attempts", 0) + 1
        try:
            draft = self._dependencies.custom_card_generator.generate(
                request=request,
                original_question=state["current_user_query"],
                rules_evidence=state.get("rules_evidence", []),
                response_language=plan.response_language,
                validation_feedback=state.get("validation_feedback"),
            )
        except Exception:
            return self._service_error(
                state,
                "custom_card_generation_failure",
                "I couldn't generate that custom card concept.",
                "generate_custom_card",
            )
        return {
            "custom_card": draft.card,
            "custom_card_draft": draft,
            "generation_attempts": attempts,
            "errors": [],
            "route_trace": self._trace(state, "generate_custom_card"),
        }

    def validate_custom_card_sources(self, state: AssistantState) -> dict[str, Any]:
        draft = state.get("custom_card_draft")
        allowed_rules = {item.chunk_id for item in state.get("rules_evidence", [])}
        feedback = []
        if draft is None:
            feedback.append("No structured custom-card draft was produced.")
        else:
            unknown_rules = set(draft.used_rule_chunk_ids) - allowed_rules
            if unknown_rules:
                feedback.append(f"Unknown rule chunk IDs: {sorted(unknown_rules)}")
            plan = self._require_plan(state)
            request = plan.custom_card_request
            if request and request.requested_mechanics and not draft.used_rule_chunk_ids:
                feedback.append("A custom card with requested mechanics must use rules evidence.")
        return {
            "source_validation_passed": not feedback,
            "validation_feedback": " ".join(feedback) or None,
            "route_trace": self._trace(state, "validate_custom_card_sources"),
        }

    def render_grounded_answer(self, state: AssistantState) -> dict[str, Any]:
        final = render_grounded_answer(
            cast(Any, state["draft_answer"]),
            state.get("rules_evidence", []),
            state.get("cards", []),
        )
        return self._final_update(state, final, "render_answer")

    def render_card_search(self, state: AssistantState) -> dict[str, Any]:
        plan = self._require_plan(state)
        final = render_card_search(state.get("cards", []), plan.response_language)
        return self._final_update(state, final, "render_card_search")

    def render_custom_card(self, state: AssistantState) -> dict[str, Any]:
        final = render_custom_card(cast(Any, state["custom_card"]))
        return self._final_update(state, final, "render_custom_card")

    def scope_response(self, state: AssistantState) -> dict[str, Any]:
        return self._final_update(state, SCOPE_RESPONSE, "scope_response")

    def service_failure(self, state: AssistantState) -> dict[str, Any]:
        errors = state.get("errors", [])
        final = errors[0].user_message if errors else "The requested Magic service is unavailable."
        return self._final_update(state, final, "service_failure")

    def grounding_fallback(self, state: AssistantState) -> dict[str, Any]:
        return self._final_update(state, GROUNDING_FALLBACK, "grounding_fallback")

    @staticmethod
    def _latest_user_question(messages: Sequence[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return str(message.content)
        return ""

    @staticmethod
    def _require_plan(state: AssistantState) -> RequestPlan:
        plan = state.get("request_plan")
        if plan is None:
            raise ValueError("Request plan is unavailable")
        return plan

    @staticmethod
    def _trace(state: AssistantState, node: str) -> list[str]:
        return [*state.get("route_trace", []), node]

    @classmethod
    def _service_error(
        cls,
        state: AssistantState,
        category: str,
        message: str,
        node: str,
    ) -> dict[str, Any]:
        return {
            "errors": [AgentError(category=category, user_message=message)],
            "route_trace": cls._trace(state, node),
        }

    @classmethod
    def _final_update(cls, state: AssistantState, answer: str, node: str) -> dict[str, Any]:
        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
            "route_trace": cls._trace(state, node),
        }


def resolve_named_card(name: str, card_searcher: CardSearcher) -> Card | None:
    candidates = card_searcher.search(CardSearchFilters(name=name, result_limit=5, max_pages=2))
    target = normalize_card_name(name)
    exact = [card for card in candidates if normalize_card_name(card.name) == target]
    if exact:
        return exact[0]
    return candidates[0] if candidates else None


def build_assistant_graph(
    dependencies: AssistantDependencies,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Any:
    nodes = AssistantNodes(dependencies)
    builder = StateGraph(AssistantState)
    builder.add_node("plan_request", nodes.plan_request)
    builder.add_node("retrieve_rules", nodes.retrieve_rules)
    builder.add_node("search_cards", nodes.search_cards)
    builder.add_node("resolve_named_cards", nodes.resolve_named_cards)
    builder.add_node("retrieve_interaction_rules", nodes.retrieve_interaction_rules)
    builder.add_node("retrieve_custom_mechanics", nodes.retrieve_custom_mechanics)
    builder.add_node("synthesize_grounded_answer", nodes.synthesize_grounded_answer)
    builder.add_node("validate_sources", nodes.validate_sources)
    builder.add_node("generate_custom_card", nodes.generate_custom_card)
    builder.add_node("validate_custom_card_sources", nodes.validate_custom_card_sources)
    builder.add_node("render_answer", nodes.render_grounded_answer)
    builder.add_node("render_card_search", nodes.render_card_search)
    builder.add_node("render_custom_card", nodes.render_custom_card)
    builder.add_node("scope_response", nodes.scope_response)
    builder.add_node("service_failure", nodes.service_failure)
    builder.add_node("grounding_fallback", nodes.grounding_fallback)

    builder.add_edge(START, "plan_request")
    builder.add_conditional_edges(
        "plan_request",
        _route_plan,
        {
            "rules_question": "retrieve_rules",
            "card_search": "search_cards",
            "card_interaction": "resolve_named_cards",
            "custom_card": "retrieve_custom_mechanics",
            "out_of_scope": "scope_response",
            "failure": "service_failure",
        },
    )
    builder.add_conditional_edges(
        "retrieve_rules",
        _route_service_result,
        {"success": "synthesize_grounded_answer", "failure": "service_failure"},
    )
    builder.add_conditional_edges(
        "search_cards",
        _route_service_result,
        {"success": "render_card_search", "failure": "service_failure"},
    )
    builder.add_conditional_edges(
        "resolve_named_cards",
        _route_service_result,
        {"success": "retrieve_interaction_rules", "failure": "service_failure"},
    )
    builder.add_conditional_edges(
        "retrieve_interaction_rules",
        _route_service_result,
        {"success": "synthesize_grounded_answer", "failure": "service_failure"},
    )
    builder.add_conditional_edges(
        "retrieve_custom_mechanics",
        _route_service_result,
        {"success": "generate_custom_card", "failure": "service_failure"},
    )
    builder.add_conditional_edges(
        "generate_custom_card",
        _route_generation_result,
        {"success": "validate_custom_card_sources", "failure": "service_failure"},
    )
    builder.add_conditional_edges(
        "validate_custom_card_sources",
        _route_validation,
        {
            "valid": "render_custom_card",
            "retry": "generate_custom_card",
            "fallback": "grounding_fallback",
        },
    )
    builder.add_conditional_edges(
        "synthesize_grounded_answer",
        _route_generation_result,
        {"success": "validate_sources", "failure": "service_failure"},
    )
    builder.add_conditional_edges(
        "validate_sources",
        _route_validation,
        {
            "valid": "render_answer",
            "retry": "synthesize_grounded_answer",
            "fallback": "grounding_fallback",
        },
    )
    for terminal_node in (
        "render_answer",
        "render_card_search",
        "render_custom_card",
        "scope_response",
        "service_failure",
        "grounding_fallback",
    ):
        builder.add_edge(terminal_node, END)
    saver = checkpointer if checkpointer is not None else InMemorySaver()
    return builder.compile(checkpointer=saver)


def _route_plan(state: AssistantState) -> str:
    if state.get("errors"):
        return "failure"
    plan = state.get("request_plan")
    return plan.intent.value if plan else "failure"


def _route_service_result(state: AssistantState) -> str:
    return "failure" if state.get("errors") else "success"


def _route_generation_result(state: AssistantState) -> str:
    return "failure" if state.get("errors") else "success"


def _route_validation(state: AssistantState) -> str:
    if state.get("source_validation_passed"):
        return "valid"
    if state.get("generation_attempts", 0) < MAX_SYNTHESIS_ATTEMPTS:
        return "retry"
    return "fallback"


class MagicAssistant:
    def __init__(self, graph: Any) -> None:
        self.graph = graph

    def invoke(self, question: str, *, thread_id: str) -> AssistantState:
        if not question.strip():
            raise ValueError("question must not be empty")
        normalized_thread_id = thread_id.strip()
        if not normalized_thread_id:
            raise ValueError("thread_id must not be empty")
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            {"configurable": {"thread_id": normalized_thread_id}},
        )
        return cast(AssistantState, result)

    def ask(self, question: str, *, thread_id: str) -> str:
        return self.invoke(question, thread_id=thread_id)["final_answer"]
