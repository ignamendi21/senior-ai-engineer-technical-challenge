from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from magic_assistant.agent.schemas import (
    AgentError,
    CustomCard,
    GroundedAnswerDraft,
    RequestPlan,
)
from magic_assistant.cards.models import Card
from magic_assistant.rules.retrieval import RuleEvidence


class AssistantState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    current_user_query: str
    request_plan: RequestPlan
    rules_evidence: list[RuleEvidence]
    cards: list[Card]
    custom_card: CustomCard
    draft_answer: GroundedAnswerDraft
    final_answer: str
    generation_attempts: int
    source_validation_passed: bool
    validation_feedback: str | None
    errors: list[AgentError]
    route_trace: list[str]
