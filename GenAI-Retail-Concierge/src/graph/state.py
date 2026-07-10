from typing import Any, TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.
    """

    # Conversation input
    user_query: str
    customer_id: str

    # Routing
    intent: str

    # Customer personalization
    customer_profile: dict[str, Any]
    customer_interactions: list[dict[str, Any]]
    vip_signals: dict[str, Any]

    # Retrieval
    retrieved_documents: list[Document]
    retrieval_sources: list[str]
    context: str

    # Tool execution
    tool_name: str
    tool_input: dict[str, Any]
    tool_result: dict[str, Any]
    action_taken: bool

    # Risk and confidence
    confidence: float
    sla_risk: bool
    requires_escalation: bool
    escalation_reason: str

    # Generation
    final_answer: str

    # Observability
    trace: list[dict[str, Any]]