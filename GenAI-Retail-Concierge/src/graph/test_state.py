from src.graph.state import AgentState


def test_agent_state() -> None:
    """
    Test creation of the shared LangGraph state.
    """

    state: AgentState = {
        "user_query": (
            "Can I return my lightly used item?"
        ),
        "customer_id": "C000001",
        "intent": "",
        "customer_profile": {},
        "customer_interactions": [],
        "vip_signals": {},
        "retrieved_documents": [],
        "retrieval_sources": [],
        "context": "",
        "tool_name": "",
        "tool_input": {},
        "tool_result": {},
        "action_taken": False,
        "confidence": 0.0,
        "sla_risk": False,
        "requires_escalation": False,
        "escalation_reason": "",
        "final_answer": "",
        "trace": [],
    }

    print("\nAGENT STATE TEST\n")

    print(
        f"User query: "
        f"{state['user_query']}"
    )

    print(
        f"Customer ID: "
        f"{state['customer_id']}"
    )

    print(
        f"Initial confidence: "
        f"{state['confidence']}"
    )

    print(
        f"Requires escalation: "
        f"{state['requires_escalation']}"
    )

    print("\nAGENT STATE CREATED SUCCESSFULLY")


if __name__ == "__main__":
    test_agent_state()