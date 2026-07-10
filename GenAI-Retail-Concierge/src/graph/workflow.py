from langgraph.graph import END, START, StateGraph

from src.agents.catalog_agent import retrieve_catalog
from src.agents.escalation_agent import evaluate_escalation
from src.agents.personalization_agent import personalize_customer
from src.agents.policy_agent import retrieve_policy
from src.agents.router import route_intent
from src.generation.llm import get_llm
from src.generation.prompts import get_rag_prompt
from src.graph.state import AgentState
from src.tools.return_tools import process_return_inquiry
from src.observability.tracing import log_workflow_trace

def personalization_node(
    state: AgentState,
) -> dict:
    """
    Inject customer profile, interactions,
    and VIP signals into shared state.
    """

    return personalize_customer(state)


def router_node(
    state: AgentState,
) -> dict:
    """
    Detect user intent and store it in state.
    """

    return route_intent(state)

def route_by_intent(
    state: AgentState,
) -> str:
    """
    Route workflow using detected intent.
    """

    intent = state.get(
        "intent",
        "general",
    )

    if intent == "catalog":
        return "catalog"

    if intent == "policy":
        return "policy"

    if intent == "return":
        return "return"

    if intent == "order":
        return "order"

    return "general"


def catalog_node(
    state: AgentState,
) -> dict:
    """
    Run catalog retrieval.
    """

    result = retrieve_catalog(state)

    result["confidence"] = 0.80

    return result


def policy_node(
    state: AgentState,
) -> dict:
    """
    Run authoritative policy retrieval.
    """

    result = retrieve_policy(state)

    result["confidence"] = 0.85

    return result


def return_node(
    state: AgentState,
) -> dict:
    """
    Run return inquiry tool.
    """

    result = process_return_inquiry(state)

    tool_result = result.get(
        "tool_result",
        {},
    )

    if tool_result.get("success"):
        result["confidence"] = 0.95
    else:
        result["confidence"] = 0.50

    return result


def order_node(
    state: AgentState,
) -> dict:
    """
    Handle order inquiry safely.

    orders.csv is not available in the supplied
    project data, so order status cannot be verified.
    """

    trace = list(
        state.get("trace", [])
    )

    trace.append(
        {
            "node": "order_inquiry",
            "action": "order_data_unavailable",
        }
    )

    return {
        "tool_name": "order_inquiry",
        "tool_input": {
            "customer_id": state.get(
                "customer_id",
                "",
            ),
        },
        "tool_result": {
            "success": False,
            "message": (
                "Order status cannot be verified "
                "because order records are not "
                "available in the supplied local data."
            ),
        },
        "action_taken": False,
        "confidence": 0.0,
        "requires_escalation": True,
        "escalation_reason": (
            "Order records are unavailable"
        ),
        "trace": trace,
    }


def general_node(
    state: AgentState,
) -> dict:
    """
    Handle general conversation.
    """

    trace = list(
        state.get("trace", [])
    )

    trace.append(
        {
            "node": "general_handler",
        }
    )

    return {
        "context": "",
        "retrieval_sources": [],
        "confidence": 0.70,
        "trace": trace,
    }


def escalation_node(
    state: AgentState,
) -> dict:
    """
    Evaluate confidence, VIP status,
    existing tool escalation, and SLA risk.
    """

    return evaluate_escalation(state)


def route_after_escalation(
    state: AgentState,
) -> str:
    """
    Route to human handoff or automated generation.
    """

    if state.get(
        "requires_escalation",
        False,
    ):
        return "human"

    return "generate"


def human_handoff_node(
    state: AgentState,
) -> dict:
    """
    Create white-glove human handoff messaging.
    """

    vip_signals = state.get(
        "vip_signals",
        {},
    )

    is_vip = vip_signals.get(
        "is_vip",
        False,
    )

    escalation_reason = state.get(
        "escalation_reason",
        "",
    )

    if is_vip:
        final_answer = (
            "I’m arranging a priority white-glove "
            "handoff to a human concierge for this "
            "request. Your conversation context and "
            "the actions already checked will be "
            "included in the handoff."
        )
    else:
        final_answer = (
            "This request requires human-agent "
            "assistance. I’m escalating it with the "
            "conversation context and the actions "
            "already checked."
        )

    trace = list(
        state.get("trace", [])
    )

    trace.append(
        {
            "node": "human_handoff",
            "is_vip": is_vip,
            "reason": escalation_reason,
        }
    )

    return {
        "final_answer": final_answer,
        "trace": trace,
    }

def observability_node(
    state: AgentState,
) -> dict:
    """
    Save the completed workflow state to the audit trail.
    """

    log_workflow_trace(state)

    trace = list(
        state.get(
            "trace",
            [],
        )
    )

    trace.append(
        {
            "node": "observability",
            "action": "workflow_trace_logged",
        }
    )

    return {
        "trace": trace,
    }
def generate_answer_node(
    state: AgentState,
) -> dict:
    """
    Generate the final grounded response.
    """

    intent = state.get(
        "intent",
        "general",
    )

    trace = list(
        state.get("trace", [])
    )

    if intent == "return":

        tool_result = state.get(
            "tool_result",
            {},
        )

        final_answer = tool_result.get(
            "message",
            (
                "The return inquiry could not "
                "be completed."
            ),
        )

    elif intent == "general":

        llm = get_llm()

        response = llm.invoke(
            (
                "You are a retail concierge assistant. "
                "Answer this general conversational "
                "message briefly and clearly. "
                "Do not invent product, policy, order, "
                "inventory, or customer facts.\n\n"
                f"User message: {state['user_query']}"
            )
        )

        final_answer = response.content

    else:

        context = state.get(
            "context",
            "",
        )

        prompt = get_rag_prompt()

        llm = get_llm()

        chain = prompt | llm

        response = chain.invoke(
            {
                "context": context,
                "question": state["user_query"],
            }
        )

        final_answer = response.content

    trace.append(
        {
            "node": "answer_generation",
            "intent": intent,
        }
    )

    return {
        "final_answer": final_answer,
        "trace": trace,
    }


def build_workflow():
    """
    Build and compile the concierge LangGraph.
    """

    workflow = StateGraph(
        AgentState
    )

    workflow.add_node(
        "personalization",
        personalization_node,
    )

    workflow.add_node(
        "router",
        router_node,
    )

    workflow.add_node(
        "catalog",
        catalog_node,
    )

    workflow.add_node(
        "policy",
        policy_node,
    )

    workflow.add_node(
        "return",
        return_node,
    )

    workflow.add_node(
        "order",
        order_node,
    )

    workflow.add_node(
        "general",
        general_node,
    )

    workflow.add_node(
        "escalation",
        escalation_node,
    )

    workflow.add_node(
        "human_handoff",
        human_handoff_node,
    )

    workflow.add_node(
        "generate",
        generate_answer_node,
    )

    workflow.add_node(
        "observability",
        observability_node,
    )

    workflow.add_edge(
        START,
        "personalization",
    )

    workflow.add_edge(
        "personalization",
        "router",
    )

    workflow.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "catalog": "catalog",
            "policy": "policy",
            "return": "return",
            "order": "order",
            "general": "general",
        },
    )

    workflow.add_edge(
        "catalog",
        "escalation",
    )

    workflow.add_edge(
        "policy",
        "escalation",
    )

    workflow.add_edge(
        "return",
        "escalation",
    )

    workflow.add_edge(
        "order",
        "escalation",
    )

    workflow.add_edge(
        "general",
        "escalation",
    )

    workflow.add_conditional_edges(
        "escalation",
        route_after_escalation,
        {
            "human": "human_handoff",
            "generate": "generate",
        },
    )

    workflow.add_edge(
        "human_handoff",
        "observability",
    )

    workflow.add_edge(
        "generate",
        "observability",
    )

    workflow.add_edge(
        "observability",
        END,
    )

    return workflow.compile()


def test_workflow() -> None:
    """
    Run initial end-to-end concierge journeys.
    """

    app = build_workflow()

    test_cases = [
        {
            "customer_id": "C000002",
            "user_query": (
                "Recommend a product for travel."
            ),
        },
        {
            "customer_id": "C000003",
            "user_query": (
                "What is your return policy "
                "for a lightly used item?"
            ),
        },
        {
            "customer_id": "C000001",
            "user_query": (
                "Can I return my item?"
            ),
        },
        {
            "customer_id": "C000002",
            "user_query": (
                "Where is my order?"
            ),
        },
        {
            "customer_id": "C000003",
            "user_query": (
                "Hello, how are you?"
            ),
        },
    ]

    print(
        "\nLANGGRAPH WORKFLOW TEST\n"
    )

    for test_case in test_cases:

        initial_state: AgentState = {
            "user_query": test_case[
                "user_query"
            ],
            "customer_id": test_case[
                "customer_id"
            ],
            "confidence": 0.0,
            "sla_risk": False,
            "requires_escalation": False,
            "escalation_reason": "",
            "trace": [],
        }

        result = app.invoke(
            initial_state
        )

        print("=" * 80)

        print(
            "Customer ID: "
            f"{result['customer_id']}"
        )

        print(
            "Question: "
            f"{result['user_query']}"
        )

        print(
            "Intent: "
            f"{result['intent']}"
        )

        print(
            "Confidence: "
            f"{result['confidence']}"
        )

        print(
            "SLA Risk: "
            f"{result['sla_risk']}"
        )

        print(
            "Requires Escalation: "
            f"{result['requires_escalation']}"
        )

        print("\nFINAL ANSWER:")

        print(
            result["final_answer"]
        )

        print("\nTRACE:")

        for trace_event in result["trace"]:
            print(trace_event)

        print()


if __name__ == "__main__":
    test_workflow()