from typing import Literal

from pydantic import BaseModel, Field

from src.generation.llm import get_llm
from src.graph.state import AgentState


IntentType = Literal[
    "catalog",
    "policy",
    "return",
    "order",
    "general",
]


class IntentClassification(BaseModel):
    """
    Structured output returned by the intent router.
    """

    intent: IntentType = Field(
        description=(
            "The single best intent for the user query."
        )
    )


ROUTER_SYSTEM_PROMPT = """
You are the intent router for a retail concierge system.

Classify the user's query into exactly one intent.

Available intents:

catalog:
Product discovery, product comparison, product details,
recommendations, product reviews, stock, or catalog questions.

policy:
Questions asking about company rules, eligibility rules,
return policies, escalation policies, or operational policies.

return:
A customer wants to inspect, start, validate, or act on
a specific return request.

order:
A customer asks about a specific order, order status,
or order inquiry.

general:
Greetings or general conversation that does not require
catalog, policy, return, or order processing.

Important distinction:

"What is your return policy?" -> policy

"Can I return my item?" -> return

"Where is my order?" -> order

"Recommend a product for me." -> catalog

Return only the structured classification.
"""


def route_intent(
    state: AgentState,
) -> dict:
    """
    Classify the user query and update agent state.
    """

    user_query = state["user_query"]

    llm = get_llm()

    structured_llm = llm.with_structured_output(
        IntentClassification
    )

    response = structured_llm.invoke(
        [
            (
                "system",
                ROUTER_SYSTEM_PROMPT,
            ),
            (
                "human",
                user_query,
            ),
        ]
    )

    intent = response.intent

    trace = list(
        state.get("trace", [])
    )

    trace.append(
        {
            "node": "intent_router",
            "intent": intent,
        }
    )

    return {
        "intent": intent,
        "trace": trace,
    }


def test_router() -> None:
    """
    Run intent-router smoke tests.
    """

    test_queries = [
        "Recommend a product for travel.",
        "What is your return policy?",
        "Can I return my lightly used item?",
        "Where is my order?",
        "Hello, how are you?",
    ]

    print("\nINTENT ROUTER TEST\n")

    for query in test_queries:

        state: AgentState = {
            "user_query": query,
            "trace": [],
        }

        result = route_intent(state)

        print(
            f"Query: {query}"
        )

        print(
            f"Intent: {result['intent']}"
        )

        print("-" * 60)


if __name__ == "__main__":
    test_router()