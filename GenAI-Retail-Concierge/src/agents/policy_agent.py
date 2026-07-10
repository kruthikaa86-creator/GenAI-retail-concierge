from src.generation.rag import (
    build_context,
    get_source_label,
)
from src.graph.state import AgentState
from src.retrieval.retriever import retrieve_documents


def retrieve_policy(
    state: AgentState,
) -> dict:
    """
    Retrieve authoritative operations policies.
    """

    user_query = state["user_query"]

    policy_documents = retrieve_documents(
        query=user_query,
        index_type="policies",
        top_k=3,
    )

    context = build_context(
        policy_documents
    )

    retrieval_sources = [
        get_source_label(document)
        for document in policy_documents
    ]

    trace = list(
        state.get("trace", [])
    )

    trace.append(
        {
            "node": "policy_lookup",
            "policy_documents": len(
                policy_documents
            ),
            "sources": retrieval_sources,
        }
    )

    return {
        "retrieved_documents": policy_documents,
        "retrieval_sources": retrieval_sources,
        "context": context,
        "trace": trace,
    }


def test_policy_agent() -> None:
    """
    Run policy lookup agent smoke test.
    """

    state: AgentState = {
        "user_query": (
            "Can I return a lightly used item?"
        ),
        "trace": [],
    }

    result = retrieve_policy(state)

    print("\nPOLICY AGENT TEST\n")

    print(
        "Retrieved Documents: "
        f"{len(result['retrieved_documents'])}"
    )

    print("Sources:")

    for source in result["retrieval_sources"]:
        print(source)

    print("\nTrace:")

    print(result["trace"])


if __name__ == "__main__":
    test_policy_agent()