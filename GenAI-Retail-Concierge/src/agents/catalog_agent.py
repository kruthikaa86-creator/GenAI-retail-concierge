from src.generation.rag import (
    build_context,
    get_source_label,
)
from src.graph.state import AgentState
from src.retrieval.retriever import retrieve_documents


def retrieve_catalog(
    state: AgentState,
) -> dict:
    """
    Retrieve relevant product and review documents.
    """

    user_query = state["user_query"]

    product_documents = retrieve_documents(
        query=user_query,
        index_type="products",
        top_k=3,
    )

    review_documents = retrieve_documents(
        query=user_query,
        index_type="reviews",
        top_k=3,
    )

    retrieved_documents = (
        product_documents
        + review_documents
    )

    context = build_context(
        retrieved_documents
    )

    retrieval_sources = [
        get_source_label(document)
        for document in retrieved_documents
    ]

    trace = list(
        state.get("trace", [])
    )

    trace.append(
        {
            "node": "catalog_retrieval",
            "product_documents": len(
                product_documents
            ),
            "review_documents": len(
                review_documents
            ),
            "sources": retrieval_sources,
        }
    )

    return {
        "retrieved_documents": retrieved_documents,
        "retrieval_sources": retrieval_sources,
        "context": context,
        "trace": trace,
    }


def test_catalog_agent() -> None:
    """
    Run catalog retrieval agent smoke test.
    """

    state: AgentState = {
        "user_query": (
            "Recommend a product for travel."
        ),
        "trace": [],
    }

    result = retrieve_catalog(state)

    print("\nCATALOG AGENT TEST\n")

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
    test_catalog_agent()