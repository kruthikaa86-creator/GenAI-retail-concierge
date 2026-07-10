from langchain_core.documents import Document

from src.retrieval.vectorstore import (
    POLICY_INDEX_DIR,
    PRODUCT_INDEX_DIR,
    REVIEW_INDEX_DIR,
    load_index,
)


def retrieve_documents(
    query: str,
    index_type: str,
    top_k: int = 5,
) -> list[Document]:
    """
    Retrieve semantically similar documents
    from a selected FAISS index.
    """

    index_directories = {
        "products": PRODUCT_INDEX_DIR,
        "reviews": REVIEW_INDEX_DIR,
        "policies": POLICY_INDEX_DIR,
    }

    if index_type not in index_directories:
        raise ValueError(
            "index_type must be one of: "
            "products, reviews, policies"
        )

    vector_store = load_index(
        index_directories[index_type]
    )

    documents = vector_store.similarity_search(
        query=query,
        k=top_k,
    )

    return documents


def print_results(
    query: str,
    index_type: str,
    documents: list[Document],
) -> None:
    """
    Print retrieved documents for inspection.
    """

    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print(f"INDEX: {index_type}")
    print("=" * 80)

    for rank, document in enumerate(
        documents,
        start=1,
    ):
        print(f"\nRESULT {rank}")
        print("-" * 40)

        print(document.page_content)

        print("\nMETADATA:")
        print(document.metadata)


def run_retrieval_tests() -> None:
    """
    Run initial semantic retrieval tests.
    """

    test_queries = [
        (
            "I need a lightweight product for travel",
            "products",
        ),
        (
            "customers are unhappy with product quality",
            "reviews",
        ),
        (
            "Can I return a lightly used item?",
            "policies",
        ),
    ]

    for query, index_type in test_queries:

        documents = retrieve_documents(
            query=query,
            index_type=index_type,
            top_k=3,
        )

        print_results(
            query=query,
            index_type=index_type,
            documents=documents,
        )


if __name__ == "__main__":
    run_retrieval_tests()