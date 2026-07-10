from langchain_core.documents import Document

from src.generation.llm import get_llm
from src.generation.prompts import get_rag_prompt
from src.retrieval.retriever import retrieve_documents


def get_source_label(
    document: Document,
) -> str:
    """
    Create a citation label from document metadata.
    """

    source = document.metadata.get("source")

    if source == "products":

        product_id = document.metadata.get(
            "product_id"
        )

        return f"[PRODUCT:{product_id}]"

    if source == "product_reviews":

        product_id = document.metadata.get(
            "product_id"
        )

        return f"[REVIEW:{product_id}]"

    if source == "operations_policies":

        policy_id = document.metadata.get(
            "policy_id"
        )

        return f"[POLICY:{policy_id}]"

    return "[SOURCE:UNKNOWN]"


def build_context(
    documents: list[Document],
) -> str:
    """
    Convert retrieved documents into labelled
    context for the language model.
    """

    context_sections = []

    for document in documents:

        source_label = get_source_label(
            document
        )

        context_section = (
            f"{source_label}\n"
            f"{document.page_content}"
        )

        context_sections.append(
            context_section
        )

    return "\n\n".join(
        context_sections
    )


def run_rag(
    question: str,
    index_type: str,
    top_k: int = 3,
) -> dict:
    """
    Run the basic grounded RAG pipeline.
    """

    documents = retrieve_documents(
        query=question,
        index_type=index_type,
        top_k=top_k,
    )

    context = build_context(
        documents
    )

    prompt = get_rag_prompt()

    llm = get_llm()

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    sources = [
        get_source_label(document)
        for document in documents
    ]

    return {
        "question": question,
        "answer": response.content,
        "sources": sources,
        "retrieved_documents": documents,
    }


def run_rag_tests() -> None:
    """
    Run initial end-to-end RAG tests.
    """

    test_queries = [
        (
            "I need a lightweight product for travel.",
            "products",
        ),
        (
            "Are customers unhappy with product quality?",
            "reviews",
        ),
        (
            "Can I return a lightly used item?",
            "policies",
        ),
    ]

    for question, index_type in test_queries:

        result = run_rag(
            question=question,
            index_type=index_type,
            top_k=3,
        )

        print("\n" + "=" * 80)

        print(
            f"QUESTION: "
            f"{result['question']}"
        )

        print("=" * 80)

        print("\nANSWER:")

        print(
            result["answer"]
        )

        print("\nRETRIEVED SOURCES:")

        for source in result["sources"]:
            print(source)


if __name__ == "__main__":
    run_rag_tests()