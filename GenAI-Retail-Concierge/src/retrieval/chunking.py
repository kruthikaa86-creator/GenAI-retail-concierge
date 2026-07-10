from src.retrieval.documents import create_rag_documents


def inspect_document_lengths():
    """
    Inspect document character lengths before deciding
    whether chunking is required.
    """

    documents = create_rag_documents()

    print("\nDOCUMENT LENGTH ANALYSIS\n")

    for document_type in [
        "products",
        "reviews",
        "policies",
    ]:

        document_list = documents[document_type]

        lengths = [
            len(document.page_content)
            for document in document_list
        ]

        print("=" * 60)
        print(
            f"DOCUMENT TYPE: "
            f"{document_type.upper()}"
        )
        print("=" * 60)

        print(
            f"Document count: "
            f"{len(document_list)}"
        )

        print(
            f"Minimum length: "
            f"{min(lengths)} characters"
        )

        print(
            f"Maximum length: "
            f"{max(lengths)} characters"
        )

        print(
            f"Average length: "
            f"{sum(lengths) / len(lengths):.2f} characters"
        )

        print()


if __name__ == "__main__":
    inspect_document_lengths()