from langchain_openai import OpenAIEmbeddings

from src.config import (
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    validate_config,
)


def get_embedding_model() -> OpenAIEmbeddings:
    """
    Create and return the OpenAI embedding model.
    """

    validate_config()

    embedding_model = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )

    return embedding_model


def test_embedding_model() -> None:
    """
    Run a small embedding API smoke test.
    """

    embedding_model = get_embedding_model()

    test_texts = [
        "lightweight laptop for travel",
        "portable notebook computer",
        "standard return policy",
    ]

    print("\nRUNNING EMBEDDING SMOKE TEST\n")

    vectors = embedding_model.embed_documents(
        test_texts
    )

    print(
        f"Texts embedded: "
        f"{len(vectors)}"
    )

    print(
        f"Vector dimensions: "
        f"{len(vectors[0])}"
    )

    print(
        "All vectors have equal dimensions: "
        f"{len(set(len(vector) for vector in vectors)) == 1}"
    )

    print("\nEMBEDDING SMOKE TEST COMPLETE")


if __name__ == "__main__":
    test_embedding_model()