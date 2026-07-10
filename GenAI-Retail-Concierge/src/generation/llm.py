from langchain_openai import ChatOpenAI

from src.config import (
    LLM_MODEL,
    OPENAI_API_KEY,
    validate_config,
)


def get_llm() -> ChatOpenAI:
    """
    Create and return the project language model.
    """

    validate_config()

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0,
    )

    return llm


def test_llm() -> None:
    """
    Run a small LLM API smoke test.
    """

    llm = get_llm()

    print("\nRUNNING LLM SMOKE TEST\n")

    response = llm.invoke(
        "Reply with exactly: LLM connection successful"
    )

    print(
        f"Model: {LLM_MODEL}"
    )

    print(
        f"Response: {response.content}"
    )

    print("\nLLM SMOKE TEST COMPLETE")


if __name__ == "__main__":
    test_llm()