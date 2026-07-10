import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]

ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-small"

LLM_MODEL = "gpt-4o"


def validate_config() -> None:
    """
    Validate required project configuration.
    """

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY was not found in the .env file."
        )


if __name__ == "__main__":
    validate_config()

    print("Configuration loaded successfully")

    print(
        f"Embedding model: {EMBEDDING_MODEL}"
    )

    print(
        f"LLM model: {LLM_MODEL}"
    )