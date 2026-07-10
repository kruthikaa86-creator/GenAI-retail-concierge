from pathlib import Path
import json
import warnings

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"


def load_csv(file_name: str) -> pd.DataFrame:
    """
    Load a CSV dataset from the raw data directory.

    Malformed rows are reported and skipped so that the
    original raw dataset remains unchanged.
    """

    file_path = DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")

        dataframe = pd.read_csv(
            file_path,
            on_bad_lines="warn"
        )

        for warning in captured_warnings:
            print(
                f"[DATA WARNING] {file_name}: "
                f"{warning.message}"
            )

    return dataframe


def load_json(file_name: str):
    """
    Load a JSON dataset from the raw data directory.
    """

    file_path = DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_all_datasets():
    """
    Load all raw datasets required by the project.
    """

    datasets = {
        "agent_performance_sla": load_csv(
            "agent_performance_sla.csv"
        ),
        "chat_conversation_metadata": load_csv(
            "chat_conversation_metadata.csv"
        ),
        "customer_interactions": load_csv(
            "customer_interactions.csv"
        ),
        "customers": load_csv(
            "customers.csv"
        ),
        "operations_policies": load_json(
            "operations_policies.json"
        ),
        "product_reviews": load_csv(
            "product_reviews.csv"
        ),
        "products": load_csv(
            "products.csv"
        ),
        "returns": load_csv(
            "returns.csv"
        ),
    }

    return datasets


if __name__ == "__main__":

    datasets = load_all_datasets()

    print("\nDATASETS LOADED SUCCESSFULLY\n")

    for dataset_name, dataset in datasets.items():

        if isinstance(dataset, pd.DataFrame):
            print(
                f"{dataset_name}: "
                f"{dataset.shape[0]} rows x "
                f"{dataset.shape[1]} columns"
            )

        elif isinstance(dataset, list):
            print(
                f"{dataset_name}: "
                f"{len(dataset)} records"
            )

        elif isinstance(dataset, dict):
            print(
                f"{dataset_name}: "
                f"{len(dataset)} top-level keys"
            )

        else:
            print(
                f"{dataset_name}: "
                f"{type(dataset).__name__}"
            )