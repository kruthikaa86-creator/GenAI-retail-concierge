from typing import Any

import pandas as pd

from src.data.loader import load_all_datasets


PRIMARY_KEYS = {
    "agent_performance_sla": "sla_record_id",
    "customer_interactions": "interaction_id",
    "customers": "customer_id",
    "product_reviews": "review_id",
    "returns": "return_id",
}


def validate_dataframe(
    dataset_name: str,
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Generate basic data-quality statistics for one DataFrame.
    """

    report = {
        "dataset": dataset_name,
        "rows": len(dataframe),
        "columns": len(dataframe.columns),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "missing_values": dataframe.isna().sum().to_dict(),
    }

    primary_key = PRIMARY_KEYS.get(dataset_name)

    if primary_key is not None:
        report["primary_key"] = primary_key

        report["missing_primary_keys"] = int(
            dataframe[primary_key].isna().sum()
        )

        report["duplicate_primary_keys"] = int(
            dataframe[primary_key].duplicated().sum()
        )

    return report


def print_validation_report(
    report: dict[str, Any],
) -> None:
    """
    Print one dataset validation report.
    """

    print("\n" + "=" * 80)
    print(f"DATASET: {report['dataset']}")
    print("=" * 80)

    print(f"Rows: {report['rows']}")
    print(f"Columns: {report['columns']}")
    print(f"Duplicate rows: {report['duplicate_rows']}")

    if "primary_key" in report:
        print(f"Primary key: {report['primary_key']}")

        print(
            "Missing primary keys: "
            f"{report['missing_primary_keys']}"
        )

        print(
            "Duplicate primary keys: "
            f"{report['duplicate_primary_keys']}"
        )

    print("\nMISSING VALUES:")

    missing_values = report["missing_values"]

    columns_with_missing_values = {
        column: count
        for column, count in missing_values.items()
        if count > 0
    }

    if not columns_with_missing_values:
        print("None")

    else:
        for column, count in columns_with_missing_values.items():
            print(f"{column}: {count}")


def validate_json_policies(
    policies: list[dict],
) -> None:
    """
    Validate the operations policy JSON records.
    """

    print("\n" + "=" * 80)
    print("DATASET: operations_policies")
    print("=" * 80)

    print(f"Records: {len(policies)}")

    required_fields = {
        "id",
        "question",
        "answer",
        "category",
        "source",
    }

    missing_field_records = 0

    for policy in policies:
        if not required_fields.issubset(policy.keys()):
            missing_field_records += 1

    policy_ids = [
        policy.get("id")
        for policy in policies
    ]

    duplicate_policy_ids = (
        len(policy_ids) - len(set(policy_ids))
    )

    print(
        "Records missing required fields: "
        f"{missing_field_records}"
    )

    print(
        "Duplicate policy IDs: "
        f"{duplicate_policy_ids}"
    )


def validate_foreign_key(
    child_dataframe: pd.DataFrame,
    child_column: str,
    parent_dataframe: pd.DataFrame,
    parent_column: str,
    relationship_name: str,
) -> None:
    """
    Validate whether child values exist in a parent dataset.
    """

    child_values = set(
        child_dataframe[child_column]
        .dropna()
        .unique()
    )

    parent_values = set(
        parent_dataframe[parent_column]
        .dropna()
        .unique()
    )

    orphan_values = child_values - parent_values

    print("\n" + "-" * 80)
    print(f"RELATIONSHIP: {relationship_name}")
    print("-" * 80)

    print(f"Unique child IDs: {len(child_values)}")
    print(f"Unique parent IDs: {len(parent_values)}")
    print(f"Orphan IDs: {len(orphan_values)}")

    if orphan_values:
        print(
            "Sample orphan IDs: "
            f"{list(orphan_values)[:10]}"
        )


def validate_datetime_order(
    dataframe: pd.DataFrame,
    start_column: str,
    end_column: str,
    dataset_name: str,
) -> None:
    """
    Check whether end timestamps occur before start timestamps.
    """

    start_dates = pd.to_datetime(
        dataframe[start_column],
        errors="coerce",
    )

    end_dates = pd.to_datetime(
        dataframe[end_column],
        errors="coerce",
    )

    invalid_dates = end_dates < start_dates

    print("\n" + "-" * 80)
    print(f"DATETIME CHECK: {dataset_name}")
    print("-" * 80)

    print(
        "Invalid datetime order rows: "
        f"{int(invalid_dates.sum())}"
    )

    print(
        f"Invalid {start_column} values: "
        f"{int(start_dates.isna().sum())}"
    )

    print(
        f"Invalid {end_column} values: "
        f"{int(end_dates.isna().sum())}"
    )


def validate_relationships(
    datasets: dict,
) -> None:
    """
    Validate known relationships between project datasets.
    """

    print("\n" + "=" * 80)
    print("RELATIONSHIP VALIDATION")
    print("=" * 80)

    validate_foreign_key(
        datasets["customer_interactions"],
        "customer_id",
        datasets["customers"],
        "customer_id",
        "customer_interactions.customer_id -> customers.customer_id",
    )

    validate_foreign_key(
        datasets["customer_interactions"],
        "product_id",
        datasets["products"],
        "product_id",
        "customer_interactions.product_id -> products.product_id",
    )

    validate_foreign_key(
        datasets["product_reviews"],
        "customer_id",
        datasets["customers"],
        "customer_id",
        "product_reviews.customer_id -> customers.customer_id",
    )

    validate_foreign_key(
        datasets["product_reviews"],
        "product_id",
        datasets["products"],
        "product_id",
        "product_reviews.product_id -> products.product_id",
    )

    validate_foreign_key(
        datasets["returns"],
        "customer_id",
        datasets["customers"],
        "customer_id",
        "returns.customer_id -> customers.customer_id",
    )

    validate_foreign_key(
        datasets["chat_conversation_metadata"],
        "customer_id",
        datasets["customers"],
        "customer_id",
        "chat_conversation_metadata.customer_id -> customers.customer_id",
    )

    validate_datetime_order(
        datasets["chat_conversation_metadata"],
        "started_at",
        "ended_at",
        "chat_conversation_metadata",
    )


def validate_product_conflicts(
    products: pd.DataFrame,
) -> None:
    """
    Count product IDs with conflicting attribute values.
    """

    columns_to_check = [
        "name",
        "category",
        "subcategory",
        "brand",
        "price",
        "stock",
        "image_url",
        "status",
        "avg_rating",
        "description",
    ]

    print("\n" + "=" * 80)
    print("PRODUCT ATTRIBUTE CONFLICT VALIDATION")
    print("=" * 80)

    print(
        "Unique product IDs: "
        f"{products['product_id'].nunique()}"
    )

    for column in columns_to_check:

        unique_counts = (
            products
            .groupby("product_id")[column]
            .nunique()
        )

        conflicting_product_ids = int(
            (unique_counts > 1).sum()
        )

        print(
            f"{column}: "
            f"{conflicting_product_ids} "
            "product IDs have conflicting values"
        )


def run_validation() -> None:
    """
    Run validation for all raw datasets.
    """

    datasets = load_all_datasets()

    for dataset_name, dataset in datasets.items():

        if isinstance(dataset, pd.DataFrame):

            report = validate_dataframe(
                dataset_name,
                dataset,
            )

            print_validation_report(report)

        elif dataset_name == "operations_policies":

            validate_json_policies(dataset)

    validate_relationships(datasets)

    validate_product_conflicts(
        datasets["products"]
    )


if __name__ == "__main__":
    run_validation()