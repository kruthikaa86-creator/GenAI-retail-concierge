from typing import Any

import pandas as pd

from src.data.loader import load_all_datasets
from src.graph.state import AgentState


PREMIUM_SEGMENTS = {
    "Gold",
    "Platinum",
}

HIGH_LIFETIME_VALUE_THRESHOLD = 18729.25

HIGH_LOYALTY_POINTS_THRESHOLD = 37537


def get_customer_profile(
    customers: pd.DataFrame,
    customer_id: str,
) -> dict[str, Any]:
    """
    Retrieve one customer profile.
    """

    customer_rows = customers[
        customers["customer_id"] == customer_id
    ]

    if customer_rows.empty:
        return {}

    customer = customer_rows.iloc[0]

    return {
        "customer_id": customer["customer_id"],
        "name": customer["name"],
        "segment": customer["segment"],
        "lifetime_value": float(
            customer["lifetime_value"]
        ),
        "loyalty_points": int(
            customer["loyalty_points"]
        ),
        "preferred_channel": customer[
            "preferred_channel"
        ],
        "preferred_language": customer[
            "preferred_language"
        ],
        "account_status": customer[
            "account_status"
        ],
    }


def get_customer_interactions(
    interactions: pd.DataFrame,
    customer_id: str,
) -> list[dict[str, Any]]:
    """
    Retrieve interaction history for one customer.
    """

    customer_interactions = interactions[
        interactions["customer_id"] == customer_id
    ].copy()

    if customer_interactions.empty:
        return []

    customer_interactions["timestamp"] = (
        pd.to_datetime(
            customer_interactions["timestamp"],
            errors="coerce",
        )
    )

    customer_interactions = (
        customer_interactions
        .sort_values(
            "timestamp",
            ascending=False,
        )
    )

    return customer_interactions.to_dict(
        orient="records"
    )


def derive_vip_signals(
    customer_profile: dict[str, Any],
    customer_interactions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Derive transparent VIP signals from customer
    profile and behavioral history.
    """

    if not customer_profile:
        return {
            "is_vip": False,
            "vip_score": 0,
            "vip_reasons": [
                "Customer profile not found"
            ],
            "purchase_count": 0,
            "wishlist_count": 0,
            "add_to_cart_count": 0,
        }

    segment = customer_profile["segment"]

    lifetime_value = customer_profile[
        "lifetime_value"
    ]

    loyalty_points = customer_profile[
        "loyalty_points"
    ]

    event_counts = {}

    for interaction in customer_interactions:

        event_type = interaction.get(
            "event_type"
        )

        event_counts[event_type] = (
            event_counts.get(event_type, 0) + 1
        )

    purchase_count = event_counts.get(
        "purchase",
        0,
    )

    wishlist_count = event_counts.get(
        "wishlist",
        0,
    )

    add_to_cart_count = event_counts.get(
        "add_to_cart",
        0,
    )

    vip_score = 0
    vip_reasons = []

    if segment in PREMIUM_SEGMENTS:

        vip_score += 1

        vip_reasons.append(
            f"Premium customer segment: {segment}"
        )

    if (
        lifetime_value
        >= HIGH_LIFETIME_VALUE_THRESHOLD
    ):

        vip_score += 1

        vip_reasons.append(
            "Lifetime value is in the high-value range"
        )

    if (
        loyalty_points
        >= HIGH_LOYALTY_POINTS_THRESHOLD
    ):

        vip_score += 1

        vip_reasons.append(
            "Loyalty points are in the high-value range"
        )

    is_vip = vip_score >= 2

    return {
        "is_vip": is_vip,
        "vip_score": vip_score,
        "vip_reasons": vip_reasons,
        "purchase_count": purchase_count,
        "wishlist_count": wishlist_count,
        "add_to_cart_count": add_to_cart_count,
    }


def personalize_customer(
    state: AgentState,
) -> dict:
    """
    Load customer context and inject VIP signals
    into shared LangGraph state.
    """

    customer_id = state.get(
        "customer_id",
        "",
    )

    datasets = load_all_datasets()

    customer_profile = get_customer_profile(
        customers=datasets["customers"],
        customer_id=customer_id,
    )

    customer_interactions = (
        get_customer_interactions(
            interactions=datasets[
                "customer_interactions"
            ],
            customer_id=customer_id,
        )
    )

    vip_signals = derive_vip_signals(
        customer_profile=customer_profile,
        customer_interactions=(
            customer_interactions
        ),
    )

    trace = list(
        state.get("trace", [])
    )

    trace.append(
        {
            "node": "personalization",
            "customer_id": customer_id,
            "profile_found": bool(
                customer_profile
            ),
            "interaction_count": len(
                customer_interactions
            ),
            "is_vip": vip_signals["is_vip"],
            "vip_score": vip_signals[
                "vip_score"
            ],
        }
    )

    return {
        "customer_profile": customer_profile,
        "customer_interactions": (
            customer_interactions
        ),
        "vip_signals": vip_signals,
        "trace": trace,
    }


def test_personalization() -> None:
    """
    Run personalization smoke tests.
    """

    test_customer_ids = [
        "C000001",
        "C000002",
        "C000003",
        "C999999",
    ]

    print("\nPERSONALIZATION TEST\n")

    for customer_id in test_customer_ids:

        state: AgentState = {
            "customer_id": customer_id,
            "trace": [],
        }

        result = personalize_customer(
            state
        )

        print("=" * 60)

        print(
            f"Customer ID: {customer_id}"
        )

        print(
            "Profile:"
        )

        print(
            result["customer_profile"]
        )

        print(
            "VIP Signals:"
        )

        print(
            result["vip_signals"]
        )

        print(
            "Interaction Count: "
            f"{len(result['customer_interactions'])}"
        )


if __name__ == "__main__":
    test_personalization()