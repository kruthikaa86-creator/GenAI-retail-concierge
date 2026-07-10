import pandas as pd

from src.data.loader import load_all_datasets
from src.graph.state import AgentState


LOW_SLA_THRESHOLD = 77.5
HIGH_ESCALATION_THRESHOLD = 8
LOW_CSAT_THRESHOLD = 3.5
SLOW_RESPONSE_THRESHOLD = 22.8
HIGH_HANDLE_TIME_THRESHOLD = 34.5

LOW_CONFIDENCE_THRESHOLD = 0.60


def calculate_sla_risk(
    sla_dataframe: pd.DataFrame,
) -> dict:
    """
    Calculate SLA risk using dataset-derived quartile thresholds.
    """

    latest_date = pd.to_datetime(
        sla_dataframe["date"],
        errors="coerce",
    ).max()

    latest_records = sla_dataframe[
        pd.to_datetime(
            sla_dataframe["date"],
            errors="coerce",
        )
        == latest_date
    ].copy()

    metrics = {
        "latest_date": str(latest_date.date()),
        "sla_met_pct": float(
            latest_records["sla_met_pct"].mean()
        ),
        "escalations": float(
            latest_records["escalations"].mean()
        ),
        "customer_satisfaction_avg": float(
            latest_records[
                "customer_satisfaction_avg"
            ].mean()
        ),
        "first_response_time_min": float(
            latest_records[
                "first_response_time_min"
            ].mean()
        ),
        "avg_handle_time_min": float(
            latest_records[
                "avg_handle_time_min"
            ].mean()
        ),
    }

    risk_reasons = []

    if metrics["sla_met_pct"] < LOW_SLA_THRESHOLD:
        risk_reasons.append(
            "SLA compliance is below the lower quartile threshold"
        )

    if (
        metrics["escalations"]
        > HIGH_ESCALATION_THRESHOLD
    ):
        risk_reasons.append(
            "Escalation volume is above the upper quartile threshold"
        )

    if (
        metrics["customer_satisfaction_avg"]
        < LOW_CSAT_THRESHOLD
    ):
        risk_reasons.append(
            "Customer satisfaction is below the lower quartile threshold"
        )

    if (
        metrics["first_response_time_min"]
        > SLOW_RESPONSE_THRESHOLD
    ):
        risk_reasons.append(
            "First response time is above the upper quartile threshold"
        )

    if (
        metrics["avg_handle_time_min"]
        > HIGH_HANDLE_TIME_THRESHOLD
    ):
        risk_reasons.append(
            "Average handle time is above the upper quartile threshold"
        )

    return {
        "sla_risk": len(risk_reasons) > 0,
        "risk_reasons": risk_reasons,
        "metrics": metrics,
    }


def evaluate_escalation(
    state: AgentState,
) -> dict:
    """
    Evaluate human escalation using confidence,
    VIP signals, tool requirements, and SLA risk.
    """

    datasets = load_all_datasets()

    sla_dataframe = datasets[
        "agent_performance_sla"
    ]

    sla_result = calculate_sla_risk(
        sla_dataframe
    )

    confidence = float(
        state.get("confidence", 1.0)
    )

    vip_signals = state.get(
        "vip_signals",
        {},
    )

    is_vip = bool(
        vip_signals.get("is_vip", False)
    )

    existing_escalation = bool(
        state.get(
            "requires_escalation",
            False,
        )
    )

    existing_reason = state.get(
        "escalation_reason",
        "",
    )

    escalation_reasons = []

    if existing_escalation:
        escalation_reasons.append(
            existing_reason
            or "Previous workflow node requested escalation"
        )

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        escalation_reasons.append(
            "Model confidence is below the escalation threshold"
        )

    if sla_result["sla_risk"]:
        escalation_reasons.extend(
            sla_result["risk_reasons"]
        )

    if is_vip and (
        confidence < 0.75
        or existing_escalation
        or sla_result["sla_risk"]
    ):
        escalation_reasons.append(
            "VIP customer requires white-glove human handling"
        )

    escalation_reasons = list(
        dict.fromkeys(escalation_reasons)
    )

    requires_escalation = (
        len(escalation_reasons) > 0
    )

    trace = list(
        state.get("trace", [])
    )

    trace.append(
        {
            "node": "escalation_evaluation",
            "confidence": confidence,
            "is_vip": is_vip,
            "sla_risk": sla_result["sla_risk"],
            "requires_escalation": requires_escalation,
            "reasons": escalation_reasons,
        }
    )

    return {
        "sla_risk": sla_result["sla_risk"],
        "requires_escalation": requires_escalation,
        "escalation_reason": "; ".join(
            escalation_reasons
        ),
        "trace": trace,
    }


def test_escalation_agent() -> None:
    """
    Run escalation agent smoke tests.
    """

    test_states = [
        {
            "customer_id": "C000001",
            "confidence": 0.90,
            "vip_signals": {
                "is_vip": True,
            },
            "requires_escalation": True,
            "escalation_reason": (
                "Existing return requires human handling"
            ),
            "trace": [],
        },
        {
            "customer_id": "C000002",
            "confidence": 0.40,
            "vip_signals": {
                "is_vip": False,
            },
            "requires_escalation": False,
            "trace": [],
        },
        {
            "customer_id": "C000003",
            "confidence": 0.90,
            "vip_signals": {
                "is_vip": False,
            },
            "requires_escalation": False,
            "trace": [],
        },
    ]

    print("\nESCALATION AGENT TEST\n")

    for state in test_states:
        result = evaluate_escalation(state)

        print("=" * 60)

        print(
            f"Customer ID: "
            f"{state['customer_id']}"
        )

        print(
            f"Confidence: "
            f"{state['confidence']}"
        )

        print(
            f"SLA Risk: "
            f"{result['sla_risk']}"
        )

        print(
            "Requires Escalation: "
            f"{result['requires_escalation']}"
        )

        print(
            "Escalation Reason: "
            f"{result['escalation_reason']}"
        )

        print("Trace:")

        print(result["trace"])


if __name__ == "__main__":
    test_escalation_agent()