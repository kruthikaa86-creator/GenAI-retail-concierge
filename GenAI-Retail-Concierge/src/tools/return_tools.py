from typing import Any

import pandas as pd

from src.data.loader import load_all_datasets
from src.graph.state import AgentState


VALID_RETURN_STATUSES = {
    "pending",
    "denied",
    "approved",
    "escalated",
}

VALID_RESOLUTION_TYPES = {
    "exchange",
    "refund",
    "store_credit",
}


def get_customer_returns(
    returns: pd.DataFrame,
    customer_id: str,
) -> list[dict[str, Any]]:
    """
    Retrieve return records belonging to one customer.
    """

    customer_returns = returns[
        returns["customer_id"] == customer_id
    ].copy()

    if customer_returns.empty:
        return []

    customer_returns["request_date"] = (
        pd.to_datetime(
            customer_returns["request_date"],
            errors="coerce",
        )
    )

    customer_returns = customer_returns.sort_values(
        "request_date",
        ascending=False,
    )

    return customer_returns.to_dict(
        orient="records"
    )


def validate_return_record(
    return_record: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate an existing return record against
    the supplied returns dataset schema.
    """

    status = return_record.get("status")

    resolution_type = return_record.get(
        "resolution_type"
    )

    validation_errors = []

    if status not in VALID_RETURN_STATUSES:
        validation_errors.append(
            f"Unknown return status: {status}"
        )

    if (
        resolution_type
        not in VALID_RESOLUTION_TYPES
    ):
        validation_errors.append(
            "Unknown resolution type: "
            f"{resolution_type}"
        )

    is_valid = len(validation_errors) == 0

    return {
        "is_valid": is_valid,
        "validation_errors": validation_errors,
    }


def determine_return_action(
    return_record: dict[str, Any],
) -> dict[str, Any]:
    """
    Determine the next allowed workflow action
    from the existing return status.

    This function does not approve or deny returns.
    """

    status = return_record["status"]

    if status == "approved":
        return {
            "action": "confirm_approved_return",
            "requires_human": False,
            "message": (
                "The return is already approved."
            ),
        }

    if status == "denied":
        return {
            "action": "confirm_denied_return",
            "requires_human": False,
            "message": (
                "The return has been denied."
            ),
        }

    if status == "pending":
        return {
            "action": "policy_validation_required",
            "requires_human": False,
            "message": (
                "The return is pending and requires "
                "policy validation before any action."
            ),
        }

    if status == "escalated":
        return {
            "action": "human_escalation_required",
            "requires_human": True,
            "message": (
                "The return is already escalated and "
                "requires human-agent handling."
            ),
        }

    return {
        "action": "no_action",
        "requires_human": True,
        "message": (
            "The return status could not be processed."
        ),
    }


def process_return_inquiry(
    state: AgentState,
) -> dict:
    """
    Execute the local return inquiry workflow:

    retrieve -> validate -> determine action -> confirm
    """

    customer_id = state.get(
        "customer_id",
        "",
    )

    datasets = load_all_datasets()

    returns = datasets["returns"]

    customer_returns = get_customer_returns(
        returns=returns,
        customer_id=customer_id,
    )

    trace = list(
        state.get("trace", [])
    )

    if not customer_returns:

        tool_result = {
            "success": False,
            "customer_id": customer_id,
            "return_found": False,
            "message": (
                "No return record was found for "
                "this customer."
            ),
        }

        trace.append(
            {
                "node": "return_tool",
                "customer_id": customer_id,
                "return_found": False,
                "action": "no_return_found",
            }
        )

        return {
            "tool_name": "return_inquiry",
            "tool_input": {
                "customer_id": customer_id,
            },
            "tool_result": tool_result,
            "action_taken": False,
            "requires_escalation": False,
            "trace": trace,
        }

    return_record = customer_returns[0]

    validation = validate_return_record(
        return_record
    )

    if not validation["is_valid"]:

        tool_result = {
            "success": False,
            "return_found": True,
            "return_id": return_record[
                "return_id"
            ],
            "validation": validation,
            "message": (
                "The return record failed validation."
            ),
        }

        trace.append(
            {
                "node": "return_tool",
                "customer_id": customer_id,
                "return_id": return_record[
                    "return_id"
                ],
                "return_found": True,
                "record_valid": False,
                "action": "validation_failed",
            }
        )

        return {
            "tool_name": "return_inquiry",
            "tool_input": {
                "customer_id": customer_id,
            },
            "tool_result": tool_result,
            "action_taken": False,
            "requires_escalation": True,
            "escalation_reason": (
                "Return record validation failed"
            ),
            "trace": trace,
        }

    action_result = determine_return_action(
        return_record
    )

    tool_result = {
        "success": True,
        "return_found": True,
        "return_id": return_record["return_id"],
        "order_id": return_record["order_id"],
        "reason": return_record["reason"],
        "status": return_record["status"],
        "request_date": str(
            return_record["request_date"]
        ),
        "resolution_type": return_record[
            "resolution_type"
        ],
        "refund_amount": float(
            return_record["refund_amount"]
        ),
        "workflow_action": action_result["action"],
        "message": action_result["message"],
    }

    trace.append(
        {
            "node": "return_tool",
            "customer_id": customer_id,
            "return_id": return_record["return_id"],
            "return_found": True,
            "record_valid": True,
            "status": return_record["status"],
            "action": action_result["action"],
        }
    )

    return {
        "tool_name": "return_inquiry",
        "tool_input": {
            "customer_id": customer_id,
        },
        "tool_result": tool_result,
        "action_taken": True,
        "requires_escalation": action_result[
            "requires_human"
        ],
        "escalation_reason": (
            "Existing return requires human handling"
            if action_result["requires_human"]
            else ""
        ),
        "trace": trace,
    }


def test_return_tool() -> None:
    """
    Run return tool smoke tests.
    """

    test_customer_ids = [
        "C000001",
        "C999999",
    ]

    print("\nRETURN TOOL TEST\n")

    for customer_id in test_customer_ids:

        state: AgentState = {
            "customer_id": customer_id,
            "trace": [],
        }

        result = process_return_inquiry(
            state
        )

        print("=" * 60)

        print(
            f"Customer ID: {customer_id}"
        )

        print(
            f"Tool Name: {result['tool_name']}"
        )

        print(
            f"Action Taken: "
            f"{result['action_taken']}"
        )

        print(
            "Requires Escalation: "
            f"{result['requires_escalation']}"
        )

        print("Tool Result:")

        print(
            result["tool_result"]
        )

        print("Trace:")

        print(
            result["trace"]
        )


if __name__ == "__main__":
    test_return_tool()