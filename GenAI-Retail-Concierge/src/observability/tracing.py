import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.graph.state import AgentState


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRACE_DIRECTORY = (
    PROJECT_ROOT
    / "logs"
)

TRACE_FILE = (
    TRACE_DIRECTORY
    / "agent_traces.jsonl"
)


def make_json_serializable(
    value: Any,
) -> Any:
    """
    Convert values into JSON-safe Python values.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            make_json_serializable(item)
            for item in value
        ]

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    return str(value)


def build_trace_record(
    state: AgentState,
) -> dict[str, Any]:
    """
    Build one complete workflow audit record.
    """

    vip_signals = state.get(
        "vip_signals",
        {},
    )

    trace_record = {
        "logged_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "customer_id": state.get(
            "customer_id",
            "",
        ),
        "user_query": state.get(
            "user_query",
            "",
        ),
        "intent": state.get(
            "intent",
            "",
        ),
        "is_vip": vip_signals.get(
            "is_vip",
            False,
        ),
        "vip_score": vip_signals.get(
            "vip_score",
            0,
        ),
        "confidence": state.get(
            "confidence",
            0.0,
        ),
        "sla_risk": state.get(
            "sla_risk",
            False,
        ),
        "requires_escalation": state.get(
            "requires_escalation",
            False,
        ),
        "escalation_reason": state.get(
            "escalation_reason",
            "",
        ),
        "tool_name": state.get(
            "tool_name",
            "",
        ),
        "tool_result": state.get(
            "tool_result",
            {},
        ),
        "action_taken": state.get(
            "action_taken",
            False,
        ),
        "retrieval_sources": state.get(
            "retrieval_sources",
            [],
        ),
        "final_answer": state.get(
            "final_answer",
            "",
        ),
        "trace": state.get(
            "trace",
            [],
        ),
    }

    return make_json_serializable(
        trace_record
    )


def log_workflow_trace(
    state: AgentState,
) -> dict[str, Any]:
    """
    Append one workflow audit record to JSONL.
    """

    TRACE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    trace_record = build_trace_record(
        state
    )

    with TRACE_FILE.open(
        "a",
        encoding="utf-8",
    ) as trace_file:
        trace_file.write(
            json.dumps(
                trace_record,
                ensure_ascii=False,
            )
            + "\n"
        )

    return trace_record


def read_trace_records() -> list[dict[str, Any]]:
    """
    Read all saved workflow trace records.
    """

    if not TRACE_FILE.exists():
        return []

    records = []

    with TRACE_FILE.open(
        "r",
        encoding="utf-8",
    ) as trace_file:

        for line in trace_file:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


def test_tracing() -> None:
    """
    Run a local observability smoke test.
    """

    test_state: AgentState = {
        "user_query": (
            "Can I return my item?"
        ),
        "customer_id": "C000001",
        "intent": "return",
        "vip_signals": {
            "is_vip": True,
            "vip_score": 2,
        },
        "confidence": 0.95,
        "sla_risk": False,
        "requires_escalation": True,
        "escalation_reason": (
            "Existing return requires "
            "human handling"
        ),
        "tool_name": "return_inquiry",
        "tool_result": {
            "success": True,
            "return_found": True,
        },
        "action_taken": True,
        "retrieval_sources": [],
        "final_answer": (
            "Priority human handoff required."
        ),
        "trace": [
            {
                "node": "personalization",
            },
            {
                "node": "intent_router",
                "intent": "return",
            },
            {
                "node": "return_tool",
            },
            {
                "node": "human_handoff",
            },
        ],
    }

    record = log_workflow_trace(
        test_state
    )

    print(
        "\nOBSERVABILITY TRACE TEST\n"
    )

    print(
        f"Trace file: {TRACE_FILE}"
    )

    print(
        "Record logged successfully: "
        f"{bool(record)}"
    )

    records = read_trace_records()

    print(
        "Total saved trace records: "
        f"{len(records)}"
    )

    print("\nLATEST TRACE RECORD:")

    print(
        json.dumps(
            records[-1],
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    test_tracing()