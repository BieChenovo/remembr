"""Pure helpers for deterministic retrieval-call control and tracing."""

import ast
import copy
import json
import re
from typing import Any, Dict, Sequence, Tuple


RETRIEVAL_TOOL_NAMES = frozenset(
    {
        "retrieve_from_text",
        "retrieve_from_position",
        "retrieve_from_time",
    }
)


class ControllerToolChoiceError(ValueError):
    """Raised when one controller turn does not contain exactly one choice."""


class RetrievalCallGate:
    """Attempt-scoped budget, signature set, and visible-result ledger."""

    def __init__(self, max_rounds: int, duplicate_replan_limit: int = 2):
        if int(max_rounds) < 1:
            raise ValueError("max_rounds must be positive")
        if int(duplicate_replan_limit) < 1:
            raise ValueError("duplicate_replan_limit must be positive")
        self.max_rounds = int(max_rounds)
        self.duplicate_replan_limit = int(duplicate_replan_limit)
        self.reset()

    def reset(self) -> None:
        self.round_count = 0
        self.executed_signatures = set()
        self.visible_result_ids = []
        self.consecutive_duplicate_replans = 0

    @property
    def can_retrieve(self) -> bool:
        return self.round_count < self.max_rounds

    def is_duplicate(self, signature: str) -> bool:
        return signature in self.executed_signatures

    def record_duplicate(self) -> Tuple[int, bool]:
        """Record one blocked replan without consuming an executed round."""

        self.consecutive_duplicate_replans += 1
        return (
            self.consecutive_duplicate_replans,
            self.consecutive_duplicate_replans >= self.duplicate_replan_limit,
        )

    def commit(self, signature: str, selected_ids: Sequence[Any]) -> int:
        if self.is_duplicate(signature):
            raise ValueError("Duplicate retrieval signatures cannot be committed")
        if not self.can_retrieve:
            raise ValueError("Retrieval round limit reached")
        self.executed_signatures.add(signature)
        self.round_count += 1
        self.consecutive_duplicate_replans = 0
        for entry_id in selected_ids:
            if entry_id not in self.visible_result_ids:
                self.visible_result_ids.append(entry_id)
        return self.round_count


def ensure_single_tool_choice(choices: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the sole controller choice, rejecting parallel/mixed batches."""

    if len(choices) != 1:
        raise ControllerToolChoiceError(
            "A controller turn must select exactly one retrieval tool or the "
            f"response tool; received {len(choices)} choices"
        )
    return choices[0]


def _canonical_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, str):
        return " ".join(value.strip().split())
    if isinstance(value, float):
        value = round(value, 9)
        return 0.0 if value == 0 else value
    return value


def raw_tool_query(arguments: Any) -> Any:
    """Extract the user-facing query while preserving its original value."""

    if isinstance(arguments, dict) and "x" in arguments:
        return arguments["x"]
    return arguments


def normalize_tool_query(tool_name: str, arguments: Any) -> Any:
    """Normalize equivalent queries for stable per-attempt deduplication."""

    query = raw_tool_query(arguments)
    if tool_name == "retrieve_from_text" and isinstance(query, str):
        return " ".join(query.strip().casefold().split())
    if tool_name == "retrieve_from_time" and isinstance(query, str):
        normalized = " ".join(query.strip().split())
        match = re.fullmatch(r"(\d{1,2}):(\d{1,2}):(\d{1,2})", normalized)
        if match:
            hour, minute, second = (int(value) for value in match.groups())
            if hour < 24 and minute < 60 and second < 60:
                return f"{hour:02d}:{minute:02d}:{second:02d}"
        return normalized.casefold()
    if tool_name == "retrieve_from_position" and isinstance(query, str):
        try:
            parsed = ast.literal_eval(query)
        except (SyntaxError, ValueError):
            parsed = query
        query = parsed
    return _canonical_value(query)


def tool_call_signature(tool_name: str, arguments: Any) -> Tuple[str, str]:
    """Return the normalized query and a stable ``(tool, query)`` signature."""

    normalized = normalize_tool_query(tool_name, arguments)
    canonical = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return normalized, f"{tool_name}:{canonical}"


def selected_entry_ids(trace: Dict[str, Any]) -> list:
    """Read stable IDs from a memory trace without leaking ``None`` values."""

    return [
        row.get("entry_id")
        for row in trace.get("selected", [])
        if isinstance(row, dict) and row.get("entry_id") is not None
    ]


def qrag_state_components(trace: Dict[str, Any]) -> list:
    """Expose the state used for the first (and in v3 only) Q-RAG action."""

    components = trace.get("qrag_state_components")
    if isinstance(components, list):
        return list(components)
    steps = trace.get("steps", [])
    if steps and isinstance(steps[0], dict):
        return list(steps[0].get("state_components", []))
    return list(trace.get("state_components", []))


def merge_controller_trace(
    memory_trace: Sequence[Dict[str, Any]],
    controller_events: Sequence[Dict[str, Any]],
) -> list:
    """Overlay controller provenance and retain blocked, unexecuted attempts."""

    output = []
    used_memory_indices = set()
    for event in controller_events:
        memory_index = event.get("memory_trace_index")
        if isinstance(memory_index, int) and 0 <= memory_index < len(memory_trace):
            record = copy.deepcopy(memory_trace[memory_index])
            used_memory_indices.add(memory_index)
            record["memory_call_index"] = record.get("call_index")
        else:
            record = {
                "tool": event.get("tool"),
                "query": event.get("raw_query"),
                "parsed_query": event.get("normalized_query"),
                "selected": [],
                "ranking": [],
                "steps": [],
                "returned_count": 0,
                "returned_context": event.get("blocked_reason", ""),
            }
        record.update(
            {
                key: copy.deepcopy(value)
                for key, value in event.items()
                if key != "memory_trace_index"
            }
        )
        record["call_index"] = len(output) + 1
        output.append(record)
    for memory_index, record in enumerate(memory_trace):
        if memory_index not in used_memory_indices:
            output.append(copy.deepcopy(record))
    return output
