#!/usr/bin/env python3
"""Audit cross-call state, masking, and evidence budgets in saved NaVQA runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SEQUENCES = (0, 3, 4, 6, 16, 21, 22)


def parse_run(value: str) -> tuple[str, str]:
    parts = value.split("|", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--run must be key|result filename")
    return parts[0], parts[1]


def add_error(errors: list[str], location: str, message: str) -> None:
    errors.append(f"{location}: {message}")


def audit_run(
    result_root: Path,
    run: tuple[str, str],
    sequences: tuple[int, ...],
) -> dict[str, Any]:
    key, filename = run
    stats: dict[str, Any] = {
        "questions": 0,
        "attempts": 0,
        "retry_questions": 0,
        "text_calls": 0,
        "multi_text_attempts": 0,
        "empty_exhausted_calls": 0,
        "duplicate_blocked_calls": 0,
        "interleaved_attempts": 0,
        "max_unique_evidence_per_attempt": 0,
        "errors": [],
    }
    errors: list[str] = stats["errors"]

    for sequence in sequences:
        path = result_root / str(sequence) / "human_qa" / filename
        result = json.loads(path.read_text(encoding="utf-8"))
        config = result.get("config", {})
        interleaved = (
            config.get("controller_retrieval_mode")
            == "interleaved_single_call"
        )
        max_retrieval_rounds = int(config.get("max_retrieval_rounds", 5))
        responses = result.get("responses", [])
        if len(responses) != 30 or result.get("in_progress", True):
            add_error(
                errors,
                f"S{sequence}",
                f"incomplete result ({len(responses)}/30, "
                f"in_progress={result.get('in_progress')})",
            )
        stats["questions"] += len(responses)

        for question_index, response in enumerate(responses):
            attempts = [
                attempt
                for attempt in response.get("retrieval_attempts", [])
                if isinstance(attempt, dict)
            ]
            stats["attempts"] += len(attempts)
            stats["retry_questions"] += len(attempts) > 1
            previous_episode_id = None

            for attempt in attempts:
                attempt_number = attempt.get("attempt")
                location = f"S{sequence}Q{question_index}A{attempt_number}"
                all_calls = [
                    call
                    for call in attempt.get("calls", [])
                    if isinstance(call, dict)
                ]
                if interleaved:
                    stats["interleaved_attempts"] += 1
                    previous_turn = 0
                    visible_ids: list[int] = []
                    executed_signatures: set[str] = set()
                    executed_count = 0
                    for call_index, call in enumerate(all_calls, start=1):
                        call_location = f"{location}C{call_index}"
                        turn = call.get("controller_turn_id")
                        if not isinstance(turn, int) or turn <= previous_turn:
                            add_error(
                                errors,
                                call_location,
                                "controller turn is missing or not increasing",
                            )
                        elif turn > 0:
                            previous_turn = turn
                        if call.get("tool_batch_size") != 1:
                            add_error(errors, call_location, "tool batch size is not one")
                        if not call.get("tool_call_id"):
                            add_error(errors, call_location, "missing tool call ID")
                        if (
                            call.get("prior_result_ids_visible_to_controller")
                            != visible_ids
                        ):
                            add_error(
                                errors,
                                call_location,
                                "prior result IDs do not match visible history",
                            )
                        selected_ids = [
                            row.get("entry_id")
                            for row in call.get("selected", [])
                            if isinstance(row, dict)
                            and row.get("entry_id") is not None
                        ]
                        if call.get("selected_ids") != selected_ids:
                            add_error(
                                errors,
                                call_location,
                                "controller selected IDs differ from memory trace",
                            )
                        signature = call.get("tool_signature")
                        if call.get("duplicate_blocked"):
                            stats["duplicate_blocked_calls"] += 1
                            if signature not in executed_signatures:
                                add_error(
                                    errors,
                                    call_location,
                                    "blocked signature was not previously executed",
                                )
                            if selected_ids:
                                add_error(
                                    errors,
                                    call_location,
                                    "blocked duplicate returned memories",
                                )
                        else:
                            executed_count += 1
                            if not signature:
                                add_error(errors, call_location, "missing tool signature")
                            elif signature in executed_signatures:
                                add_error(
                                    errors,
                                    call_location,
                                    "duplicate signature executed",
                                )
                            else:
                                executed_signatures.add(signature)
                            for entry_id in selected_ids:
                                if entry_id not in visible_ids:
                                    visible_ids.append(entry_id)
                        if call.get("tool") in {
                            "retrieve_from_time",
                            "retrieve_from_position",
                        } and call.get("retrieval_kind") != "non_qrag":
                            add_error(
                                errors,
                                call_location,
                                "numeric retrieval is not marked non_qrag",
                            )
                    if executed_count > max_retrieval_rounds:
                        add_error(
                            errors,
                            location,
                            "executed retrieval calls exceed controller limit",
                        )
                text_calls = [
                    call
                    for call in all_calls
                    if call.get("tool") == "retrieve_from_text"
                    and not call.get("duplicate_blocked")
                ]
                stats["text_calls"] += len(text_calls)
                stats["multi_text_attempts"] += len(text_calls) > 1
                selected_so_far: list[int] = []
                episode_id = None

                for text_call_index, call in enumerate(text_calls, start=1):
                    call_location = f"{location}C{text_call_index}"
                    call_episode_id = call.get(
                        "qrag_episode_id", call.get("text_episode_id")
                    )
                    if episode_id is None:
                        episode_id = call_episode_id
                    elif call_episode_id != episode_id:
                        add_error(errors, call_location, "episode ID changed within attempt")

                    selected = [
                        item.get("entry_id") for item in call.get("selected", [])
                    ]
                    if None in selected:
                        add_error(errors, call_location, "selected entry without ID")
                    if len(selected) != len(set(selected)):
                        add_error(errors, call_location, "duplicate ID within call")
                    overlap = set(selected_so_far) & set(selected)
                    if overlap:
                        add_error(
                            errors,
                            call_location,
                            f"repeated IDs across calls: {sorted(overlap)}",
                        )

                    before_ids = call.get("episode_selected_entry_ids_before")
                    after_ids = call.get("episode_selected_entry_ids_after")
                    if before_ids != selected_so_far:
                        add_error(errors, call_location, "incorrect episode IDs before call")
                    expected_after = selected_so_far + selected
                    if after_ids != expected_after:
                        add_error(errors, call_location, "incorrect episode IDs after call")

                    budget = call.get("question_evidence_budget")
                    before = call.get("question_budget_remaining_before")
                    after = call.get("question_budget_remaining_after")
                    if not all(isinstance(value, int) for value in (budget, before, after)):
                        add_error(errors, call_location, "missing integer budget fields")
                    else:
                        expected_before = max(0, budget - len(selected_so_far))
                        expected_remaining = max(0, before - len(selected))
                        if before != expected_before or after != expected_remaining:
                            add_error(errors, call_location, "inconsistent budget accounting")
                        if call.get("budget_exhausted") != (after == 0):
                            add_error(errors, call_location, "incorrect exhausted flag")
                        if before == 0:
                            if selected:
                                add_error(errors, call_location, "selected evidence after exhaustion")
                            else:
                                stats["empty_exhausted_calls"] += 1

                    if call.get("effective_requested_k") != len(selected):
                        add_error(errors, call_location, "effective k differs from returned count")

                    if key == "qrag_static":
                        if call.get("qrag_state_format") != "controller":
                            add_error(errors, call_location, "state format is not controller")
                        components = call.get("state_components", [])
                        if len(components) < 2 or components[1] != call.get("query"):
                            add_error(errors, call_location, "tool query absent from static state")
                    elif key == "qrag":
                        if call.get("qrag_state_format") != "controller":
                            add_error(errors, call_location, "state format is not controller")
                        steps = call.get("steps", [])
                        if interleaved:
                            if len(selected) > 1:
                                add_error(
                                    errors,
                                    call_location,
                                    "interleaved Q-RAG returned more than one item",
                                )
                            if call.get("retrieval_method") != (
                                "qrag_interleaved_top1_zero_shot"
                            ):
                                add_error(
                                    errors,
                                    call_location,
                                    "Q-RAG call is not marked interleaved top-1",
                                )
                        if len(steps) != len(selected):
                            add_error(errors, call_location, "step count differs from selection count")
                        for step_index, step in enumerate(steps):
                            components = step.get("state_components", [])
                            expected_length = 2 + len(selected_so_far) + step_index
                            if len(components) != expected_length:
                                add_error(errors, call_location, "incorrect sequential state length")
                            elif components[1] != call.get("query"):
                                add_error(errors, call_location, "tool query absent from step state")
                            if (
                                step_index < len(selected)
                                and step.get("selected_entry_id") != selected[step_index]
                            ):
                                add_error(errors, call_location, "step/selection order mismatch")

                    selected_so_far = expected_after

                if episode_id is not None:
                    if previous_episode_id == episode_id:
                        add_error(errors, location, "retry reused the previous episode ID")
                    previous_episode_id = episode_id
                stats["max_unique_evidence_per_attempt"] = max(
                    stats["max_unique_evidence_per_attempt"], len(selected_so_far)
                )

    stats["error_count"] = len(errors)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument(
        "--sequences", default=" ".join(str(value) for value in DEFAULT_SEQUENCES)
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sequences = tuple(int(value) for value in args.sequences.replace(",", " ").split())
    report = {
        key: audit_run(args.result_root, (key, filename), sequences)
        for key, filename in args.run
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if any(item["error_count"] for item in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
