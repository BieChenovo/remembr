#!/usr/bin/env python3
"""Plan resumable NaVQA shards and merge them into complete result files.

The planner reuses valid responses from an earlier run, but deliberately
reschedules missing responses and responses containing ``evaluation_failure``.
Each shard writes a distinct eval.py output file, so concurrent workers never
touch the same JSON file.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from collections import defaultdict
from pathlib import Path


DEFAULT_SEQUENCES = (0, 3, 4, 6, 16, 21, 22)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def result_path(
    result_root: Path,
    sequence: int,
    model: str,
    caption_file: str,
    tag: str,
) -> Path:
    name = f"remembr+{model}__{caption_file}_{tag}.json"
    return result_root / str(sequence) / "human_qa" / name


def question_data(question_root: Path, sequence: int) -> list[dict]:
    payload = load_json(question_root / str(sequence) / "human_qa.json")
    return payload["data"]


def response_failed(response: dict) -> bool:
    error = response.get("error", {})
    return "evaluation_failure" in response or "evaluation_failure" in error


def reusable_responses(path: Path, expected_ids: set[str]) -> dict[str, dict]:
    if not path.exists():
        return {}
    payload = load_json(path)
    reusable: dict[str, dict] = {}
    for response in payload.get("responses", []):
        response_id = str(response.get("id", ""))
        if response_id not in expected_ids or response_failed(response):
            continue
        reusable[response_id] = response
    return reusable


def parse_sequences(raw: str) -> list[int]:
    return [int(value) for value in raw.replace(",", " ").split()]


def plan(args: argparse.Namespace) -> int:
    sequences = parse_sequences(args.sequences)
    tasks: list[tuple[int, int]] = []
    reused = 0
    failed_to_retry = 0
    per_sequence: dict[str, dict] = {}

    for sequence in sequences:
        questions = question_data(args.question_root, sequence)
        expected_ids = {str(item["id"]) for item in questions}
        source = result_path(
            args.result_root, sequence, args.model, args.caption_file, args.source_tag
        )
        source_payload = load_json(source) if source.exists() else {"responses": []}
        source_by_id = {
            str(item.get("id", "")): item
            for item in source_payload.get("responses", [])
            if str(item.get("id", "")) in expected_ids
        }
        reusable = reusable_responses(source, expected_ids)
        retry_ids = {
            item_id for item_id, item in source_by_id.items() if response_failed(item)
        }
        failed_to_retry += len(retry_ids)
        reused += len(reusable)
        missing_indices = [
            index
            for index, question in enumerate(questions)
            if str(question["id"]) not in reusable
        ]
        tasks.extend((sequence, index) for index in missing_indices)
        per_sequence[str(sequence)] = {
            "questions": len(questions),
            "reused": len(reusable),
            "scheduled": len(missing_indices),
            "failed_responses_retried": len(retry_ids),
            "source": str(source),
        }

    # Round-robin keeps question types and both remaining sequences spread over
    # all workers. With two workers per GPU this also fills CPU retrieval gaps.
    assignments: list[list[tuple[int, int]]] = [
        [] for _ in range(args.workers)
    ]
    for index, task in enumerate(tasks):
        assignments[index % args.workers].append(task)

    rows: list[dict] = []
    for worker, assigned in enumerate(assignments):
        grouped: dict[int, list[int]] = defaultdict(list)
        for sequence, question_index in assigned:
            grouped[sequence].append(question_index)
        for sequence, indices in sorted(grouped.items()):
            rows.append(
                {
                    "worker": worker,
                    "gpu": worker % args.gpus,
                    "port": args.base_port + (worker % args.gpus),
                    "sequence": sequence,
                    "indices": ",".join(str(value) for value in indices),
                    "shard_tag": (
                        f"{args.target_tag}_shard_w{worker}_s{sequence}"
                    ),
                }
            )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("worker", "gpu", "port", "sequence", "indices", "shard_tag"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    plan_payload = {
        "version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_tag": args.source_tag,
        "target_tag": args.target_tag,
        "workers": args.workers,
        "gpus": args.gpus,
        "questions_reused": reused,
        "questions_scheduled": len(tasks),
        "failed_responses_retried": failed_to_retry,
        "per_sequence": per_sequence,
        "manifest": str(args.manifest),
    }
    atomic_json(args.plan_json, plan_payload)
    print(json.dumps(plan_payload, ensure_ascii=False, indent=2))
    return 0


def compute_metrics(questions: list[dict], responses: list[dict]) -> dict:
    binary_correct = 0
    binary_count = 0
    text_correct = 0
    text_count = 0
    position_error = 0.0
    position_count = 0
    time_error = 0.0
    time_count = 0
    duration_error = 0.0
    duration_count = 0
    failed = 0
    text_skipped = 0

    for question, response in zip(questions, responses):
        error = response.get("error", {})
        if response_failed(response):
            failed += 1
            continue
        question_type = question["type"]
        if question_type == "position" and "position_error" in error:
            position_count += 1
            position_error += float(error["position_error"])
        elif question_type == "binary" and "binary_iscorrect" in error:
            binary_count += 1
            binary_correct += int(error["binary_iscorrect"])
        elif question_type == "text" and "text_iscorrect" in error:
            text_count += 1
            text_correct += int(error["text_iscorrect"])
        elif question_type == "time" and "time_error" in error:
            time_count += 1
            time_error += float(error["time_error"])
        elif question_type == "duration" and "duration_error" in error:
            duration_count += 1
            duration_error += float(error["duration_error"])
        elif question_type == "text":
            text_skipped += 1

    scored = binary_count + text_count + position_count + time_count + duration_count
    descriptive_count = binary_count + text_count
    return {
        "questions_total": len(questions),
        "questions_completed": len(responses),
        "questions_scored": scored,
        "questions_failed": failed,
        "text_questions_skipped": text_skipped,
        "binary_count": binary_count,
        "binary_accuracy": binary_correct / binary_count if binary_count else None,
        "text_count": text_count,
        "text_accuracy": text_correct / text_count if text_count else None,
        "descriptive_count": descriptive_count,
        "descriptive_accuracy": (
            (binary_correct + text_correct) / descriptive_count
            if descriptive_count
            else None
        ),
        "position_count": position_count,
        "position_mean_l2_error": (
            position_error / position_count if position_count else None
        ),
        "time_count": time_count,
        "time_mean_absolute_error": time_error / time_count if time_count else None,
        "duration_count": duration_count,
        "duration_mean_absolute_error": (
            duration_error / duration_count if duration_count else None
        ),
    }


def merge(args: argparse.Namespace) -> int:
    sequences = parse_sequences(args.sequences)
    with args.manifest.open("r", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle, delimiter="\t"))

    summary_path = args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows: list[list[str]] = []

    for sequence in sequences:
        questions = question_data(args.question_root, sequence)
        expected_ids = [str(item["id"]) for item in questions]
        expected_set = set(expected_ids)
        source = result_path(
            args.result_root, sequence, args.model, args.caption_file, args.source_tag
        )
        responses_by_id = reusable_responses(source, expected_set)
        config = load_json(source).get("config", {}) if source.exists() else {}

        relevant_rows = [
            row for row in manifest_rows if int(row["sequence"]) == sequence
        ]
        for row in relevant_rows:
            shard = result_path(
                args.result_root,
                sequence,
                args.model,
                args.caption_file,
                row["shard_tag"],
            )
            # The first parallel launcher wrote a CRLF TSV. Bash retained the
            # trailing CR in the last field and therefore in the temporary
            # shard filename. Normalize that legacy filename atomically before
            # reading it; newly planned manifests always use Unix newlines.
            legacy_cr_shard = shard.with_name(f"{shard.stem}\r{shard.suffix}")
            if not shard.exists() and legacy_cr_shard.exists():
                os.replace(legacy_cr_shard, shard)
            if not shard.exists():
                raise FileNotFoundError(f"Missing shard result: {shard}")
            payload = load_json(shard)
            if payload.get("in_progress", True):
                raise RuntimeError(f"Shard is incomplete: {shard}")
            if not config:
                config = payload.get("config", {})
            scheduled_indices = [int(value) for value in row["indices"].split(",")]
            scheduled_ids = {expected_ids[index] for index in scheduled_indices}
            shard_responses = payload.get("responses", [])
            returned_ids = {str(item.get("id", "")) for item in shard_responses}
            if returned_ids != scheduled_ids:
                raise RuntimeError(
                    f"Shard IDs do not match its plan: {shard}; "
                    f"expected {len(scheduled_ids)}, got {len(returned_ids)}"
                )
            for response in shard_responses:
                responses_by_id[str(response["id"])] = response

        missing = [item_id for item_id in expected_ids if item_id not in responses_by_id]
        if missing:
            raise RuntimeError(
                f"Sequence {sequence} is still missing {len(missing)} responses: {missing}"
            )
        responses = [responses_by_id[item_id] for item_id in expected_ids]
        metrics = compute_metrics(questions, responses)
        config = dict(config)
        config["parallel_resume"] = {
            "source_tag": args.source_tag,
            "target_tag": args.target_tag,
            "shards_merged": len(relevant_rows),
            "merged_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        output = {
            "version": 0.3,
            "in_progress": False,
            "config": config,
            "metrics": metrics,
            "responses": responses,
        }
        target = result_path(
            args.result_root, sequence, args.model, args.caption_file, args.target_tag
        )
        atomic_json(target, output)
        summary_rows.append(
            [
                str(sequence),
                str(metrics["questions_completed"]),
                str(metrics["questions_scored"]),
                str(metrics["questions_failed"]),
                str(metrics["descriptive_count"]),
                str(metrics["descriptive_accuracy"]),
                str(metrics["binary_accuracy"]),
                str(metrics["text_accuracy"]),
                str(metrics["position_mean_l2_error"]),
                str(metrics["time_mean_absolute_error"]),
                str(metrics["duration_mean_absolute_error"]),
                str(target),
            ]
        )
        print(f"Merged sequence {sequence}: {target}")

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "sequence",
                "completed",
                "scored",
                "failed",
                "descriptive_count",
                "descriptive_accuracy",
                "binary_accuracy",
                "text_accuracy",
                "position_l2",
                "time_mae",
                "duration_mae",
                "result",
            )
        )
        writer.writerows(summary_rows)
    print(f"Summary: {summary_path}")
    return 0


def common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--question-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--source-tag", required=True)
    parser.add_argument("--target-tag", required=True)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument(
        "--caption-file", default="captions_VILA1.5-13b_3_secs"
    )
    parser.add_argument(
        "--sequences", default=" ".join(str(value) for value in DEFAULT_SEQUENCES)
    )
    parser.add_argument("--manifest", type=Path, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    common_arguments(plan_parser)
    plan_parser.add_argument("--plan-json", type=Path, required=True)
    plan_parser.add_argument("--workers", type=int, default=8)
    plan_parser.add_argument("--gpus", type=int, default=4)
    plan_parser.add_argument("--base-port", type=int, default=11434)
    plan_parser.set_defaults(func=plan)

    merge_parser = subparsers.add_parser("merge")
    common_arguments(merge_parser)
    merge_parser.add_argument("--summary", type=Path, required=True)
    merge_parser.set_defaults(func=merge)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(parsed.func(parsed))
