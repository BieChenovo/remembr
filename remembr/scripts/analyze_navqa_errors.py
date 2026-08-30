#!/usr/bin/env python3
"""Build an aggregate, self-contained NaVQA error-analysis report.

The report applies the ReMEmbR paper thresholds to the selected sequences,
counts invalid structured outputs as incorrect, and audits whether each
question's reference ``context`` caption is present in the candidate window
used by the local evaluator. Retrieval-vs-reader claims are only made when the
corresponding response persisted a retrieval trace.
"""

from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SEQUENCES = (0, 3, 4, 6, 16, 21, 22)
POSITION_THRESHOLD_M = 15.0
TIME_THRESHOLD_MIN = 2.0
DURATION_THRESHOLD_MIN = 2.0

TYPE_LABELS = {
    "binary": "二元判断",
    "position": "空间位置",
    "time": "时间点",
    "duration": "持续时间",
    "text": "描述文本",
}

REASON_LABELS = {
    "correct": "正确",
    "generation_or_json_failure": "生成/JSON 失败",
    "answer_type_mismatch": "题型字段不匹配",
    "required_field_null": "必需答案字段为空",
    "position_threshold_miss": "位置误差超过 15 m",
    "time_threshold_miss": "时间误差超过 2 min",
    "duration_threshold_miss": "持续时间误差超过 2 min",
    "binary_label_mismatch": "二元答案错误",
    "text_semantic_mismatch": "文本语义判分错误",
}

# These notes come from the saved worker logs and a manual inspection of the
# seven failed records.  They are evidence annotations, not model corrections.
MANUAL_FAILURE_NOTES = {
    "SHORT_31d323c2-930f-4111-85b0-d92e960ee600": (
        "256-token 输出在 duration 字段前被截断；内外重试仍生成同类长 JSON。"
    ),
    "3_LONG_3a877fd4-beaf-4f2f-b1dc-9d461707dd2d": (
        "模型选对 position 类型但返回 position=null；问题含 hourse 拼写错误，"
        "且参考 context 位于当前候选窗口之后。"
    ),
    "4_LONG_d0815d48-8e28-417e-90f0-997e708e73b4": (
        "模型把 call the police 理解为拨打 911，输出 text；参考 caption 只把"
        "目标描述成带标志的黄色杆，未表达其报警电话语义。"
    ),
    "4_SHORT_65732436-fbcb-4c0f-b2ed-51c07e7e4821": (
        "模型未找到 skateboard，返回 time=null；参考 caption 本身也没有描述"
        "skateboard，属于明显 caption 信息瓶颈。"
    ),
    "16_SHORT_c20410c1-b187-4759-94e2-e47287a25574": (
        "参考 context 明确写 overcast，但位于当前候选窗口之前；模型返回无法"
        "访问天气信息而非 binary=no。"
    ),
    "21_MEDIUM_18ebe275-8985-4af8-8030-3d125dc48c00": (
        "两次回答均使用 text 而未填 binary；第一次语义接近 No，第二次改成 Yes。"
    ),
    "22_MEDIUM_7f9b905a-4363-4634-84b4-12700b064460": (
        "模型回答 moderately busy 但未填 binary；crowded 判断主观，参考 context"
        "仅描述远处有少量行人。"
    ),
}

STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "been", "before", "by", "can",
    "current", "did", "do", "does", "driving", "during", "for", "from", "get",
    "go", "had", "has", "have", "how", "i", "in", "is", "it", "located",
    "me", "minutes", "moving", "my", "of", "on", "or", "see", "seen", "side",
    "started", "take", "that", "the", "there", "this", "time", "to", "today",
    "very", "was", "were", "what", "when", "where", "which", "while", "with",
    "you", "your",
}

REFERENCE_BLOCK_RE = re.compile(
    r"At time=(?P<time>.*?), the robot was at an average position of "
    r"(?P<position>\[.*?\])\.The robot saw the following: "
    r"(?P<caption>.*?)(?=\n\nAt time=|\Z)",
    re.DOTALL,
)


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def pct(numerator: int | float, denominator: int | float, digits: int = 1) -> str:
    if not denominator:
        return "—"
    return f"{100 * float(numerator) / float(denominator):.{digits}f}%"


def fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list)):
        return esc(json.dumps(value, ensure_ascii=False))
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return esc(value)
    if not math.isfinite(numeric):
        return "—"
    return f"{numeric:.{digits}f}{suffix}"


def parse_sequences(raw: str) -> list[int]:
    return [int(value) for value in raw.replace(",", " ").split()]


def normalize_token(token: str) -> str:
    token = token.lower()
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def target_tokens(question: str) -> list[str]:
    tail = question.splitlines()[-1]
    tokens = []
    for token in re.findall(r"[A-Za-z]+", tail.lower()):
        normalized = normalize_token(token)
        if len(normalized) >= 3 and normalized not in STOPWORDS:
            tokens.append(normalized)
    return sorted(set(tokens))


def token_coverage(tokens: list[str], captions: Iterable[str]) -> tuple[int, int]:
    caption_tokens = {
        normalize_token(token)
        for caption in captions
        for token in re.findall(r"[A-Za-z]+", caption.lower())
    }
    hits = sum(token in caption_tokens for token in tokens)
    return hits, len(tokens)


def parse_reference_context(context: str) -> list[dict[str, Any]]:
    blocks = []
    for match in REFERENCE_BLOCK_RE.finditer(context.strip()):
        position = ast.literal_eval(match.group("position"))
        blocks.append(
            {
                "display_time": match.group("time"),
                "position": [float(value) for value in position],
                "caption": match.group("caption").strip(),
            }
        )
    if not blocks:
        raise ValueError(f"Could not parse reference context: {context[:200]!r}")
    return blocks


def squared_distance(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def map_reference_entries(
    blocks: list[dict[str, Any]], captions: list[dict[str, Any]]
) -> list[int]:
    candidates: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(captions):
        candidates[item["caption"].strip()].append(index)
    mapped = []
    for block in blocks:
        matches = candidates.get(block["caption"], [])
        if not matches:
            raise ValueError("Reference context caption is absent from caption memory")
        mapped.append(
            min(
                matches,
                key=lambda index: squared_distance(
                    block["position"], captions[index]["position"]
                ),
            )
        )
    return mapped


def candidate_range(
    question: dict[str, Any], captions: list[dict[str, Any]]
) -> tuple[int, int]:
    starts = [float(item["file_start"][:-4]) for item in captions]
    ends = [float(item["file_end"][:-4]) for item in captions]
    start = min(
        range(len(captions)),
        key=lambda index: abs(starts[index] - question["start_time"]),
    )
    end = min(
        range(len(captions)),
        key=lambda index: abs(ends[index] - question["end_time"]),
    )
    return start, end


def prediction_value(question_type: str, response: dict[str, Any]) -> Any:
    value = response.get(question_type)
    if value is not None:
        return value
    nested = response.get("response")
    if isinstance(nested, dict):
        return nested.get(question_type)
    return None


def ground_truth(question: dict[str, Any]) -> Any:
    answers = question.get("answers", {})
    question_type = question["type"]
    if question_type in answers:
        return answers[question_type]
    text = answers.get("text")
    if isinstance(text, list):
        return text[0] if text else None
    return text


def classify_output(
    question: dict[str, Any], response: dict[str, Any]
) -> tuple[bool, str, float | None, float | None, str | None]:
    question_type = question["type"]
    failure = response.get("evaluation_failure")
    if failure:
        nested = response.get("response")
        if not isinstance(nested, dict):
            return False, "generation_or_json_failure", None, None, None
        if nested.get("type") != question_type:
            return False, "answer_type_mismatch", None, None, None
        return False, "required_field_null", None, None, None

    error = response.get("error") or {}
    if question_type == "binary":
        value = error.get("binary_iscorrect")
        correct = bool(value)
        return correct, "correct" if correct else "binary_label_mismatch", value, 1.0, None
    if question_type == "text":
        value = error.get("text_iscorrect")
        correct = bool(value)
        return correct, "correct" if correct else "text_semantic_mismatch", value, 1.0, None
    if question_type == "position":
        value = error.get("position_error")
        correct = isinstance(value, (int, float)) and value <= POSITION_THRESHOLD_M
        return (
            correct,
            "correct" if correct else "position_threshold_miss",
            float(value) if value is not None else None,
            POSITION_THRESHOLD_M,
            "m",
        )
    if question_type == "time":
        value = error.get("time_error")
        correct = isinstance(value, (int, float)) and value <= TIME_THRESHOLD_MIN
        return (
            correct,
            "correct" if correct else "time_threshold_miss",
            float(value) if value is not None else None,
            TIME_THRESHOLD_MIN,
            "min",
        )
    if question_type == "duration":
        value = error.get("duration_error")
        correct = isinstance(value, (int, float)) and value <= DURATION_THRESHOLD_MIN
        return (
            correct,
            "correct" if correct else "duration_threshold_miss",
            float(value) if value is not None else None,
            DURATION_THRESHOLD_MIN,
            "min",
        )
    raise ValueError(f"Unsupported question type: {question_type}")


def numeric_severity(question_type: str, value: float | None) -> str | None:
    if question_type not in {"position", "time", "duration"}:
        return None
    if value is None:
        return "invalid"
    if question_type == "position":
        if value <= 15:
            return "correct"
        if value <= 30:
            return "near"
        if value <= 60:
            return "medium"
        return "large"
    if value <= 2:
        return "correct"
    if value <= 5:
        return "near"
    if value <= 10:
        return "medium"
    return "large"


def group_accuracy(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        group: {
            "total": len(items),
            "correct": sum(item["official_correct"] for item in items),
            "invalid": sum(item["output_status"] == "invalid" for item in items),
            "accuracy": sum(item["official_correct"] for item in items) / len(items),
        }
        for group, items in sorted(grouped.items())
    }


def build_analysis(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    reference_entry_total = 0
    reference_entry_in_pool = 0
    outside_before = 0
    outside_after = 0
    outside_deltas = []

    for sequence in parse_sequences(args.sequences):
        questions_path = args.questions_root / str(sequence) / "human_qa.json"
        captions_path = (
            args.captions_root
            / str(sequence)
            / "captions"
            / f"{args.caption_file}.json"
        )
        result_path = (
            args.result_root
            / str(sequence)
            / "human_qa"
            / args.result_name.format(tag=args.tag)
        )
        questions_document = json.loads(questions_path.read_text(encoding="utf-8"))
        questions = questions_document["data"]
        captions = json.loads(captions_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        responses = result["responses"]
        if len(questions) != 30 or len(responses) != len(questions):
            raise ValueError(
                f"Sequence {sequence}: expected 30 aligned questions/responses, "
                f"got {len(questions)}/{len(responses)}"
            )

        caption_starts = [float(item["file_start"][:-4]) for item in captions]
        caption_ends = [float(item["file_end"][:-4]) for item in captions]

        for index, (question, response) in enumerate(zip(questions, responses)):
            blocks = parse_reference_context(question["context"])
            reference_ids = map_reference_entries(blocks, captions)
            start_index, end_index = candidate_range(question, captions)
            reference_inside = [
                start_index <= entry_id <= end_index for entry_id in reference_ids
            ]
            reference_entry_total += len(reference_ids)
            reference_entry_in_pool += sum(reference_inside)

            outside = []
            for entry_id, inside in zip(reference_ids, reference_inside):
                if inside:
                    continue
                if entry_id < start_index:
                    direction = "before"
                    delta = caption_starts[start_index] - caption_starts[entry_id]
                    outside_before += 1
                else:
                    direction = "after"
                    delta = caption_starts[entry_id] - caption_ends[end_index]
                    outside_after += 1
                outside_deltas.append(delta)
                outside.append(
                    {"entry_id": entry_id, "direction": direction, "gap_seconds": delta}
                )

            correct, reason, metric_error, threshold, unit = classify_output(
                question, response
            )
            tokens = target_tokens(question["question"])
            reference_captions = [captions[entry_id]["caption"] for entry_id in reference_ids]
            candidate_captions = [
                item["caption"] for item in captions[start_index : end_index + 1]
            ]
            reference_hits, token_count = token_coverage(tokens, reference_captions)
            candidate_hits, _ = token_coverage(tokens, candidate_captions)

            flags = []
            if not all(reference_inside):
                flags.append("reference_context_outside_candidate_pool")
            if response.get("evaluation_failure"):
                flags.append("output_contract_failure")
            if token_count and not reference_hits:
                flags.append("reference_caption_target_token_gap_proxy")
            retrieval_trace_available = "retrieval_trace" in response
            if not correct and not retrieval_trace_available:
                flags.append("retrieval_trace_missing_root_cause_unresolved")

            error = response.get("error") or {}
            judge = error.get("text_judge") or {}
            row = {
                "sequence": sequence,
                "question_index": index,
                "display_index": index + 1,
                "question_id": question["id"],
                "length_category": question.get("length_category", "unknown").lower(),
                "length_seconds": question.get("length"),
                "question_type": question["type"],
                "category": question.get("category"),
                "question": question["question"],
                "ground_truth": ground_truth(question),
                "prediction": prediction_value(question["type"], response),
                "response_type": (
                    response.get("type")
                    or (response.get("response") or {}).get("type")
                    if isinstance(response.get("response"), dict)
                    else response.get("type")
                ),
                "answer_text": response.get("text"),
                "official_correct": bool(correct),
                "outcome": "correct" if correct else "wrong",
                "output_status": "invalid" if response.get("evaluation_failure") else "valid",
                "observable_reason": reason,
                "observable_reason_label": REASON_LABELS[reason],
                "metric_error": metric_error,
                "threshold": threshold,
                "unit": unit,
                "numeric_severity": numeric_severity(question["type"], metric_error),
                "elapsed_seconds": response.get("elapsed"),
                "evaluation_failure": response.get("evaluation_failure"),
                "text_judge_rationale": judge.get("rationale"),
                "retrieval_trace_available": retrieval_trace_available,
                "retrieval_call_count": len(response.get("retrieval_trace", [])),
                "candidate_start_index": start_index,
                "candidate_end_index": end_index,
                "candidate_count": end_index - start_index + 1,
                "reference_entry_ids": reference_ids,
                "reference_entry_count": len(reference_ids),
                "reference_all_in_candidate_pool": all(reference_inside),
                "reference_entry_recall_in_pool": sum(reference_inside) / len(reference_inside),
                "reference_outside_details": outside,
                "target_tokens": tokens,
                "reference_target_token_hits": reference_hits,
                "candidate_target_token_hits": candidate_hits,
                "target_token_count": token_count,
                "reference_target_token_coverage": (
                    reference_hits / token_count if token_count else None
                ),
                "diagnostic_flags": flags,
                "manual_failure_note": (
                    MANUAL_FAILURE_NOTES.get(question["id"])
                    if args.include_baseline_manual_notes
                    else None
                ),
                "reference_captions": reference_captions,
                "result_path": str(result_path),
            }
            rows.append(row)

    expected_questions = 30 * len(parse_sequences(args.sequences))
    if len(rows) != expected_questions:
        raise ValueError(f"Expected {expected_questions} questions, got {len(rows)}")

    correct = sum(row["official_correct"] for row in rows)
    invalid = sum(row["output_status"] == "invalid" for row in rows)
    reasons = Counter(
        row["observable_reason"] for row in rows if not row["official_correct"]
    )
    reference_questions_in_pool = sum(
        row["reference_all_in_candidate_pool"] for row in rows
    )
    latency_values = [
        row["elapsed_seconds"]
        for row in rows
        if isinstance(row["elapsed_seconds"], (int, float))
    ]

    summary = {
        "questions": len(rows),
        "correct": correct,
        "wrong": len(rows) - correct,
        "invalid": invalid,
        "retrieval_trace_questions": sum(
            row["retrieval_trace_available"] for row in rows
        ),
        "strict_overall_accuracy": correct / len(rows),
        "valid_output_coverage": (len(rows) - invalid) / len(rows),
        "latency_mean_seconds": statistics.mean(latency_values),
        "latency_median_seconds": statistics.median(latency_values),
        "accuracy_by_type": group_accuracy(rows, "question_type"),
        "accuracy_by_sequence": group_accuracy(rows, "sequence"),
        "accuracy_by_length": group_accuracy(rows, "length_category"),
        "wrong_reason_counts": dict(reasons),
        "reference_context_audit": {
            "questions_with_context": len(rows),
            "questions_all_reference_entries_in_candidate_pool": reference_questions_in_pool,
            "questions_with_reference_entries_outside_candidate_pool": (
                len(rows) - reference_questions_in_pool
            ),
            "reference_entries": reference_entry_total,
            "reference_entries_in_candidate_pool": reference_entry_in_pool,
            "reference_entries_outside_candidate_pool": (
                reference_entry_total - reference_entry_in_pool
            ),
            "outside_before": outside_before,
            "outside_after": outside_after,
            "outside_within_6_1_seconds": sum(delta <= 6.1 for delta in outside_deltas),
            "outside_gap_seconds": sorted(outside_deltas),
            "accuracy_when_all_reference_in_pool": (
                sum(
                    row["official_correct"]
                    for row in rows
                    if row["reference_all_in_candidate_pool"]
                )
                / reference_questions_in_pool
            ),
            "accuracy_when_reference_outside_pool": (
                sum(
                    row["official_correct"]
                    for row in rows
                    if not row["reference_all_in_candidate_pool"]
                )
                / (len(rows) - reference_questions_in_pool)
            ),
        },
    }
    return {
        "version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_tag": args.tag,
        "thresholds": {
            "position_m": POSITION_THRESHOLD_M,
            "time_min": TIME_THRESHOLD_MIN,
            "duration_min": DURATION_THRESHOLD_MIN,
        },
        "limitations": [
            f"Retrieval traces are available for {sum(row['retrieval_trace_available'] for row in rows)}/{len(rows)} responses.",
            "Reference context is treated as a derived support candidate, not yet as a proven exhaustive gold support set.",
            "Question-token coverage is a lexical proxy and cannot detect paraphrases or visual evidence omitted by captions.",
        ],
        "summary": summary,
        "rows": rows,
    }


def bar_rows(groups: dict[str, dict[str, Any]], labels: dict[str, str] | None = None) -> str:
    rendered = []
    for key, item in groups.items():
        label = labels.get(key, key) if labels else key
        width = 100 * item["accuracy"]
        rendered.append(
            f"""
            <div class="bar-row">
              <span>{esc(label)}</span>
              <div class="bar-track"><i style="width:{width:.2f}%"></i></div>
              <strong>{item['correct']}/{item['total']} · {width:.1f}%</strong>
            </div>"""
        )
    return "".join(rendered)


def build_html(analysis: dict[str, Any], json_path: Path, csv_path: Path) -> str:
    summary = analysis["summary"]
    rows = analysis["rows"]
    audit = summary["reference_context_audit"]

    reason_rows = []
    for reason, count in sorted(
        summary["wrong_reason_counts"].items(), key=lambda item: (-item[1], item[0])
    ):
        reason_rows.append(
            f"<div class='cause'><span>{esc(REASON_LABELS[reason])}</span>"
            f"<i style='width:{100 * count / summary['wrong']:.2f}%'></i>"
            f"<strong>{count}</strong></div>"
        )

    numeric_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row["numeric_severity"]:
            numeric_counts[row["question_type"]][row["numeric_severity"]] += 1
    severity_labels = {
        "correct": "阈值内",
        "near": "轻度越界",
        "medium": "中度越界",
        "large": "严重越界",
        "invalid": "无有效数值",
    }
    numeric_html = []
    for question_type in ("position", "time", "duration"):
        counts = numeric_counts[question_type]
        total = sum(counts.values())
        segments = "".join(
            f"<span class='sev-{key}' style='width:{100 * counts[key] / total:.2f}%' "
            f"title='{severity_labels[key]} {counts[key]}'>{counts[key] or ''}</span>"
            for key in ("correct", "near", "medium", "large", "invalid")
            if counts[key]
        )
        numeric_html.append(
            f"<div class='stack-row'><b>{TYPE_LABELS[question_type]}</b>"
            f"<div class='stack'>{segments}</div><strong>{total}</strong></div>"
        )

    table_rows = []
    for row in rows:
        failure_detail = row["manual_failure_note"] or row["evaluation_failure"] or ""
        reference_note = (
            f"全部在池内 · {row['reference_entry_count']} 条"
            if row["reference_all_in_candidate_pool"]
            else f"有参考 context 在池外 · {row['reference_outside_details']}"
        )
        metric = (
            fmt(row["metric_error"], 2, f" {row['unit'] or ''}")
            if row["metric_error"] is not None
            else "—"
        )
        support_captions = "\n\n".join(row["reference_captions"])
        search = " ".join(
            [
                row["question"],
                row["question_id"],
                row["answer_text"] or "",
                row["observable_reason_label"],
                failure_detail,
                support_captions,
            ]
        ).lower()
        table_rows.append(
            f"""
            <tr data-outcome="{row['outcome']}" data-status="{row['output_status']}"
                data-type="{row['question_type']}" data-sequence="{row['sequence']}"
                data-length="{row['length_category']}"
                data-pool="{'inside' if row['reference_all_in_candidate_pool'] else 'outside'}"
                data-search="{esc(search)}">
              <td><b>S{row['sequence']}·Q{row['display_index']}</b><small>{esc(row['length_category'])}</small></td>
              <td><span class="pill type-{row['question_type']}">{TYPE_LABELS[row['question_type']]}</span></td>
              <td><span class="pill outcome-{row['outcome']}">{'正确' if row['official_correct'] else '错误'}</span></td>
              <td class="wide"><details><summary>{esc(row['question'].splitlines()[-1].strip())}</summary>
                <p><b>完整问题：</b>{esc(row['question'])}</p>
                <p><b>ID：</b><code>{esc(row['question_id'])}</code></p>
                <p><b>参考答案：</b>{fmt(row['ground_truth'])}</p>
                <p><b>模型答案：</b>{fmt(row['prediction'])}</p>
                <p><b>文本输出：</b>{esc(row['answer_text'] or '—')}</p>
              </details></td>
              <td><b>{esc(row['observable_reason_label'])}</b><small>{metric}</small>
                {f'<p class="warn">{esc(failure_detail)}</p>' if failure_detail else ''}</td>
              <td><details><summary>{esc(reference_note)}</summary>
                <p><b>entry IDs：</b>{esc(row['reference_entry_ids'])}</p>
                <p><b>候选范围：</b>{row['candidate_start_index']}–{row['candidate_end_index']}（{row['candidate_count']} 条）</p>
                <p><b>目标词覆盖：</b>{row['reference_target_token_hits']}/{row['target_token_count']}；词：{esc(row['target_tokens'])}</p>
                <p class="caption">{esc(support_captions)}</p>
              </details></td>
              <td>{fmt(row['elapsed_seconds'], 1, ' s')}</td>
            </tr>"""
        )

    generated = esc(analysis["generated_at"])
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NaVQA {summary['questions']} 题聚合错误分析</title>
<style>
:root{{--bg:#08111f;--panel:#111d30;--panel2:#17263d;--line:#2a3b55;--text:#eef5ff;--muted:#9cabc1;--blue:#55a7ff;--green:#31d18b;--red:#ff6f7d;--orange:#ffb45e;--purple:#ae91ff;--cyan:#43d8dc}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 8% 0,#173b66 0,transparent 30rem),var(--bg);color:var(--text);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}}
.wrap{{width:min(1500px,calc(100% - 36px));margin:auto;padding:42px 0 70px}} header{{display:flex;justify-content:space-between;gap:24px;align-items:end;margin-bottom:24px}} h1{{font-size:clamp(30px,5vw,54px);line-height:1.05;margin:5px 0}} h2{{margin:0 0 16px;font-size:20px}} p{{margin:6px 0}} .eyebrow{{color:var(--blue);font-size:12px;font-weight:800;letter-spacing:.16em;text-transform:uppercase}} .muted,small{{color:var(--muted)}} .source{{text-align:right;color:var(--muted);font-size:12px}}
.cards{{display:grid;grid-template-columns:repeat(6,1fr);gap:11px;margin-bottom:18px}} .card,.panel{{border:1px solid var(--line);background:linear-gradient(145deg,rgba(22,38,61,.97),rgba(12,24,41,.97));border-radius:16px;box-shadow:0 18px 45px rgba(0,0,0,.16)}} .card{{padding:17px}} .card span{{display:block;color:var(--muted);font-size:11px}} .card strong{{display:block;font-size:25px;margin-top:5px}} .panel{{padding:22px;margin-bottom:18px}} .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} .grid3{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}}
.bar-row{{display:grid;grid-template-columns:90px 1fr 125px;gap:10px;align-items:center;margin:10px 0}} .bar-track{{height:14px;background:#091423;border-radius:99px;overflow:hidden}} .bar-track i{{display:block;height:100%;background:linear-gradient(90deg,var(--blue),var(--cyan));border-radius:99px}} .bar-row strong{{text-align:right;font-size:12px}}
.cause{{display:grid;grid-template-columns:180px 1fr 35px;gap:10px;align-items:center;margin:9px 0}} .cause:has(i){{position:relative}} .cause i{{display:block;height:12px;background:var(--red);border-radius:99px}} .cause strong{{text-align:right}}
.stack-row{{display:grid;grid-template-columns:90px 1fr 30px;gap:10px;align-items:center;margin:12px 0}} .stack{{display:flex;height:27px;background:#091423;border-radius:8px;overflow:hidden}} .stack span{{display:flex;align-items:center;justify-content:center;color:#06101c;font-size:11px;font-weight:900}} .sev-correct{{background:var(--green)}} .sev-near{{background:#ffd166}} .sev-medium{{background:var(--orange)}} .sev-large{{background:var(--red)}} .sev-invalid{{background:#9ca3af}}
.callout{{border-left:4px solid var(--orange);padding:13px 16px;background:#211d1d;border-radius:8px;margin:12px 0}} .finding{{margin:0;padding-left:21px;display:grid;gap:8px}}
.toolbar{{display:flex;flex-wrap:wrap;gap:9px;margin-bottom:14px}} input,select{{color:var(--text);background:#0a1525;border:1px solid var(--line);border-radius:9px;padding:10px 12px}} input{{min-width:280px;flex:1}} .table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:12px}} table{{border-collapse:collapse;width:100%;min-width:1450px}} th,td{{padding:12px;border-bottom:1px solid #25364f;text-align:left;vertical-align:top}} th{{position:sticky;top:0;background:#192941;color:#bbc8da;font-size:11px;letter-spacing:.06em;text-transform:uppercase;z-index:2}} tbody tr:hover{{background:rgba(85,167,255,.055)}} td small{{display:block}} .wide{{min-width:360px;max-width:520px}} summary{{cursor:pointer;font-weight:650}} details p{{font-size:12px;color:#cbd7e7}} code{{color:#acd5ff}} .caption{{white-space:pre-wrap;max-width:520px}} .warn{{color:#ffabb3}} .pill{{display:inline-block;padding:3px 8px;border-radius:99px;font-size:11px;font-weight:750;white-space:nowrap}} .outcome-correct{{color:#64e9ac;background:#123b30}} .outcome-wrong{{color:#ff9faa;background:#47202a}} .type-position{{color:#ffc985;background:#3c2f1e}} .type-time{{color:#cbbcff;background:#30294e}} .type-duration{{color:#85eeef;background:#183e43}} .type-binary{{color:#9ad0ff;background:#173754}} .type-text{{color:#d1d7e0;background:#323b48}}
footer{{color:var(--muted);font-size:12px;margin-top:16px}} @media(max-width:1100px){{.cards{{grid-template-columns:repeat(3,1fr)}}.grid2,.grid3{{grid-template-columns:1fr}}}} @media(max-width:700px){{.wrap{{width:calc(100% - 22px);padding-top:24px}}header{{display:block}}.source{{text-align:left;margin-top:10px}}.cards{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main class="wrap">
<header><div><div class="eyebrow">ReMEmbR retrieval run · error audit</div><h1>NaVQA {summary['questions']} 题聚合错误分析</h1><p class="muted">run={esc(analysis['run_tag'])} · Qwen3-8B · no thinking · 256 tokens · VILA1.5-13B 3 s captions · America/Los_Angeles</p></div><div class="source">生成于 {generated}<br>JSON：{esc(json_path)}<br>CSV：{esc(csv_path)}</div></header>
<section class="cards">
  <article class="card"><span>严格 Overall</span><strong>{pct(summary['correct'], summary['questions'])}</strong><small>{summary['correct']}/{summary['questions']}，失效按错误计</small></article>
  <article class="card"><span>错误总数</span><strong>{summary['wrong']}</strong><small>含 {summary['invalid']} 个输出失效</small></article>
  <article class="card"><span>有效输出覆盖</span><strong>{pct(summary['questions']-summary['invalid'], summary['questions'])}</strong><small>{summary['questions']-summary['invalid']}/{summary['questions']}</small></article>
  <article class="card"><span>参考 context 入池</span><strong>{pct(audit['questions_all_reference_entries_in_candidate_pool'], summary['questions'])}</strong><small>{audit['questions_all_reference_entries_in_candidate_pool']}/{summary['questions']} 题全部入池</small></article>
  <article class="card"><span>平均耗时</span><strong>{summary['latency_mean_seconds']:.1f}s</strong><small>中位数 {summary['latency_median_seconds']:.1f}s</small></article>
  <article class="card"><span>参考 caption 映射</span><strong>{audit['reference_entries']}</strong><small>{summary['questions']} 题全部可精确映射</small></article>
</section>
<section class="panel"><h2>结论边界</h2><div class="callout"><b>{summary['wrong']} 个错误可按可观察现象归类；{summary['retrieval_trace_questions']}/{summary['questions']} 题保存了 retrieval query、top-k entry ID 与分数。</b> 只有保存了 trace 的题目才可以进一步区分 retrieval 失败和 reader 失败。</div>
<ol class="finding"><li>按论文阈值，严格 overall 为 {summary['correct']}/{summary['questions']}；描述题准确率不是全题型 overall。</li><li>参考 context 共 {audit['reference_entries']} 条，全部能映射回 caption memory；{audit['reference_entries_outside_candidate_pool']} 条位于当前候选窗口外，影响 {audit['questions_with_reference_entries_outside_candidate_pool']} 题。</li><li>参考 context 全部入池时准确率 {pct(round(audit['accuracy_when_all_reference_in_pool']*audit['questions_all_reference_entries_in_candidate_pool']), audit['questions_all_reference_entries_in_candidate_pool'])}；有 context 在池外时准确率 {pct(round(audit['accuracy_when_reference_outside_pool']*audit['questions_with_reference_entries_outside_candidate_pool']), audit['questions_with_reference_entries_outside_candidate_pool'])}。</li><li>池外条目中 {audit['outside_within_6_1_seconds']} 条仅差约一个到两个 3 s caption，属于边界索引问题；其余是更明显的标注窗口不一致。</li></ol></section>

<section class="grid2"><article class="panel"><h2>{summary['wrong']} 个错误的可观察原因</h2>{''.join(reason_rows)}</article><article class="panel"><h2>按题型严格准确率</h2>{bar_rows(summary['accuracy_by_type'], TYPE_LABELS)}</article></section>
<section class="grid3"><article class="panel"><h2>按序列</h2>{bar_rows(summary['accuracy_by_sequence'])}</article><article class="panel"><h2>按长度</h2>{bar_rows(summary['accuracy_by_length'], {'short':'短','medium':'中','long':'长'})}</article><article class="panel"><h2>数值题误差分层</h2>{''.join(numeric_html)}<p class="muted">位置：15/30/60 m；时间与持续时间：2/5/10 min 分层。</p></article></section>

<section class="panel"><h2>逐题诊断</h2><div class="toolbar">
<input id="search" type="search" placeholder="搜索问题、答案、错误或参考 caption">
<select id="outcome"><option value="wrong" selected>仅错误</option><option value="correct">仅正确</option><option value="all">全部结果</option></select>
<select id="status"><option value="all">全部输出状态</option><option value="invalid">仅输出失效</option><option value="valid">仅有效输出</option></select>
<select id="type"><option value="all">全部题型</option>{''.join(f'<option value="{key}">{label}</option>' for key,label in TYPE_LABELS.items())}</select>
<select id="sequence"><option value="all">全部序列</option>{''.join(f'<option value="{seq}">序列 {seq}</option>' for seq in DEFAULT_SEQUENCES)}</select>
<select id="pool"><option value="all">全部候选池状态</option><option value="outside">参考 context 在池外</option><option value="inside">参考 context 全部入池</option></select>
<span id="visible" class="muted"></span></div><div class="table-wrap"><table><thead><tr><th>题号</th><th>题型</th><th>结果</th><th>问题/真值/预测</th><th>可观察错误</th><th>参考 context 与候选池</th><th>耗时</th></tr></thead><tbody id="body">{''.join(table_rows)}</tbody></table></div></section>
<footer>参考 context 是数据文件自带、可映射到 caption 的证据候选；在完成数据来源审计前，本报告不把它称为穷尽式 gold support。词覆盖仅作 caption 信息缺失代理指标。</footer>
</main><script>
const controls=['search','outcome','status','type','sequence','pool'].map(id=>document.getElementById(id));
const rows=[...document.querySelectorAll('#body tr')], visible=document.getElementById('visible');
function filter(){{const q=controls[0].value.trim().toLowerCase(), outcome=controls[1].value,status=controls[2].value,type=controls[3].value,seq=controls[4].value,pool=controls[5].value;let n=0;rows.forEach(r=>{{const ok=(!q||r.dataset.search.includes(q))&&(outcome==='all'||r.dataset.outcome===outcome)&&(status==='all'||r.dataset.status===status)&&(type==='all'||r.dataset.type===type)&&(seq==='all'||r.dataset.sequence===seq)&&(pool==='all'||r.dataset.pool===pool);r.hidden=!ok;if(ok)n++}});visible.textContent=`显示 ${{n}} / ${{rows.length}} 题`}}
controls.forEach(c=>c.addEventListener('input',filter));filter();
</script></body></html>"""


def write_outputs(analysis: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "error_analysis.json"
    csv_path = output_dir / "question_diagnostics.csv"
    support_path = output_dir / "reference_context_manifest.jsonl"
    html_path = output_dir / "index.html"

    json_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_fields = [
        "sequence", "question_index", "question_id", "length_category",
        "question_type", "official_correct", "output_status", "observable_reason",
        "metric_error", "threshold", "unit", "candidate_count",
        "reference_entry_ids", "reference_all_in_candidate_pool",
        "reference_entry_recall_in_pool", "target_tokens",
        "reference_target_token_coverage", "diagnostic_flags", "elapsed_seconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in analysis["rows"]:
            writer.writerow(
                {
                    key: json.dumps(row[key], ensure_ascii=False)
                    if isinstance(row[key], (list, dict))
                    else row[key]
                    for key in csv_fields
                }
            )
    with support_path.open("w", encoding="utf-8") as handle:
        for row in analysis["rows"]:
            record = {
                "question_id": row["question_id"],
                "sequence_id": row["sequence"],
                "reference_entry_ids": row["reference_entry_ids"],
                "reference_entry_count": row["reference_entry_count"],
                "reference_all_in_candidate_pool": row[
                    "reference_all_in_candidate_pool"
                ],
                "reference_outside_details": row["reference_outside_details"],
                "support_status": "derived_from_navqa_context",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    html_path.write_text(build_html(analysis, json_path, csv_path), encoding="utf-8")
    print(html_path)
    print(json_path)
    print(csv_path)
    print(support_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", default="0,3,4,6,16,21,22")
    parser.add_argument("--questions-root", type=Path, default=Path("artifacts/questions"))
    parser.add_argument("--captions-root", type=Path, default=Path("artifacts/captions"))
    parser.add_argument("--caption-file", default="captions_VILA1.5-13b_3_secs")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--result-name",
        default="remembr+qwen3:8b__captions_VILA1.5-13b_3_secs_{tag}.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include-baseline-manual-notes",
        action="store_true",
        help=(
            "Attach the seven hand-audited failure notes from the original "
            "B0 run; leave disabled for new retriever ablations"
        ),
    )
    args = parser.parse_args()
    analysis = build_analysis(args)
    write_outputs(analysis, args.output_dir)


if __name__ == "__main__":
    main()
