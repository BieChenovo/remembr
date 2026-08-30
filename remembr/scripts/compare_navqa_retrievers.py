#!/usr/bin/env python3
"""Build a 210-item NaVQA retrieval-hit and grounded-answer comparison."""

from __future__ import annotations

import argparse
import html
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from remembr.scripts.analyze_navqa_errors import (
    candidate_range,
    classify_output,
    ground_truth,
    map_reference_entries,
    parse_reference_context,
    prediction_value,
)


SEQUENCES = (0, 3, 4, 6, 16, 21, 22)
TYPE_LABELS = {
    "binary": "二元",
    "position": "位置",
    "time": "时间",
    "duration": "时长",
    "text": "描述",
}


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def pct(numerator: int | float, denominator: int | float) -> str:
    return "—" if not denominator else f"{100 * numerator / denominator:.1f}%"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def portable_config(value: Any) -> Any:
    """Remove machine-specific prefixes before publishing result metadata."""

    if isinstance(value, dict):
        return {key: portable_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_config(item) for item in value]
    if isinstance(value, str) and "/projects/remembr/" in value:
        return "remembr/" + value.split("/projects/remembr/", 1)[1]
    return value


def parse_run(value: str) -> tuple[str, str, str]:
    parts = value.split("|", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--run must be key|display label|result filename"
        )
    return parts[0], parts[1], parts[2]


def parse_diagnostic_run(value: str) -> tuple[str, str, str, tuple[int, ...]]:
    parts = value.split("|", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--diagnostic-run must be key|display label|result filename|sequences"
        )
    try:
        sequences = tuple(int(item) for item in parts[3].split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "diagnostic sequences must be comma-separated integers"
        ) from exc
    if not sequences:
        raise argparse.ArgumentTypeError("diagnostic sequences cannot be empty")
    return parts[0], parts[1], parts[2], sequences


def retrieval_calls_for_scoring(response: dict[str, Any]) -> tuple[list[dict], bool]:
    """Return calls that could have informed the persisted answer.

    Evaluation retries may preserve calls from an earlier failed attempt in the
    root ``retrieval_trace``. Those calls did not necessarily inform the final
    successfully parsed answer. Prefer the last successful attempt, including
    its legitimate zero-call case. If the question failed every attempt, use
    only the final failed attempt rather than unioning every retry. Fall back to
    the root trace only for old trace-bearing files without attempt records.
    """

    attempts = response.get("retrieval_attempts")
    if isinstance(attempts, list):
        recorded = [attempt for attempt in attempts if isinstance(attempt, dict)]
        succeeded = [
            attempt
            for attempt in recorded
            if attempt.get("status") == "succeeded"
        ]
        chosen = succeeded[-1] if succeeded else (recorded[-1] if recorded else None)
        if chosen is not None:
            calls = chosen.get("calls")
            return (calls if isinstance(calls, list) else []), True
    if "retrieval_trace" in response:
        calls = response.get("retrieval_trace")
        return (calls if isinstance(calls, list) else []), True
    return [], False


def load_aligned_inputs(args, sequences=SEQUENCES):
    aligned = []
    for sequence in sequences:
        questions = load_json(
            args.questions_root / str(sequence) / "human_qa.json"
        )["data"]
        captions = load_json(
            args.captions_root
            / str(sequence)
            / "captions"
            / f"{args.caption_file}.json"
        )
        if len(questions) != 30:
            raise ValueError(f"Sequence {sequence} must have 30 questions")
        for index, question in enumerate(questions):
            reference_ids = map_reference_entries(
                parse_reference_context(question["context"]),
                captions,
            )
            start_index, end_index = candidate_range(question, captions)
            aligned.append(
                {
                    "sequence": sequence,
                    "index": index,
                    "question": question,
                    "reference_ids": reference_ids,
                    "candidate_start": start_index,
                    "candidate_end": end_index,
                }
            )
    return aligned


def summarize_run(args, run, aligned, sequences=SEQUENCES):
    key, label, filename = run
    responses = []
    configs = []
    for sequence in sequences:
        result_path = args.result_root / str(sequence) / "human_qa" / filename
        result = load_json(result_path)
        sequence_responses = result.get("responses", [])
        if len(sequence_responses) != 30 or result.get("in_progress", True):
            raise ValueError(
                f"Incomplete {key} result for sequence {sequence}: "
                f"{len(sequence_responses)}/30, in_progress={result.get('in_progress')}"
            )
        responses.extend(sequence_responses)
        configs.append(portable_config(result.get("config", {})))
    if len(responses) != len(aligned):
        raise ValueError(f"Run {key} has {len(responses)} responses, expected 210")

    rows = []
    qrag_trace_errors = []
    for item, response in zip(aligned, responses):
        question = item["question"]
        correct, reason, metric_value, threshold, unit = classify_output(
            question, response
        )
        traces, trace_available = retrieval_calls_for_scoring(response)
        text_calls = [
            call
            for call in traces
            if call.get("tool") == "retrieve_from_text"
        ]
        selected_ids = {
            record.get("entry_id")
            for call in traces
            for record in call.get("selected", [])
        }
        selected_ids.discard(None)
        text_selected_ids = {
            record.get("entry_id")
            for call in text_calls
            for record in call.get("selected", [])
        }
        text_selected_ids.discard(None)
        references = set(item["reference_ids"])
        in_pool = {
            entry_id
            for entry_id in references
            if item["candidate_start"] <= entry_id <= item["candidate_end"]
        }
        hit = bool(selected_ids & references)
        complete_hit = bool(references) and references.issubset(selected_ids)
        text_hit = bool(text_selected_ids & references)
        if key in {"qrag_static", "qrag"}:
            expected_method = (
                "qrag_static_topk_zero_shot"
                if key == "qrag_static"
                else "qrag_sequential_zero_shot"
            )
            for call in text_calls:
                expected_steps = min(
                    int(call.get("requested_k", 5)),
                    int(call.get("candidate_count", 0)),
                )
                selected = [
                    record.get("entry_id")
                    for record in call.get("selected", [])
                ]
                invalid_steps = (
                    len(call.get("steps", [])) != expected_steps
                    if key == "qrag"
                    else bool(call.get("steps"))
                )
                if (
                    call.get("retrieval_method") != expected_method
                    or invalid_steps
                    or len(selected) != expected_steps
                    or len(selected) != len(set(selected))
                ):
                    qrag_trace_errors.append(
                        f"S{item['sequence']}Q{item['index']} call "
                        f"{call.get('call_index')}"
                    )
        rows.append(
            {
                "sequence": item["sequence"],
                "index": item["index"],
                "question_id": question["id"],
                "question": question["question"].splitlines()[-1].strip(),
                "type": question["type"],
                "ground_truth": ground_truth(question),
                "prediction": prediction_value(question["type"], response),
                "correct": bool(correct),
                "reason": reason,
                "metric_value": metric_value,
                "threshold": threshold,
                "metric_unit": unit,
                "failed": bool(response.get("evaluation_failure")),
                "elapsed": response.get("elapsed"),
                "trace_available": trace_available,
                "retrieval_call_count": len(traces),
                "text_call_count": len(text_calls),
                "retrieval_tools": [call.get("tool") for call in traces],
                "reference_ids": sorted(references),
                "reference_in_pool": sorted(in_pool),
                "reference_all_in_pool": bool(references) and references.issubset(in_pool),
                "selected_ids": sorted(selected_ids),
                "reference_hit": hit,
                "reference_complete_hit": complete_hit,
                "text_reference_hit": text_hit,
                "reference_hits": len(selected_ids & references),
                "grounded_any_correct": bool(correct and hit),
                "grounded_complete_correct": bool(correct and complete_hit),
                "ungrounded_correct": bool(correct and trace_available and not hit),
                "unverifiable_correct": bool(correct and not trace_available),
            }
        )

    if qrag_trace_errors:
        raise ValueError(
            "Invalid Q-RAG sequential traces: " + ", ".join(qrag_trace_errors[:10])
        )
    correct = sum(row["correct"] for row in rows)
    latencies = [
        row["elapsed"]
        for row in rows
        if isinstance(row["elapsed"], (int, float))
    ]
    retrieval_call_rows = [row for row in rows if row["retrieval_call_count"]]
    text_call_rows = [row for row in rows if row["text_call_count"]]
    trace_questions = sum(row["trace_available"] for row in rows)
    full_trace_coverage = trace_questions == len(rows)
    retrievable_text_rows = [
        row for row in text_call_rows if row["reference_in_pool"]
    ]
    traced_rows = [row for row in rows if row["trace_available"]]
    reference_total = sum(len(row["reference_ids"]) for row in traced_rows)
    reference_hits = sum(row["reference_hits"] for row in traced_rows)
    reference_hit_rows = [row for row in traced_rows if row["reference_hit"]]
    eligible_rows = [row for row in rows if row["reference_all_in_pool"]]
    grounded_any_correct = sum(row["grounded_any_correct"] for row in rows)
    grounded_complete_correct = sum(
        row["grounded_complete_correct"] for row in rows
    )

    def grouped(field):
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        return {
            group: {
                "correct": sum(row["correct"] for row in values),
                "grounded_any_correct": sum(
                    row["grounded_any_correct"] for row in values
                ),
                "grounded_complete_correct": sum(
                    row["grounded_complete_correct"] for row in values
                ),
                "reference_hit": sum(row["reference_hit"] for row in values),
                "reference_complete_hit": sum(
                    row["reference_complete_hit"] for row in values
                ),
                "trace_questions": sum(row["trace_available"] for row in values),
                "total": len(values),
            }
            for group, values in sorted(groups.items())
        }

    return {
        "key": key,
        "label": label,
        "filename": filename,
        "configs": configs,
        "rows": rows,
        "summary": {
            "correct": correct,
            "questions": len(rows),
            "accuracy": correct / len(rows),
            "failures": sum(row["failed"] for row in rows),
            "latency_mean": statistics.mean(latencies),
            "latency_median": statistics.median(latencies),
            "retrieval_call_questions": len(retrieval_call_rows),
            "retrieval_calls": sum(row["retrieval_call_count"] for row in rows),
            "text_call_questions": len(text_call_rows),
            "text_calls": sum(row["text_call_count"] for row in rows),
            "trace_questions": trace_questions,
            "full_trace_coverage": full_trace_coverage,
            "grounded_any_correct": grounded_any_correct,
            "grounded_any_accuracy": (
                grounded_any_correct / len(rows) if full_trace_coverage else None
            ),
            "grounded_complete_correct": grounded_complete_correct,
            "grounded_complete_accuracy": (
                grounded_complete_correct / len(rows)
                if full_trace_coverage
                else None
            ),
            "ungrounded_correct": sum(row["ungrounded_correct"] for row in rows),
            "unverifiable_correct": sum(
                row["unverifiable_correct"] for row in rows
            ),
            "eligible_questions": len(eligible_rows),
            "grounded_any_correct_eligible": sum(
                row["grounded_any_correct"] for row in eligible_rows
            ),
            "grounded_any_accuracy_eligible": (
                sum(row["grounded_any_correct"] for row in eligible_rows)
                / len(eligible_rows)
                if full_trace_coverage and eligible_rows
                else None
            ),
            "reference_hit_questions": sum(row["reference_hit"] for row in rows),
            "reference_hit_all": (
                sum(row["reference_hit"] for row in rows) / len(rows)
                if full_trace_coverage
                else None
            ),
            "reference_complete_hit_questions": sum(
                row["reference_complete_hit"] for row in rows
            ),
            "reference_hit_given_text_call": (
                sum(row["text_reference_hit"] for row in text_call_rows)
                / len(text_call_rows)
                if text_call_rows
                else None
            ),
            "reference_hit_given_text_call_and_retrievable": (
                sum(row["text_reference_hit"] for row in retrievable_text_rows)
                / len(retrievable_text_rows)
                if retrievable_text_rows
                else None
            ),
            "answer_correct_given_reference_hit": (
                sum(row["correct"] for row in reference_hit_rows)
                / len(reference_hit_rows)
                if reference_hit_rows
                else None
            ),
            "reference_recall": (
                reference_hits / reference_total
                if reference_total and full_trace_coverage
                else None
            ),
            "reference_hits": reference_hits,
            "reference_total": reference_total,
            "accuracy_by_sequence": grouped("sequence"),
            "accuracy_by_type": grouped("type"),
        },
    }


def paired_delta(candidate, baseline):
    improved = regressed = unchanged_correct = unchanged_wrong = 0
    for current, base in zip(candidate["rows"], baseline["rows"]):
        if current["correct"] and not base["correct"]:
            improved += 1
        elif base["correct"] and not current["correct"]:
            regressed += 1
        elif current["correct"]:
            unchanged_correct += 1
        else:
            unchanged_wrong += 1
    return {
        "improved": improved,
        "regressed": regressed,
        "net": improved - regressed,
        "unchanged_correct": unchanged_correct,
        "unchanged_wrong": unchanged_wrong,
    }


def method_cards(runs):
    cards = []
    colors = {
        "b0": "base",
        "gte": "gte",
        "qrag_static": "qrag-static",
        "qrag": "qrag",
    }
    for run in runs:
        summary = run["summary"]
        grounded_any = summary["grounded_any_accuracy"]
        grounded_complete = summary["grounded_complete_accuracy"]
        retrieval_hit = summary["reference_hit_all"]
        retrieval_display = (
            "不可验证" if retrieval_hit is None else f"{100 * retrieval_hit:.1f}%"
        )
        retrieval_count = (
            "旧结果未保存 retrieval trace"
            if retrieval_hit is None
            else f"{summary['reference_hit_questions']}/{summary['questions']} 取回至少一条 reference"
        )
        cards.append(
            f"""<article class="method {colors.get(run['key'], '')}">
            <h2>{esc(run['label'])}</h2>
            <strong>{retrieval_display}</strong>
            <p>{retrieval_count}</p>
            <dl><div><dt>Grounded 回答准确率</dt><dd>{'—' if grounded_any is None else f'{100*grounded_any:.1f}%'}</dd></div>
            <div><dt>完整 reference 检索</dt><dd>{'—' if retrieval_hit is None else pct(summary['reference_complete_hit_questions'], summary['questions'])}</dd></div>
            <div><dt>完整 support grounded</dt><dd>{'—' if grounded_complete is None else f'{100*grounded_complete:.1f}%'}</dd></div>
            <div><dt>原答案准确率</dt><dd>{pct(summary['correct'], summary['questions'])}</dd></div>
            <div><dt>答对但未命中</dt><dd>{summary['ungrounded_correct'] if grounded_any is not None else '—'}</dd></div>
            <div><dt>Trace 覆盖</dt><dd>{summary['trace_questions']}/{summary['questions']}</dd></div></dl></article>"""
        )
    return "".join(cards)


def group_table(runs, field, labels=None, metric="correct"):
    keys = sorted(
        {
            key
            for run in runs
            for key in run["summary"][field]
        },
        key=lambda value: int(value) if value.isdigit() else value,
    )
    rows = []
    for key in keys:
        cells = []
        for run in runs:
            item = run["summary"][field][key]
            if metric != "correct" and item["trace_questions"] != item["total"]:
                cells.append("<td><b>—</b><small>无完整 trace</small></td>")
            else:
                cells.append(
                    f"<td><b>{pct(item[metric], item['total'])}</b>"
                    f"<small>{item[metric]}/{item['total']}</small></td>"
                )
        rows.append(
            f"<tr><th>{esc(labels.get(key, key) if labels else key)}</th>"
            + "".join(cells)
            + "</tr>"
        )
    return "".join(rows)


def grounded_badge(row):
    if not row["trace_available"]:
        return "unverified", (
            "答对·trace 缺失" if row["correct"] else "答错·trace 缺失"
        )
    if row["grounded_any_correct"]:
        return "grounded", "Grounded 正确"
    if row["correct"]:
        return "ungrounded", "答对·未命中"
    if row["reference_hit"]:
        return "reader-fail", "命中·答错"
    return "bad", "未命中·答错"


def metric_display(row):
    value = row.get("metric_value")
    threshold = row.get("threshold")
    if isinstance(value, (int, float)) and isinstance(threshold, (int, float)):
        unit = f" {row['metric_unit']}" if row.get("metric_unit") else ""
        return f"error={value:.2f}{unit} ≤ {threshold:.2f}"
    return str(row.get("reason", ""))


def build_html(report):
    runs = report["runs"]
    headers = "".join(f"<th>{esc(run['label'])}</th>" for run in runs)
    pairs = "".join(
        f"<article><h3>{esc(item['candidate'])} vs {esc(item['baseline'])}</h3>"
        f"<b class='net'>{item['net']:+d}</b><p>改善 {item['improved']} · "
        f"退化 {item['regressed']} · 净变化 {item['net']:+d}</p></article>"
        for item in report["paired"]
    )
    question_rows = []
    for index, base_row in enumerate(runs[0]["rows"]):
        cells = []
        search_parts = [base_row["question"], base_row["question_id"]]
        for run in runs:
            row = run["rows"][index]
            badge_class, badge_label = grounded_badge(row)
            cells.append(
                f"<td><span class='pill {badge_class}'>{badge_label}</span><small>"
                f"prediction={esc(row['prediction'])}<br>"
                f"calls={row['retrieval_call_count']} · selected={esc(row['selected_ids'])}<br>"
                f"reference={esc(row['reference_ids'])}</small></td>"
            )
            search_parts.append(row["reason"])
        question_rows.append(
            f"<tr data-search='{esc(' '.join(search_parts).lower())}' "
            f"data-type='{base_row['type']}'><th>S{base_row['sequence']}·Q{base_row['index']}</th>"
            f"<td class='question'><b>{esc(base_row['question'])}</b>"
            f"<small>{esc(base_row['type'])} · {esc(base_row['question_id'])}<br>"
            f"ground truth={esc(base_row['ground_truth'])}</small></td>"
            + "".join(cells)
            + "</tr>"
        )
    diagnostics = report.get("diagnostics", [])
    diagnostic_cards = method_cards(diagnostics) if diagnostics else ""
    diagnostic_rows = []
    for run in diagnostics:
        for row in run["rows"]:
            badge_class, badge_label = grounded_badge(row)
            diagnostic_rows.append(
                f"<tr><th>{esc(run['label'])}</th><td>S{row['sequence']}·Q{row['index']}</td>"
                f"<td class='question'><b>{esc(row['question'])}</b></td>"
                f"<td><span class='pill {badge_class}'>{badge_label}</span></td>"
                f"<td>{esc(row['reference_ids'])}</td><td>{esc(row['selected_ids'])}</td>"
                f"<td>{esc(row['retrieval_tools'])}</td></tr>"
            )
    diagnostic_section = ""
    example_section = ""
    if diagnostics:
        diagnostic_section = f"""<section class="panel"><h2>B0 trace 补充诊断（非 210 题主结果）</h2>
<p class="muted">原 B0 210 题文件没有保存 retrieval trace，因此不能诚实计算其 210 题 grounded accuracy。这里单列后来重跑且保存完整 trace 的 sequence 0（30 题），不与原 B0 115/210 混合。</p>
<div class="methods">{diagnostic_cards}</div><div class="table-wrap"><table><thead><tr><th>运行</th><th>题号</th><th>问题</th><th>分类</th><th>Reference IDs</th><th>最终成功 attempt 的 IDs</th><th>工具</th></tr></thead><tbody>{''.join(diagnostic_rows)}</tbody></table></div></section>"""
        example = diagnostics[0]["rows"][0]
        example_section = f"""<section class="panel example"><h2>用户指出的 B0 S0·Q0</h2>
<p><b>{esc(example['question'])}</b></p><div class="example-grid"><div><span>标准答案</span><strong>{esc(example['ground_truth'])}</strong></div><div><span>模型答案</span><strong>{esc(example['prediction'])}</strong></div><div><span>原答案判分</span><strong>{'正确' if example['correct'] else '错误'} · {esc(metric_display(example))}</strong></div><div><span>Derived reference</span><strong>{esc(example['reference_ids'])}</strong></div><div><span>实际返回 IDs</span><strong>{esc(example['selected_ids'])}</strong></div><div><span>新指标</span><strong>{'检索命中且 Grounded 正确' if example['grounded_any_correct'] else '检索未命中，不计 Grounded 正确'}</strong></div></div>
<p class="muted">答案即使落入 NaVQA 数值容差，只要最终成功 attempt 没有取回 reference memory，就归入“答对·未命中”，grounded accuracy 记 0。</p></section>"""
    generated = esc(report["generated_at"])
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>NaVQA 210 · Retrieval-hit Accuracy</title>
<style>
:root{{--bg:#07111e;--panel:#112038;--line:#2b4161;--text:#eef5ff;--muted:#9db0c8;--blue:#53a9ff;--green:#3bdda0;--red:#ff7180;--orange:#ffb454;--purple:#ba8cff}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 10% 0,#194774 0,transparent 36rem),var(--bg);color:var(--text);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}}.wrap{{width:min(1500px,calc(100% - 32px));margin:auto;padding:38px 0 70px}}h1{{font-size:clamp(32px,5vw,55px);margin:5px 0;line-height:1.05}}h2{{margin:0 0 8px}}.muted,small{{display:block;color:var(--muted)}}.eyebrow{{color:#55dbe2;font-weight:900;letter-spacing:.14em}}.methods{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:15px;margin:24px 0}}.method,.panel,.paired article{{background:linear-gradient(145deg,#172a46,#0c192b);border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 16px 38px #0004}}.method>strong{{font-size:42px}}.method.base{{border-top:4px solid #97a5b7}}.method.gte{{border-top:4px solid var(--blue)}}.method.qrag-static{{border-top:4px solid var(--orange)}}.method.qrag{{border-top:4px solid var(--purple)}}dl{{display:grid;grid-template-columns:1fr 1fr;gap:9px}}dl div{{background:#091627;border-radius:9px;padding:9px}}dt{{color:var(--muted);font-size:11px}}dd{{margin:2px 0;font-weight:800}}.panel{{margin:16px 0}}.example{{border-color:#ffb454}}.example-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0}}.example-grid div{{background:#091627;border-radius:10px;padding:12px}}.example-grid span{{display:block;color:var(--muted);font-size:11px}}.example-grid strong{{display:block;margin-top:4px;color:#ffd28e}}.formula{{font:700 16px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:#b8efff}}.paired{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.paired article{{padding:14px}}.net{{font-size:30px;color:var(--green)}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:12px}}table{{border-collapse:collapse;width:100%;min-width:900px}}th,td{{text-align:left;padding:11px;border-bottom:1px solid #263b58;vertical-align:top}}thead th{{background:#192d49;position:sticky;top:0}}td b{{font-size:16px}}input,select{{background:#091627;color:var(--text);border:1px solid var(--line);padding:10px 12px;border-radius:9px}}.toolbar{{display:flex;gap:9px;margin-bottom:12px}}input{{flex:1}}.question{{min-width:360px}}.pill{{display:inline-block;padding:3px 9px;border-radius:99px;font-size:11px;font-weight:800;white-space:nowrap}}.grounded{{background:#174b39;color:#79edbc}}.ungrounded{{background:#533a17;color:#ffd28e}}.reader-fail{{background:#3f2a5f;color:#d4bdff}}.unverified{{background:#303b4b;color:#cbd5e3}}.bad{{background:#55232e;color:#ff9eaa}}@media(max-width:900px){{.methods,.paired,.example-grid{{grid-template-columns:1fr}}}}
</style></head><body><main class="wrap"><div class="eyebrow">ReMEmbR · evidence-grounded retrieval audit</div><h1>NaVQA 210 题<br>Retrieval-hit Accuracy</h1><p class="muted">B0 mxbai dense、B1 GTE base dense、B2 Q-RAG static top-k、B3 Q-RAG sequential。生成于 {generated}</p>
<section class="panel"><h2>新指标定义</h2><p class="formula">RetrievalHit(q) = RetrievedIDs(q) ∩ DerivedReferenceIDs(q) ≠ ∅</p><p class="formula">GroundedCorrect(q) = AnswerCorrect(q) ∧ RetrievalHit(q)</p><p>卡片大数字是“检索到正确记忆”的命中率，以全部题目为分母；Grounded 回答准确率进一步要求最终答案也正确。“完整 reference 检索”要求所有 derived reference IDs 都被取回。RetrievedIDs 联合 text、time、position 三类工具；成功时只取最后成功 attempt，整题失败时只取最后一次失败 attempt，不合并旧重试。</p></section>
{example_section}
<section class="methods">{method_cards(runs)}</section>
<section class="panel"><h2>按序列检索命中率（Any-hit）</h2><div class="table-wrap"><table><thead><tr><th>序列</th>{headers}</tr></thead><tbody>{group_table(runs,'accuracy_by_sequence',metric='reference_hit')}</tbody></table></div></section>
<section class="panel"><h2>按题型检索命中率（Any-hit）</h2><div class="table-wrap"><table><thead><tr><th>题型</th>{headers}</tr></thead><tbody>{group_table(runs,'accuracy_by_type',TYPE_LABELS,metric='reference_hit')}</tbody></table></div></section>
<section class="panel"><h2>按序列 Grounded Accuracy（Any-hit）</h2><div class="table-wrap"><table><thead><tr><th>序列</th>{headers}</tr></thead><tbody>{group_table(runs,'accuracy_by_sequence',metric='grounded_any_correct')}</tbody></table></div></section>
<section class="panel"><h2>按题型 Grounded Accuracy（Any-hit）</h2><div class="table-wrap"><table><thead><tr><th>题型</th>{headers}</tr></thead><tbody>{group_table(runs,'accuracy_by_type',TYPE_LABELS,metric='grounded_any_correct')}</tbody></table></div></section>
{diagnostic_section}
<section class="panel"><h2>原答案准确率的配对变化</h2><div class="paired">{pairs}</div></section>
<section class="panel"><h2>按序列严格准确率</h2><div class="table-wrap"><table><thead><tr><th>序列</th>{headers}</tr></thead><tbody>{group_table(runs,'accuracy_by_sequence')}</tbody></table></div></section>
<section class="panel"><h2>按题型严格准确率</h2><div class="table-wrap"><table><thead><tr><th>题型</th>{headers}</tr></thead><tbody>{group_table(runs,'accuracy_by_type',TYPE_LABELS)}</tbody></table></div></section>
<section class="panel"><h2>逐题配对结果</h2><div class="toolbar"><input id="search" placeholder="搜索问题、ID 或错误原因"><select id="type"><option value="all">全部题型</option>{''.join(f'<option value="{key}">{label}</option>' for key,label in TYPE_LABELS.items())}</select><span id="visible" class="muted"></span></div><div class="table-wrap"><table><thead><tr><th>题号</th><th>问题</th>{headers}</tr></thead><tbody id="rows">{''.join(question_rows)}</tbody></table></div></section>
<p class="muted">边界：Derived reference 来自 NaVQA 数据内 context 到 caption entry 的确定性映射，不保证是穷尽式 gold support；存在等价但未被 context 标注的证据时，本指标会偏保守。19 题存在至少一条 reference 不在冻结候选池，主指标仍保留在 210 分母中。</p>
</main><script>const q=document.getElementById('search'),t=document.getElementById('type'),rows=[...document.querySelectorAll('#rows tr')],v=document.getElementById('visible');function filter(){{let n=0;rows.forEach(r=>{{const ok=(!q.value||r.dataset.search.includes(q.value.toLowerCase()))&&(t.value==='all'||r.dataset.type===t.value);r.hidden=!ok;if(ok)n++}});v.textContent=`显示 ${{n}} / ${{rows.length}}`}}q.oninput=filter;t.oninput=filter;filter();</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--questions-root", type=Path, default=Path("artifacts/questions"))
    parser.add_argument("--captions-root", type=Path, default=Path("artifacts/captions"))
    parser.add_argument("--caption-file", default="captions_VILA1.5-13b_3_secs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        required=True,
        help="Repeat key|label|result filename in display order",
    )
    parser.add_argument(
        "--diagnostic-run",
        action="append",
        type=parse_diagnostic_run,
        default=[],
        help=(
            "Optional trace-bearing subset as "
            "key|label|result filename|comma-separated sequences"
        ),
    )
    args = parser.parse_args()
    aligned = load_aligned_inputs(args)
    runs = [summarize_run(args, run, aligned) for run in args.run]
    diagnostics = []
    for key, label, filename, sequences in args.diagnostic_run:
        diagnostic_aligned = load_aligned_inputs(args, sequences)
        diagnostics.append(
            summarize_run(
                args,
                (key, label, filename),
                diagnostic_aligned,
                sequences,
            )
        )
    paired = []
    for candidate_index in range(1, len(runs)):
        for baseline_index in range(candidate_index):
            candidate = runs[candidate_index]
            pair_baseline = runs[baseline_index]
            delta = paired_delta(candidate, pair_baseline)
            delta.update(
                {
                    "candidate": candidate["label"],
                    "baseline": pair_baseline["label"],
                }
            )
            paired.append(delta)
    report = {
        "version": 2,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "metric": {
            "primary_name": "retrieval-hit accuracy",
            "primary_definition": (
                "any retrieved entry matches a derived reference entry"
            ),
            "grounded_answer_name": "retrieval-grounded answer accuracy",
            "grounded_answer_definition": (
                "answer_correct AND retrieval_hit"
            ),
            "retrieval_scope": (
                "union of text, time, and position calls from the last "
                "successful attempt, or the final failed attempt if none "
                "succeeded"
            ),
            "denominator": "all evaluated questions",
        },
        "runs": runs,
        "diagnostics": diagnostics,
        "paired": paired,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "index.html").write_text(
        build_html(report),
        encoding="utf-8",
    )
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
