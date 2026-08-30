#!/usr/bin/env python3
"""Build a self-contained B0/B1/B2/B3 comparison for all 210 NaVQA items."""

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
    map_reference_entries,
    parse_reference_context,
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


def parse_run(value: str) -> tuple[str, str, str]:
    parts = value.split("|", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--run must be key|display label|result filename"
        )
    return parts[0], parts[1], parts[2]


def load_aligned_inputs(args):
    aligned = []
    for sequence in SEQUENCES:
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


def summarize_run(args, run, aligned):
    key, label, filename = run
    responses = []
    configs = []
    for sequence in SEQUENCES:
        result_path = args.result_root / str(sequence) / "human_qa" / filename
        result = load_json(result_path)
        sequence_responses = result.get("responses", [])
        if len(sequence_responses) != 30 or result.get("in_progress", True):
            raise ValueError(
                f"Incomplete {key} result for sequence {sequence}: "
                f"{len(sequence_responses)}/30, in_progress={result.get('in_progress')}"
            )
        responses.extend(sequence_responses)
        configs.append(result.get("config", {}))
    if len(responses) != len(aligned):
        raise ValueError(f"Run {key} has {len(responses)} responses, expected 210")

    rows = []
    qrag_trace_errors = []
    for item, response in zip(aligned, responses):
        question = item["question"]
        correct, reason, _, _, _ = classify_output(question, response)
        traces = response.get("retrieval_trace") or []
        text_calls = [
            call
            for call in traces
            if call.get("tool") == "retrieve_from_text"
        ]
        selected_ids = {
            record.get("entry_id")
            for call in text_calls
            for record in call.get("selected", [])
        }
        selected_ids.discard(None)
        references = set(item["reference_ids"])
        in_pool = {
            entry_id
            for entry_id in references
            if item["candidate_start"] <= entry_id <= item["candidate_end"]
        }
        hit = bool(selected_ids & references)
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
                "correct": bool(correct),
                "reason": reason,
                "failed": bool(response.get("evaluation_failure")),
                "elapsed": response.get("elapsed"),
                "text_call_count": len(text_calls),
                "reference_ids": sorted(references),
                "reference_in_pool": sorted(in_pool),
                "selected_ids": sorted(selected_ids),
                "reference_hit": hit,
                "reference_hits": len(selected_ids & references),
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
    text_call_rows = [row for row in rows if row["text_call_count"]]
    trace_questions = sum(bool(response.get("retrieval_trace")) for response in responses)
    retrievable_text_rows = [
        row for row in text_call_rows if row["reference_in_pool"]
    ]
    reference_total = sum(len(row["reference_ids"]) for row in rows)
    reference_hits = sum(row["reference_hits"] for row in rows)

    def grouped(field):
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        return {
            group: {
                "correct": sum(row["correct"] for row in values),
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
            "text_call_questions": len(text_call_rows),
            "text_calls": sum(row["text_call_count"] for row in rows),
            "trace_questions": trace_questions,
            "reference_hit_questions": sum(row["reference_hit"] for row in rows),
            "reference_hit_all": (
                sum(row["reference_hit"] for row in rows) / len(rows)
                if trace_questions
                else None
            ),
            "reference_hit_given_text_call": (
                sum(row["reference_hit"] for row in text_call_rows)
                / len(text_call_rows)
                if text_call_rows
                else None
            ),
            "reference_hit_given_text_call_and_retrievable": (
                sum(row["reference_hit"] for row in retrievable_text_rows)
                / len(retrievable_text_rows)
                if retrievable_text_rows
                else None
            ),
            "reference_recall": (
                reference_hits / reference_total
                if reference_total and trace_questions
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
        conditional = summary["reference_hit_given_text_call_and_retrievable"]
        text_call_display = (
            summary["text_call_questions"]
            if summary["trace_questions"]
            else "无 trace"
        )
        cards.append(
            f"""<article class="method {colors.get(run['key'], '')}">
            <h2>{esc(run['label'])}</h2>
            <strong>{pct(summary['correct'], summary['questions'])}</strong>
            <p>{summary['correct']}/{summary['questions']} 严格正确 · {summary['failures']} 个失效</p>
            <dl><div><dt>平均耗时</dt><dd>{summary['latency_mean']:.1f}s</dd></div>
            <div><dt>Text-tool 题</dt><dd>{text_call_display}</dd></div>
            <div><dt>可检索条件 Hit</dt><dd>{'—' if conditional is None else f'{100*conditional:.1f}%'}</dd></div>
            <div><dt>Derived ref recall</dt><dd>{'—' if summary['reference_recall'] is None else f"{100*summary['reference_recall']:.1f}%"}</dd></div></dl></article>"""
        )
    return "".join(cards)


def group_table(runs, field, labels=None):
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
            cells.append(
                f"<td><b>{pct(item['correct'], item['total'])}</b>"
                f"<small>{item['correct']}/{item['total']}</small></td>"
            )
        rows.append(
            f"<tr><th>{esc(labels.get(key, key) if labels else key)}</th>"
            + "".join(cells)
            + "</tr>"
        )
    return "".join(rows)


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
            hit = " · ref hit" if row["reference_hit"] else ""
            cells.append(
                f"<td><span class='pill {'ok' if row['correct'] else 'bad'}'>"
                f"{'正确' if row['correct'] else '错误'}</span><small>"
                f"text calls={row['text_call_count']}{hit}</small></td>"
            )
            search_parts.append(row["reason"])
        question_rows.append(
            f"<tr data-search='{esc(' '.join(search_parts).lower())}' "
            f"data-type='{base_row['type']}'><th>S{base_row['sequence']}·Q{base_row['index']}</th>"
            f"<td class='question'><b>{esc(base_row['question'])}</b>"
            f"<small>{esc(base_row['type'])} · {esc(base_row['question_id'])}</small></td>"
            + "".join(cells)
            + "</tr>"
        )
    generated = esc(report["generated_at"])
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>NaVQA 210 · Retriever 对照</title>
<style>
:root{{--bg:#07111e;--panel:#112038;--line:#2b4161;--text:#eef5ff;--muted:#9db0c8;--blue:#53a9ff;--green:#3bdda0;--red:#ff7180;--orange:#ffb454;--purple:#ba8cff}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 10% 0,#194774 0,transparent 36rem),var(--bg);color:var(--text);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}}.wrap{{width:min(1500px,calc(100% - 32px));margin:auto;padding:38px 0 70px}}h1{{font-size:clamp(32px,5vw,55px);margin:5px 0;line-height:1.05}}h2{{margin:0 0 8px}}.muted,small{{display:block;color:var(--muted)}}.eyebrow{{color:#55dbe2;font-weight:900;letter-spacing:.14em}}.methods{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:15px;margin:24px 0}}.method,.panel,.paired article{{background:linear-gradient(145deg,#172a46,#0c192b);border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 16px 38px #0004}}.method>strong{{font-size:42px}}.method.base{{border-top:4px solid #97a5b7}}.method.gte{{border-top:4px solid var(--blue)}}.method.qrag-static{{border-top:4px solid var(--orange)}}.method.qrag{{border-top:4px solid var(--purple)}}dl{{display:grid;grid-template-columns:1fr 1fr;gap:9px}}dl div{{background:#091627;border-radius:9px;padding:9px}}dt{{color:var(--muted);font-size:11px}}dd{{margin:2px 0;font-weight:800}}.panel{{margin:16px 0}}.paired{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.paired article{{padding:14px}}.net{{font-size:30px;color:var(--green)}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:12px}}table{{border-collapse:collapse;width:100%;min-width:900px}}th,td{{text-align:left;padding:11px;border-bottom:1px solid #263b58;vertical-align:top}}thead th{{background:#192d49;position:sticky;top:0}}td b{{font-size:16px}}input,select{{background:#091627;color:var(--text);border:1px solid var(--line);padding:10px 12px;border-radius:9px}}.toolbar{{display:flex;gap:9px;margin-bottom:12px}}input{{flex:1}}.question{{min-width:360px}}.pill{{display:inline-block;padding:3px 9px;border-radius:99px;font-size:11px;font-weight:800}}.ok{{background:#174b39;color:#79edbc}}.bad{{background:#55232e;color:#ff9eaa}}@media(max-width:900px){{.methods,.paired{{grid-template-columns:1fr}}}}
</style></head><body><main class="wrap"><div class="eyebrow">ReMEmbR · paired retrieval ablation</div><h1>NaVQA 210 题 Retriever 对照</h1><p class="muted">B0 mxbai dense、B1 GTE base dense、B2 Q-RAG static top-k、B3 Q-RAG sequential；reader、候选池、每次证据预算和 evaluator 固定。生成于 {generated}</p>
<section class="methods">{method_cards(runs)}</section>
<section class="panel"><h2>配对变化</h2><div class="paired">{pairs}</div></section>
<section class="panel"><h2>按序列严格准确率</h2><div class="table-wrap"><table><thead><tr><th>序列</th>{headers}</tr></thead><tbody>{group_table(runs,'accuracy_by_sequence')}</tbody></table></div></section>
<section class="panel"><h2>按题型严格准确率</h2><div class="table-wrap"><table><thead><tr><th>题型</th>{headers}</tr></thead><tbody>{group_table(runs,'accuracy_by_type',TYPE_LABELS)}</tbody></table></div></section>
<section class="panel"><h2>逐题配对结果</h2><div class="toolbar"><input id="search" placeholder="搜索问题、ID 或错误原因"><select id="type"><option value="all">全部题型</option>{''.join(f'<option value="{key}">{label}</option>' for key,label in TYPE_LABELS.items())}</select><span id="visible" class="muted"></span></div><div class="table-wrap"><table><thead><tr><th>题号</th><th>问题</th>{headers}</tr></thead><tbody id="rows">{''.join(question_rows)}</tbody></table></div></section>
<p class="muted">Derived reference 来自 NaVQA context 的 caption 映射，不等同于穷尽式 gold support；条件 Hit 仅在 agent 调用 text tool 且至少一条 derived reference 位于固定候选池内时统计。B2 trace 已校验为单次 state 编码的静态 top-k；B3 trace 已校验为固定步数、无重复 action。</p>
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
    args = parser.parse_args()
    aligned = load_aligned_inputs(args)
    runs = [summarize_run(args, run, aligned) for run in args.run]
    baseline = runs[0]
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
        "version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "runs": runs,
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
