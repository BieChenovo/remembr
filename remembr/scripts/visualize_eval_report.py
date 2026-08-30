#!/usr/bin/env python3
"""Build a self-contained HTML report for a ReMEmbR NaVQA evaluation.

The report deliberately has no CDN or JavaScript package dependencies, so it
can be opened directly from a shared filesystem after a compute node stops.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


STATUS_COLORS = {
    "scored": "#32d583",
    "failed": "#ff6b6b",
    "skipped": "#98a2b3",
}

TYPE_LABELS = {
    "binary": "二元判断",
    "position": "位置",
    "time": "时间点",
    "duration": "持续时间",
    "text": "文本",
}

STATUS_LABELS = {
    "scored": "已计分",
    "failed": "失败",
    "skipped": "未计分",
}


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def number(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return esc(value)
    if not math.isfinite(value):
        return "—"
    return f"{value:.{digits}f}{suffix}"


def status_for(response: dict[str, Any]) -> str:
    if not response:
        return "skipped"
    if response.get("evaluation_failure"):
        return "failed"
    return "scored"


def ground_truth(question: dict[str, Any]) -> Any:
    answers = question.get("answers", {})
    question_type = question.get("type")
    if question_type in answers:
        return answers[question_type]
    text = answers.get("text")
    if isinstance(text, list):
        return text[0] if text else None
    return text


def prediction_value(question: dict[str, Any], response: dict[str, Any]) -> Any:
    if not response:
        return None
    question_type = question.get("type")
    value = response.get(question_type)
    if value is not None:
        return value
    nested = response.get("response")
    if isinstance(nested, dict):
        return nested.get(question_type)
    return None


def error_value(question: dict[str, Any], response: dict[str, Any]) -> tuple[Any, str]:
    error = response.get("error") or {}
    question_type = question.get("type")
    if question_type == "binary":
        correct = error.get("binary_iscorrect")
        if correct is None:
            return None, ""
        return correct, "正确" if bool(correct) else "错误"
    if question_type == "text":
        correct = error.get("text_iscorrect")
        if correct is None:
            return None, ""
        return correct, "语义正确" if bool(correct) else "语义错误"
    key_and_unit = {
        "position": ("position_error", "m"),
        "time": ("time_error", "min"),
        "duration": ("duration_error", "min"),
    }
    if question_type not in key_and_unit:
        return None, ""
    key, unit = key_and_unit[question_type]
    value = error.get(key)
    return value, number(value, 2, f" {unit}") if value is not None else ""


def format_structured(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (list, dict)):
        return esc(json.dumps(value, ensure_ascii=False))
    if isinstance(value, float):
        return number(value, 3)
    return esc(value)


def horizontal_bars(
    rows: Iterable[tuple[str, float]], unit: str, color: str
) -> str:
    rows = list(rows)
    if not rows:
        return '<p class="muted">没有可绘制的数据。</p>'
    max_value = max(value for _, value in rows) or 1
    rendered = []
    for label, value in rows:
        width = max(1.5, value / max_value * 100)
        rendered.append(
            f"""
            <div class="bar-row">
              <span class="bar-label">{esc(label)}</span>
              <div class="bar-track"><div class="bar-fill" style="width:{width:.2f}%;background:{color}"></div></div>
              <strong>{number(value, 2, f' {unit}')}</strong>
            </div>"""
        )
    return "".join(rendered)


def metric_card(label: str, value: str, note: str, tone: str = "blue") -> str:
    return f"""
    <article class="metric-card {tone}">
      <span>{esc(label)}</span>
      <strong>{value}</strong>
      <small>{esc(note)}</small>
    </article>"""


def build_report(
    result: dict[str, Any],
    questions: list[dict[str, Any]],
    result_path: Path,
    questions_path: Path,
) -> str:
    responses = result.get("responses", [])
    if len(responses) != len(questions):
        raise ValueError(
            f"Question/response count mismatch: {len(questions)} questions, "
            f"{len(responses)} responses"
        )

    rows: list[dict[str, Any]] = []
    type_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    errors: dict[str, list[tuple[str, float]]] = defaultdict(list)
    latencies: list[tuple[str, float]] = []
    failure_types: Counter[str] = Counter()

    for index, (question, response) in enumerate(zip(questions, responses), start=1):
        response = response or {}
        status = status_for(response)
        question_type = question.get("type", "unknown")
        value, error_text = error_value(question, response)
        type_outcomes[question_type][status] += 1
        if status == "failed":
            failure_types[question_type] += 1
        if status == "scored" and question_type in {"position", "time", "duration"}:
            if isinstance(value, (int, float)):
                errors[question_type].append((f"Q{index}", float(value)))
        elapsed = response.get("elapsed")
        if isinstance(elapsed, (int, float)) and elapsed > 0:
            latencies.append((f"Q{index}", float(elapsed)))
        rows.append(
            {
                "index": index,
                "question": question,
                "response": response,
                "status": status,
                "type": question_type,
                "error_text": error_text,
            }
        )

    metrics = result.get("metrics", {})
    status_counts = Counter(row["status"] for row in rows)
    total = len(rows)
    scored = status_counts["scored"]
    failed = status_counts["failed"]
    skipped = status_counts["skipped"]
    binary_count = int(metrics.get("binary_count") or 0)
    binary_accuracy = metrics.get("binary_accuracy")
    text_count = int(metrics.get("text_count") or 0)
    text_accuracy = metrics.get("text_accuracy")
    descriptive_count = int(metrics.get("descriptive_count") or 0)
    descriptive_accuracy = metrics.get("descriptive_accuracy")

    status_cursor = 0.0
    conic_parts = []
    for status in ("scored", "failed", "skipped"):
        start = status_cursor
        status_cursor += status_counts[status] / total * 100 if total else 0
        conic_parts.append(
            f"{STATUS_COLORS[status]} {start:.3f}% {status_cursor:.3f}%"
        )
    conic = ",".join(conic_parts)

    type_order = ["position", "time", "duration", "binary", "text"]
    type_chart = []
    for question_type in type_order:
        counts = type_outcomes.get(question_type, Counter())
        count_total = sum(counts.values())
        if not count_total:
            continue
        segments = []
        for status in ("scored", "failed", "skipped"):
            count = counts[status]
            if not count:
                continue
            width = count / count_total * 100
            segments.append(
                f'<span title="{STATUS_LABELS[status]} {count}" '
                f'style="width:{width:.3f}%;background:{STATUS_COLORS[status]}">{count}</span>'
            )
        type_chart.append(
            f"""
            <div class="type-row">
              <span>{esc(TYPE_LABELS.get(question_type, question_type))}</span>
              <div class="stacked-bar">{''.join(segments)}</div>
              <strong>{count_total}</strong>
            </div>"""
        )

    latency_values = [value for _, value in latencies]
    latency_mean = statistics.mean(latency_values) if latency_values else None
    latency_max = max(latency_values) if latency_values else None
    position_values = [value for _, value in errors["position"]]
    position_median = statistics.median(position_values) if position_values else None

    cards = "".join(
        [
            metric_card("完成情况", f"{scored}/{total}", "成功计分题目", "green"),
            metric_card("结构化失败", str(failed), "超过重试上限", "red"),
            metric_card("描述题准确率", number(descriptive_accuracy * 100 if descriptive_accuracy is not None else None, 1, "%"), f"binary + text, n={descriptive_count}", "blue"),
            metric_card("二元准确率", number(binary_accuracy * 100 if binary_accuracy is not None else None, 1, "%"), f"n={binary_count}", "blue"),
            metric_card("文本准确率", number(text_accuracy * 100 if text_accuracy is not None else None, 1, "%"), f"LLM 语义判分, n={text_count}", "purple"),
            metric_card("位置 L2 误差", number(metrics.get("position_mean_l2_error"), 2, " m"), f"n={metrics.get('position_count', 0)}", "orange"),
            metric_card("时间 MAE", number(metrics.get("time_mean_absolute_error"), 2, " min"), f"n={metrics.get('time_count', 0)}", "purple"),
            metric_card("持续时间 MAE", number(metrics.get("duration_mean_absolute_error"), 2, " min"), f"n={metrics.get('duration_count', 0)}", "cyan"),
        ]
    )

    failure_summary = "、".join(
        f"{TYPE_LABELS.get(key, key)} {value} 题"
        for key, value in failure_types.most_common()
    ) or "无失败"
    findings = [
        f"共有 {failed} 个失败：{failure_summary}。失败不会被计入相应题型的分母，阅读准确率时应同时检查计分数量。",
        f"描述题准确率为 {number(descriptive_accuracy * 100 if descriptive_accuracy is not None else None, 1, '%')}（二元 {binary_count} 题 + 文本 {text_count} 题）；文本题采用固定本地 LLM 语义判分，不与论文的判分器严格等价。",
        f"位置误差中位数为 {number(position_median, 2, ' m')}，均值为 {number(metrics.get('position_mean_l2_error'), 2, ' m')}，需要进一步拆分检索误差与答案生成误差。",
        f"有效模型调用平均耗时 {number(latency_mean, 1, ' s')}，最慢题目 {number(latency_max, 1, ' s')}；带重试题显著拉高总耗时。",
        f"二元题准确率为 {number(binary_accuracy * 100 if binary_accuracy is not None else None, 1, '%')}；文本题准确率为 {number(text_accuracy * 100 if text_accuracy is not None else None, 1, '%')}。",
    ]

    table_rows = []
    for row in rows:
        index = row["index"]
        question = row["question"]
        response = row["response"]
        status = row["status"]
        question_type = row["type"]
        question_text = question.get("question", "")
        answer_text = response.get("text") or ""
        failure = response.get("evaluation_failure") or ""
        judge = (response.get("error") or {}).get("text_judge") or {}
        judge_rationale = judge.get("rationale", "")
        gt = ground_truth(question)
        prediction = prediction_value(question, response)
        elapsed = response.get("elapsed")
        searchable = " ".join(
            [question_text, answer_text, failure, judge_rationale, question.get("id", ""), question_type]
        ).lower()
        if status == "skipped":
            prediction_html = '<span class="muted">未运行或没有可计分输出</span>'
        elif status == "failed":
            prediction_html = (
                f'<span class="failure-reason">{esc(failure)}</span>'
                + (f'<div class="answer-text">{esc(answer_text)}</div>' if answer_text else "")
            )
        else:
            prediction_html = (
                f'<strong>{format_structured(prediction)}</strong>'
                + (f'<div class="answer-text">{esc(answer_text)}</div>' if answer_text else "")
                + (f'<div class="answer-text">判分理由：{esc(judge_rationale)}</div>' if judge_rationale else "")
            )
        table_rows.append(
            f"""
            <tr data-status="{status}" data-type="{esc(question_type)}" data-search="{esc(searchable)}">
              <td><strong>Q{index}</strong><small>{esc(question.get('length_category', ''))}</small></td>
              <td><span class="type-pill type-{esc(question_type)}">{esc(TYPE_LABELS.get(question_type, question_type))}</span></td>
              <td><span class="status-pill status-{status}">{STATUS_LABELS[status]}</span></td>
              <td class="question-cell">
                <details>
                  <summary>{esc(question_text)}</summary>
                  <div class="detail-grid">
                    <div><span>题目 ID</span><code>{esc(question.get('id', ''))}</code></div>
                    <div><span>标准答案</span><strong>{format_structured(gt)}</strong></div>
                    <div><span>轨迹长度</span><strong>{number(question.get('length'), 1, ' s')}</strong></div>
                    <div><span>类别</span><strong>{esc(question.get('category', '—'))}</strong></div>
                  </div>
                </details>
              </td>
              <td>{prediction_html}</td>
              <td><strong>{esc(row['error_text']) if row['error_text'] else '—'}</strong></td>
              <td>{number(elapsed, 1, ' s')}</td>
            </tr>"""
        )

    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    sequence_id = questions_path.parent.name
    answer_model = result.get("config", {}).get("answer_model", "ReMEmbR")
    title = f"{answer_model} · NaVQA 序列 {sequence_id}"

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg:#08111f; --panel:#101c2e; --panel2:#142238; --line:#273750;
      --text:#edf4ff; --muted:#9eacc2; --blue:#53a7ff; --green:#32d583;
      --red:#ff6b6b; --orange:#ffae57; --purple:#ad8cff; --cyan:#3ddbd9;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:radial-gradient(circle at 12% 0,#15345a 0,transparent 28rem),var(--bg); color:var(--text); font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif; }}
    .container {{ width:min(1440px,calc(100% - 40px)); margin:auto; padding:44px 0 72px; }}
    header {{ display:flex; justify-content:space-between; gap:24px; align-items:flex-end; margin-bottom:28px; }}
    .eyebrow {{ color:var(--blue); text-transform:uppercase; letter-spacing:.16em; font-weight:800; font-size:12px; }}
    h1 {{ margin:6px 0 5px; font-size:clamp(28px,4vw,48px); line-height:1.1; letter-spacing:-.04em; }}
    h2 {{ margin:0 0 18px; font-size:20px; }}
    h3 {{ margin:0 0 14px; font-size:15px; }}
    p {{ margin:0; }}
    .muted, small {{ color:var(--muted); }}
    .source {{ text-align:right; color:var(--muted); font-size:12px; max-width:440px; overflow-wrap:anywhere; }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:18px; }}
    .metric-card,.panel {{ border:1px solid var(--line); background:linear-gradient(145deg,rgba(20,34,56,.96),rgba(12,24,41,.96)); box-shadow:0 18px 50px rgba(0,0,0,.17); border-radius:16px; }}
    .metric-card {{ padding:18px; position:relative; overflow:hidden; }}
    .metric-card:before {{ content:""; position:absolute; inset:0 auto 0 0; width:3px; background:var(--blue); }}
    .metric-card.green:before {{ background:var(--green); }} .metric-card.red:before {{ background:var(--red); }}
    .metric-card.orange:before {{ background:var(--orange); }} .metric-card.purple:before {{ background:var(--purple); }} .metric-card.cyan:before {{ background:var(--cyan); }}
    .metric-card span {{ display:block; color:var(--muted); font-size:12px; }}
    .metric-card strong {{ display:block; font-size:26px; margin:7px 0 1px; white-space:nowrap; }}
    .dashboard {{ display:grid; grid-template-columns:1fr 1.4fr; gap:18px; margin-bottom:18px; }}
    .panel {{ padding:22px; }}
    .status-layout {{ display:flex; gap:28px; align-items:center; }}
    .donut {{ width:154px; aspect-ratio:1; flex:0 0 auto; border-radius:50%; background:conic-gradient({conic}); display:grid; place-items:center; }}
    .donut:after {{ content:"{total} 题"; display:grid; place-items:center; width:104px; aspect-ratio:1; border-radius:50%; background:var(--panel); font-size:22px; font-weight:800; }}
    .legend {{ flex:1; display:grid; gap:10px; }}
    .legend div {{ display:grid; grid-template-columns:10px 1fr auto; align-items:center; gap:9px; }}
    .dot {{ width:10px; height:10px; border-radius:50%; }}
    .type-row {{ display:grid; grid-template-columns:74px 1fr 28px; align-items:center; gap:12px; margin:12px 0; }}
    .stacked-bar {{ height:25px; display:flex; overflow:hidden; border-radius:7px; background:#0a1424; }}
    .stacked-bar span {{ min-width:18px; display:flex; align-items:center; justify-content:center; color:#07101c; font-size:11px; font-weight:900; }}
    .findings {{ margin:0; padding-left:21px; display:grid; gap:9px; color:#dce7f7; }}
    .charts {{ display:grid; grid-template-columns:repeat(3,1fr); gap:18px; margin-bottom:18px; }}
    .bar-row {{ display:grid; grid-template-columns:34px 1fr 82px; gap:8px; align-items:center; margin:9px 0; font-size:12px; }}
    .bar-track {{ height:10px; background:#091423; border-radius:99px; overflow:hidden; }}
    .bar-fill {{ height:100%; border-radius:99px; }}
    .bar-row strong {{ text-align:right; font-size:11px; }}
    .toolbar {{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:16px; }}
    input,select {{ color:var(--text); background:#0a1526; border:1px solid var(--line); padding:10px 12px; border-radius:9px; font:inherit; }}
    input {{ min-width:280px; flex:1; }}
    .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:12px; }}
    table {{ width:100%; border-collapse:collapse; min-width:1120px; }}
    th {{ position:sticky; top:0; background:#182840; color:#b9c6d9; text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.08em; z-index:1; }}
    th,td {{ padding:13px 12px; border-bottom:1px solid #22324a; vertical-align:top; }}
    tbody tr:hover {{ background:rgba(83,167,255,.055); }}
    td:first-child small {{ display:block; }}
    .question-cell {{ min-width:370px; max-width:560px; }}
    summary {{ cursor:pointer; font-weight:650; }}
    .detail-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:9px; padding:13px; margin-top:10px; border-radius:9px; background:#091423; }}
    .detail-grid span {{ display:block; color:var(--muted); font-size:10px; text-transform:uppercase; }}
    code {{ color:#b9d9ff; overflow-wrap:anywhere; }}
    .answer-text {{ color:var(--muted); font-size:12px; margin-top:6px; max-width:330px; }}
    .failure-reason {{ color:#ff9f9f; font-size:12px; }}
    .status-pill,.type-pill {{ display:inline-block; padding:3px 8px; border-radius:99px; font-size:11px; font-weight:750; white-space:nowrap; }}
    .status-scored {{ color:#5ee6a5; background:#11382d; }} .status-failed {{ color:#ff9f9f; background:#451f2a; }} .status-skipped {{ color:#c3ccda; background:#303a49; }}
    .type-position {{ color:#ffc279; background:#3b2d1e; }} .type-time {{ color:#c5b3ff; background:#30284e; }} .type-duration {{ color:#79efed; background:#193c43; }} .type-binary {{ color:#8ec8ff; background:#193553; }} .type-text {{ color:#c9d1dd; background:#333b48; }}
    footer {{ color:var(--muted); font-size:12px; margin-top:16px; }}
    @media(max-width:1100px) {{ .metrics {{ grid-template-columns:repeat(3,1fr); }} .charts {{ grid-template-columns:1fr; }} }}
    @media(max-width:720px) {{ .container {{ width:min(100% - 22px,1440px); padding-top:24px; }} header {{ display:block; }} .source {{ text-align:left; margin-top:12px; }} .metrics {{ grid-template-columns:1fr 1fr; }} .dashboard {{ grid-template-columns:1fr; }} .status-layout {{ align-items:flex-start; }} }}
    @media print {{ body {{ background:#fff; color:#111; }} .metric-card,.panel {{ box-shadow:none; }} .toolbar {{ display:none; }} }}
  </style>
</head>
<body>
  <main class="container">
    <header>
      <div>
        <div class="eyebrow">Evaluation report · Sequence {esc(sequence_id)}</div>
        <h1>{title}</h1>
        <p class="muted">VILA1.5-13B captions · 3 秒间隔 · 本地向量检索 · Qwen3-8B 回答</p>
      </div>
      <div class="source">生成于 {esc(generated_at)}<br>结果：{esc(result_path)}<br>题目：{esc(questions_path)}</div>
    </header>

    <section class="metrics">{cards}</section>

    <section class="dashboard">
      <article class="panel">
        <h2>题目处理结果</h2>
        <div class="status-layout">
          <div class="donut" aria-label="{total} 题"></div>
          <div class="legend">
            <div><i class="dot" style="background:{STATUS_COLORS['scored']}"></i><span>已计分</span><strong>{scored}</strong></div>
            <div><i class="dot" style="background:{STATUS_COLORS['failed']}"></i><span>结构化失败</span><strong>{failed}</strong></div>
            <div><i class="dot" style="background:{STATUS_COLORS['skipped']}"></i><span>未计分</span><strong>{skipped}</strong></div>
          </div>
        </div>
      </article>
      <article class="panel">
        <h2>不同题型的处理结果</h2>
        {''.join(type_chart)}
      </article>
    </section>

    <section class="dashboard">
      <article class="panel">
        <h2>当前结果暴露的问题</h2>
        <ol class="findings">{''.join(f'<li>{esc(item)}</li>' for item in findings)}</ol>
      </article>
      <article class="panel">
        <h2>推理耗时</h2>
        <p class="muted">仅包含实际调用模型的题目；失败重试时间已计入。</p>
        <div style="margin-top:16px">{horizontal_bars(latencies, 's', '#53a7ff')}</div>
      </article>
    </section>

    <section class="charts">
      <article class="panel"><h2>位置误差</h2>{horizontal_bars(errors['position'], 'm', '#ffae57')}</article>
      <article class="panel"><h2>时间误差</h2>{horizontal_bars(errors['time'], 'min', '#ad8cff')}</article>
      <article class="panel"><h2>持续时间误差</h2>{horizontal_bars(errors['duration'], 'min', '#3ddbd9')}</article>
    </section>

    <section class="panel">
      <h2>逐题结果</h2>
      <div class="toolbar">
        <input id="search" type="search" placeholder="搜索问题、ID、答案或失败原因">
        <select id="statusFilter"><option value="all">全部状态</option><option value="scored">已计分</option><option value="failed">失败</option><option value="skipped">跳过</option></select>
        <select id="typeFilter"><option value="all">全部题型</option>{''.join(f'<option value="{key}">{TYPE_LABELS[key]}</option>' for key in type_order)}</select>
        <span class="muted" id="visibleCount"></span>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>序号</th><th>题型</th><th>状态</th><th>问题与真值</th><th>模型预测</th><th>误差</th><th>耗时</th></tr></thead>
          <tbody id="resultsBody">{''.join(table_rows)}</tbody>
        </table>
      </div>
    </section>
    <footer>这是序列 {esc(sequence_id)} 的端到端 NaVQA 报告；文本题由 {esc(result.get('config', {}).get('text_judge_model') or '未配置')} 做语义判分，不是独立的纯检索 Recall@K 评测。</footer>
  </main>
  <script>
    const search = document.getElementById('search');
    const statusFilter = document.getElementById('statusFilter');
    const typeFilter = document.getElementById('typeFilter');
    const tableRows = [...document.querySelectorAll('#resultsBody tr')];
    const visibleCount = document.getElementById('visibleCount');
    function applyFilters() {{
      const term = search.value.trim().toLowerCase();
      let shown = 0;
      tableRows.forEach(row => {{
        const visible = (statusFilter.value === 'all' || row.dataset.status === statusFilter.value)
          && (typeFilter.value === 'all' || row.dataset.type === typeFilter.value)
          && (!term || row.dataset.search.includes(term));
        row.hidden = !visible;
        if (visible) shown += 1;
      }});
      visibleCount.textContent = `显示 ${{shown}} / ${{tableRows.length}} 题`;
    }}
    [search,statusFilter,typeFilter].forEach(control => control.addEventListener('input',applyFilters));
    applyFilters();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path, help="Evaluation JSON")
    parser.add_argument("--questions", required=True, type=Path, help="Question JSON")
    parser.add_argument("--output", required=True, type=Path, help="Output HTML")
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    question_document = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = question_document.get("data", question_document)
    if not isinstance(questions, list):
        raise TypeError("Question JSON must contain a list or a top-level 'data' list")

    report = build_report(result, questions, args.result, args.questions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
