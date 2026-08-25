#!/usr/bin/env bash
# Compare Qwen3 no-think output budgets on the same stratified NaVQA subset.

set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/hpc2hdd/home/yichenwang/jhaidata/projects/remembr}
REMEMBR_ROOT="$PROJECT_ROOT/remembr"
RUNTIME_ROOT=${RUNTIME_ROOT:-/hpc2ssd/JH_DATA/spooler/yichenwang/projects/remembr/artifacts}
MODEL=${MODEL:-qwen3:8b}
TOKEN_BUDGETS=${TOKEN_BUDGETS:-"256 512"}
# Ten non-text questions spanning long/medium/short histories and all scored
# answer types. Indices 9, 15, and 28 were difficult position questions in the
# thinking baseline, making the subset useful for detecting format regressions.
QUESTION_INDICES=${QUESTION_INDICES:-"0,2,3,8,9,14,15,20,28,29"}

RUN_ONE="$REMEMBR_ROOT/scripts/bash_scripts/run_navqa_qwen.sh"
BENCHMARK_DIR="$RUNTIME_ROOT/benchmarks/qwen_token_budget"
SUMMARY="$BENCHMARK_DIR/summary.tsv"
mkdir -p "$BENCHMARK_DIR"

printf 'budget\tcompleted\tscored\tfailed\tmean_latency_s\tmedian_latency_s\tbinary_accuracy\tposition_l2\ttime_mae\tduration_mae\tresult\n' >"$SUMMARY"

baseline="$PROJECT_ROOT/artifacts/eval_outs/0/human_qa/remembr+$MODEL"'__captions_VILA1.5-13b_3_secs_full.json'
questions="$PROJECT_ROOT/artifacts/questions/0/human_qa.json"
if [[ -f "$baseline" && -f "$questions" ]]; then
    python3 - "$QUESTION_INDICES" "$questions" "$baseline" <<'PY' >>"$SUMMARY"
import json
import statistics
import sys

indices_text, questions_path, result_path = sys.argv[1:]
indices = [int(value) for value in indices_text.split(",")]
questions = json.load(open(questions_path))["data"]
responses = json.load(open(result_path))["responses"]
selected = [(questions[index], responses[index]) for index in indices]
latencies = [
    response.get("elapsed")
    for _, response in selected
    if response and isinstance(response.get("elapsed"), (int, float))
]
failed = sum(
    bool(response and response.get("evaluation_failure"))
    for _, response in selected
)

def mean_metric(question_type, key):
    values = [
        response["error"][key]
        for question, response in selected
        if question["type"] == question_type
        and response
        and key in response.get("error", {})
    ]
    return statistics.mean(values) if values else None

fields = [
    "2048-thinking",
    str(len(selected)),
    str(len(selected) - failed),
    str(failed),
    str(statistics.mean(latencies)),
    str(statistics.median(latencies)),
    str(mean_metric("binary", "binary_iscorrect")),
    str(mean_metric("position", "position_error")),
    str(mean_metric("time", "time_error")),
    str(mean_metric("duration", "duration_error")),
    result_path,
]
print("\t".join(fields))
PY
fi

for budget in $TOKEN_BUDGETS; do
    tag="benchmark_nothink_${budget}"
    result="$RUNTIME_ROOT/eval_outs/0/human_qa/remembr+$MODEL"'__captions_VILA1.5-13b_3_secs_'"$tag.json"
    printf '[%s] Running budget=%s on indices=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$budget" "$QUESTION_INDICES"

    SEQUENCE_ID=0 \
    MODEL="$MODEL" \
    SMOKE_EVAL=0 \
    FULL_EVAL=1 \
    EVAL_TAG="$tag" \
    NUM_PREDICT="$budget" \
    DISABLE_THINKING=1 \
    QUESTION_INDICES="$QUESTION_INDICES" \
    PROJECT_ROOT="$PROJECT_ROOT" \
    RUNTIME_ROOT="$RUNTIME_ROOT" \
        bash "$RUN_ONE"

    python3 - "$budget" "$result" <<'PY' >>"$SUMMARY"
import json
import statistics
import sys

budget, path = sys.argv[1:]
result = json.load(open(path))
metrics = result["metrics"]
if result.get("in_progress") or metrics.get("questions_completed") != 10:
    raise SystemExit(f"Incomplete benchmark result: {path}")
latencies = [
    response.get("elapsed")
    for response in result["responses"]
    if response and isinstance(response.get("elapsed"), (int, float))
]
fields = [
    budget,
    str(metrics.get("questions_completed")),
    str(metrics.get("questions_scored")),
    str(metrics.get("questions_failed")),
    str(statistics.mean(latencies) if latencies else None),
    str(statistics.median(latencies) if latencies else None),
    str(metrics.get("binary_accuracy")),
    str(metrics.get("position_mean_l2_error")),
    str(metrics.get("time_mean_absolute_error")),
    str(metrics.get("duration_mean_absolute_error")),
    path,
]
print("\t".join(fields))
PY
done

printf '[%s] Token-budget benchmark complete: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$SUMMARY"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || sed -n '1,20p' "$SUMMARY"
