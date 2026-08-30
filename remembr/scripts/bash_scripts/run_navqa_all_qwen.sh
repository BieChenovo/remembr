#!/usr/bin/env bash
# Run all seven CODa/NaVQA sequences with resumable, comparable settings.
# Launch this wrapper through gpu-guard so the Web GPU environment shuts down
# after the entire batch finishes or aborts.

set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/hpc2hdd/home/yichenwang/jhaidata/projects/remembr}
REMEMBR_ROOT="$PROJECT_ROOT/remembr"
RUNTIME_ROOT=${RUNTIME_ROOT:-/hpc2ssd/JH_DATA/spooler/yichenwang/projects/remembr/artifacts}
MODEL=${MODEL:-qwen3:8b}
SEQUENCE_IDS=${SEQUENCE_IDS:-"0 3 4 6 16 21 22"}
EVAL_TAG=${EVAL_TAG:-full_nothink_1024}
SMOKE_TAG=${SMOKE_TAG:-smoke_nothink_1024}
NUM_PREDICT=${NUM_PREDICT:-1024}
DISABLE_THINKING=${DISABLE_THINKING:-1}
RUN_SMOKE=${RUN_SMOKE:-1}
TEXT_JUDGE_MODEL=${TEXT_JUDGE_MODEL:-$MODEL}
TEXT_JUDGE_NUM_PREDICT=${TEXT_JUDGE_NUM_PREDICT:-96}
NAVQA_TIMEZONE=${NAVQA_TIMEZONE:-America/Los_Angeles}

BATCH_DIR="$RUNTIME_ROOT/batches/$EVAL_TAG"
SUMMARY_FILE="$BATCH_DIR/summary.tsv"
RUN_ONE="$REMEMBR_ROOT/scripts/bash_scripts/run_navqa_qwen.sh"
REPORT_DIR="$RUNTIME_ROOT/eval_reports/$EVAL_TAG"
mkdir -p "$BATCH_DIR" "$REPORT_DIR"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

step() {
    printf '[%s] %s\n' "$(timestamp)" "$*"
}

result_path() {
    local sequence_id=$1
    printf '%s/eval_outs/%s/human_qa/remembr+%s__captions_VILA1.5-13b_3_secs_%s.json' \
        "$RUNTIME_ROOT" "$sequence_id" "$MODEL" "$EVAL_TAG"
}

result_is_complete() {
    local result=$1
    [[ -f "$result" ]] || return 1
    python3 - "$result" <<'PY'
import json
import sys

try:
    result = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    raise SystemExit(1)
metrics = result.get("metrics", {})
complete = (
    result.get("in_progress") is False
    and metrics.get("questions_completed") == metrics.get("questions_total") == 30
)
raise SystemExit(0 if complete else 1)
PY
}

write_summary() {
    local sequence_id=$1
    local result=$2
    python3 - "$sequence_id" "$result" <<'PY' >>"$SUMMARY_FILE"
import json
import sys

sequence_id, path = sys.argv[1:]
result = json.load(open(path))
metrics = result["metrics"]
fields = [
    sequence_id,
    str(metrics.get("questions_completed")),
    str(metrics.get("questions_scored")),
    str(metrics.get("questions_failed")),
    str(metrics.get("descriptive_count")),
    str(metrics.get("descriptive_accuracy")),
    str(metrics.get("binary_accuracy")),
    str(metrics.get("text_accuracy")),
    str(metrics.get("position_mean_l2_error")),
    str(metrics.get("time_mean_absolute_error")),
    str(metrics.get("duration_mean_absolute_error")),
    path,
]
print("\t".join(fields))
PY
}

write_report() {
    local sequence_id=$1
    local result=$2
    python3 "$REMEMBR_ROOT/scripts/visualize_eval_report.py" \
        --result "$result" \
        --questions "$PROJECT_ROOT/artifacts/questions/$sequence_id/human_qa.json" \
        --output "$REPORT_DIR/sequence_${sequence_id}.html"
}

printf 'sequence\tcompleted\tscored\tfailed\tdescriptive_count\tdescriptive_accuracy\tbinary_accuracy\ttext_accuracy\tposition_l2\ttime_mae\tduration_mae\tresult\n' >"$SUMMARY_FILE"

smoke_pending=$RUN_SMOKE
batch_start=$SECONDS
for sequence_id in $SEQUENCE_IDS; do
    result=$(result_path "$sequence_id")
    if result_is_complete "$result"; then
        step "Sequence $sequence_id already complete; skipping"
        write_summary "$sequence_id" "$result"
        write_report "$sequence_id" "$result"
        continue
    fi

    smoke_eval=0
    if [[ "$smoke_pending" == 1 ]]; then
        smoke_eval=1
        smoke_pending=0
    fi

    step "Starting sequence $sequence_id (smoke=$smoke_eval, tag=$EVAL_TAG)"
    SEQUENCE_ID="$sequence_id" \
    MODEL="$MODEL" \
    SMOKE_EVAL="$smoke_eval" \
    FULL_EVAL=1 \
    SMOKE_TAG="$SMOKE_TAG" \
    EVAL_TAG="$EVAL_TAG" \
    NUM_PREDICT="$NUM_PREDICT" \
    DISABLE_THINKING="$DISABLE_THINKING" \
    TEXT_JUDGE_MODEL="$TEXT_JUDGE_MODEL" \
    TEXT_JUDGE_NUM_PREDICT="$TEXT_JUDGE_NUM_PREDICT" \
    NAVQA_TIMEZONE="$NAVQA_TIMEZONE" \
    PROJECT_ROOT="$PROJECT_ROOT" \
    RUNTIME_ROOT="$RUNTIME_ROOT" \
        bash "$RUN_ONE"

    if ! result_is_complete "$result"; then
        step "Sequence $sequence_id exited without a complete result: $result"
        exit 1
    fi
    write_summary "$sequence_id" "$result"
    write_report "$sequence_id" "$result"
    step "Sequence $sequence_id complete"
done

step "All requested sequences complete in $((SECONDS - batch_start)) seconds"
step "Summary: $SUMMARY_FILE"
