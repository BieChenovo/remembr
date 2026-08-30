#!/usr/bin/env bash
# Wait on the management node for the unguarded GPU evaluation, then build the
# sequence-0 trace analysis and audit page. This script never touches Jupyter,
# the GPU allocation, or any watchdog.

set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/hpc2hdd/JH_DATA/jhai_data/yichenwang/projects/remembr}
RUNTIME_ROOT=${RUNTIME_ROOT:-/hpc2ssd/JH_DATA/spooler/yichenwang/projects/remembr/artifacts}
EVAL_TAG=${EVAL_TAG:-trace_seq0_nothink_256_20260828_v1}
RUN_NAME=${RUN_NAME:-navqa-trace-seq0-20260828-v1}
RUN_DIR=$RUNTIME_ROOT/manual_runs/$RUN_NAME
RESULT_FILE=$RUNTIME_ROOT/eval_outs/0/human_qa/remembr+qwen3:8b__captions_VILA1.5-13b_3_secs_${EVAL_TAG}.json
ANALYSIS_DIR=$PROJECT_ROOT/artifacts/eval_reports/sequence_0_trace_analysis_v1
AUDIT_DIR=$PROJECT_ROOT/artifacts/eval_reports/sequence_0_audit_trace_v1
PYTHON=${PYTHON:-/hpc2hdd/home/yichenwang/envs/coda/bin/python}

printf '[%s] Waiting for %s\n' "$(date --iso-8601=seconds)" "$RUN_DIR/exit_code"
while [[ ! -s "$RUN_DIR/exit_code" ]]; do
    sleep 30
done

read -r task_rc <"$RUN_DIR/exit_code"
if [[ "$task_rc" != 0 ]]; then
    printf 'GPU evaluation failed with exit code %s; inspect %s\n' \
        "$task_rc" "$RUN_DIR/run.log" >&2
    exit "$task_rc"
fi
if [[ ! -s "$RESULT_FILE" ]]; then
    printf 'Evaluation reported success but result is missing: %s\n' "$RESULT_FILE" >&2
    exit 4
fi

cd "$PROJECT_ROOT"
"$PYTHON" - "$RESULT_FILE" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    responses = json.load(handle)["responses"]
if len(responses) != 30:
    raise SystemExit(f"expected 30 responses, got {len(responses)}")
missing = [index + 1 for index, row in enumerate(responses) if "retrieval_trace" not in row]
if missing:
    raise SystemExit(f"responses missing retrieval_trace fields: {missing}")
calls = sum(len(row["retrieval_trace"]) for row in responses)
zero_call = [index + 1 for index, row in enumerate(responses) if not row["retrieval_trace"]]
print(
    f"validated {len(responses)} trace-enabled responses, "
    f"{calls} retrieval calls; zero-call questions: {zero_call}"
)
PY

"$PYTHON" remembr/scripts/analyze_navqa_errors.py \
    --sequences 0 \
    --result-root "$RUNTIME_ROOT/eval_outs" \
    --tag "$EVAL_TAG" \
    --output-dir "$ANALYSIS_DIR"

mkdir -p "$AUDIT_DIR"
if [[ ! -e "$AUDIT_DIR/frames" ]]; then
    ln -s ../sequence_0_audit_v1/frames "$AUDIT_DIR/frames"
fi
if [[ ! -e "$AUDIT_DIR/media" ]]; then
    ln -s ../sequence_0_audit_v1/media "$AUDIT_DIR/media"
fi

"$PYTHON" remembr/scripts/build_sequence_audit.py \
    --sequence-id 0 \
    --result "$RESULT_FILE" \
    --analysis "$ANALYSIS_DIR/error_analysis.json" \
    --output-dir "$AUDIT_DIR" \
    --skip-video \
    --skip-contact-sheets

printf '[%s] Trace audit ready: %s\n' \
    "$(date --iso-8601=seconds)" "$AUDIT_DIR/index.html"
