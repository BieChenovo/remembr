#!/usr/bin/env bash
# Run all 30 sequence-0 NaVQA questions with persisted retrieval traces.
#
# This launcher intentionally does not invoke gpu-guard or gpu_watchdog and
# does not stop the Web GPU environment when evaluation finishes.

set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/hpc2hdd/JH_DATA/jhai_data/yichenwang/projects/remembr}
RUNTIME_ROOT=${RUNTIME_ROOT:-/hpc2ssd/JH_DATA/spooler/yichenwang/projects/remembr/artifacts}
EVAL_TAG=${EVAL_TAG:-trace_seq0_nothink_256_20260828_v1}
RUN_NAME=${RUN_NAME:-navqa-trace-seq0-20260828-v1}
RUN_DIR=${RUN_DIR:-$RUNTIME_ROOT/manual_runs/$RUN_NAME}
RUN_LOG=$RUN_DIR/run.log
RESULT_FILE=$RUNTIME_ROOT/eval_outs/0/human_qa/remembr+qwen3:8b__captions_VILA1.5-13b_3_secs_${EVAL_TAG}.json

mkdir -p "$RUN_DIR"
printf '%s\n' "$$" >"$RUN_DIR/pid"
printf '%s\n' "$(date --iso-8601=seconds)" >"$RUN_DIR/started_at"
printf '%s\n' "$RESULT_FILE" >"$RUN_DIR/result_path"
exec >>"$RUN_LOG" 2>&1

on_exit() {
    local rc=$?
    printf '%s\n' "$rc" >"$RUN_DIR/exit_code"
    printf '%s\n' "$(date --iso-8601=seconds)" >"$RUN_DIR/finished_at"
    printf '[%s] Evaluation process exited with code %s. GPU environment left running.\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "$rc"
}
trap on_exit EXIT

printf '[%s] Starting unguarded retrieval-trace evaluation.\n' "$(date '+%Y-%m-%d %H:%M:%S')"
printf '[%s] Result: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$RESULT_FILE"

if pgrep -af 'gpu_watchdog.py' >"$RUN_DIR/watchdog_processes.txt"; then
    printf '%s\n' \
        'Refusing to start because an existing gpu_watchdog.py process could stop this environment.' \
        "Inspect $RUN_DIR/watchdog_processes.txt and stop that watchdog explicitly first."
    exit 3
fi

nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader

env \
    PROJECT_ROOT="$PROJECT_ROOT" \
    INPUT_ARTIFACT_ROOT="$PROJECT_ROOT/artifacts" \
    RUNTIME_ROOT="$RUNTIME_ROOT" \
    MODEL=qwen3:8b \
    SEQUENCE_ID=0 \
    SMOKE_EVAL=0 \
    FULL_EVAL=1 \
    EVAL_TAG="$EVAL_TAG" \
    NUM_PREDICT=256 \
    DISABLE_THINKING=1 \
    QUESTION_INDICES= \
    TEXT_JUDGE_MODEL=qwen3:8b \
    TEXT_JUDGE_NUM_PREDICT=96 \
    NAVQA_TIMEZONE=America/Los_Angeles \
    bash "$PROJECT_ROOT/remembr/scripts/bash_scripts/run_navqa_qwen.sh"

printf '[%s] Retrieval-trace evaluation completed. Environment remains running.\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')"
