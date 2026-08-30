#!/usr/bin/env bash
# Re-run sequence 0, question 1 with retrieval tracing and release the Web GPU
# environment after the task exits.

set -Eeuo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' \
        'This launcher requires an NVIDIA GPU. For Ascend, use the dedicated' \
        'vLLM Ascend/OpenAI-compatible launcher after the NPU environment is verified.' >&2
    exit 2
fi

PROJECT_ROOT=${PROJECT_ROOT:-/hpc2hdd/JH_DATA/jhai_data/yichenwang/projects/remembr}
RUNTIME_ROOT=${RUNTIME_ROOT:-/hpc2ssd/JH_DATA/spooler/yichenwang/projects/remembr/artifacts}
EVAL_TAG=${EVAL_TAG:-trace_q1_nothink_256_v1}
RUN_NAME=${RUN_NAME:-navqa-seq0-q1-trace}
GPU_GUARD=${GPU_GUARD:-/hpc2hdd/home/yichenwang/.local/bin/gpu-guard}
# Leave time to inspect the retrieval trace after the task finishes.
POST_RUN_GRACE_SECONDS=${POST_RUN_GRACE_SECONDS:-600}

export GPU_GUARD_STATE_ROOT=${GPU_GUARD_STATE_ROOT:-$RUNTIME_ROOT/gpu-guard}

"$GPU_GUARD" run \
    --name "$RUN_NAME" \
    --startup-grace 900 \
    --missing-timeout "$POST_RUN_GRACE_SECONDS" \
    --no-sample-timeout 300 \
    --low-util-window 1800 \
    --low-util-threshold 20 \
    -- env \
        PROJECT_ROOT="$PROJECT_ROOT" \
        RUNTIME_ROOT="$RUNTIME_ROOT" \
        MODEL=qwen3:8b \
        SEQUENCE_ID=0 \
        QUESTION_INDICES=0 \
        SMOKE_EVAL=0 \
        FULL_EVAL=1 \
        EVAL_TAG="$EVAL_TAG" \
        NUM_PREDICT=256 \
        DISABLE_THINKING=1 \
        TEXT_JUDGE_MODEL=qwen3:8b \
        TEXT_JUDGE_NUM_PREDICT=96 \
        NAVQA_TIMEZONE=America/Los_Angeles \
        bash "$PROJECT_ROOT/remembr/scripts/bash_scripts/run_navqa_qwen.sh"
