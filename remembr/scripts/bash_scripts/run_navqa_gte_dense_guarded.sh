#!/usr/bin/env bash
# Launch the sequence-0 B1 GTE dense ablation under the GPU fail-safe guard.

set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/hpc2hdd/home/yichenwang/jhaidata/projects/remembr}
RUNTIME_ROOT=${RUNTIME_ROOT:-/hpc2ssd/JH_DATA/spooler/yichenwang/projects/remembr/artifacts}
GPU_GUARD=${GPU_GUARD:-/hpc2hdd/home/yichenwang/.local/bin/gpu-guard}
RUN_NAME=${RUN_NAME:-navqa-seq0-gte-dense}
POST_RUN_GRACE_SECONDS=${POST_RUN_GRACE_SECONDS:-600}

export GPU_GUARD_STATE_ROOT=${GPU_GUARD_STATE_ROOT:-$RUNTIME_ROOT/gpu-guard}

"$GPU_GUARD" run \
    --name "$RUN_NAME" \
    --startup-grace 1200 \
    --missing-timeout "$POST_RUN_GRACE_SECONDS" \
    --no-sample-timeout 300 \
    --low-util-window 1800 \
    --low-util-threshold 20 \
    -- env \
        PROJECT_ROOT="$PROJECT_ROOT" \
        RUNTIME_ROOT="$RUNTIME_ROOT" \
        SEQUENCE_ID=0 \
        SMOKE_EVAL=1 \
        FULL_EVAL=1 \
        NUM_PREDICT=256 \
        DISABLE_THINKING=1 \
        bash "$PROJECT_ROOT/remembr/scripts/bash_scripts/run_navqa_gte_dense_qwen.sh"
