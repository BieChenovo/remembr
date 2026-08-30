#!/usr/bin/env bash
# Start the balanced four-GPU resume under the account-wide fail-safe guard.

set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/hpc2hdd/JH_DATA/jhai_data/yichenwang/projects/remembr}
RUNTIME_ROOT=${RUNTIME_ROOT:-/hpc2ssd/JH_DATA/spooler/yichenwang/projects/remembr/artifacts}
GPU_GUARD=${GPU_GUARD:-/hpc2hdd/home/yichenwang/.local/bin/gpu-guard}
RUN_NAME=${RUN_NAME:-navqa-4gpu-resume}
# Delay environment shutdown after the worker exits. This is a grace period,
# not a synthetic GPU workload.
POST_RUN_GRACE_SECONDS=${POST_RUN_GRACE_SECONDS:-600}

export GPU_GUARD_STATE_ROOT=${GPU_GUARD_STATE_ROOT:-$RUNTIME_ROOT/gpu-guard}

"$GPU_GUARD" run \
    --name "$RUN_NAME" \
    --startup-grace 600 \
    --missing-timeout "$POST_RUN_GRACE_SECONDS" \
    --no-sample-timeout 180 \
    --low-util-window 900 \
    --low-util-threshold 25 \
    -- env \
        PROJECT_ROOT="$PROJECT_ROOT" \
        RUNTIME_ROOT="$RUNTIME_ROOT" \
        GPU_COUNT=4 \
        WORKERS_PER_GPU=2 \
        SOURCE_TAG=full_nothink_256_pst_descriptive_v1 \
        TARGET_TAG=full_nothink_256_pst_descriptive_4gpu_resume_v1 \
        bash "$PROJECT_ROOT/remembr/scripts/bash_scripts/run_navqa_4gpu_resume.sh"
