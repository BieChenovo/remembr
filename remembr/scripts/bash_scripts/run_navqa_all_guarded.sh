#!/usr/bin/env bash
# Launch the complete, fixed NaVQA experiment under the account-wide GPU guard.

set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/hpc2hdd/JH_DATA/jhai_data/yichenwang/projects/remembr}
RUNTIME_ROOT=${RUNTIME_ROOT:-/hpc2ssd/JH_DATA/spooler/yichenwang/projects/remembr/artifacts}
EVAL_TAG=${EVAL_TAG:-full_nothink_256_pst_descriptive_v1}
RUN_NAME=${RUN_NAME:-navqa-all-256-pst-desc}
GPU_GUARD=${GPU_GUARD:-/hpc2hdd/home/yichenwang/.local/bin/gpu-guard}
# Keep the environment available briefly after the real workload exits so logs
# and outputs can be inspected. Override per run, for example:
# POST_RUN_GRACE_SECONDS=1800 bash .../run_navqa_all_guarded.sh
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
        SEQUENCE_IDS="0 3 4 6 16 21 22" \
        EVAL_TAG="$EVAL_TAG" \
        NUM_PREDICT=256 \
        DISABLE_THINKING=1 \
        RUN_SMOKE=0 \
        TEXT_JUDGE_MODEL=qwen3:8b \
        TEXT_JUDGE_NUM_PREDICT=96 \
        NAVQA_TIMEZONE=America/Los_Angeles \
        bash "$PROJECT_ROOT/remembr/scripts/bash_scripts/run_navqa_all_qwen.sh"
