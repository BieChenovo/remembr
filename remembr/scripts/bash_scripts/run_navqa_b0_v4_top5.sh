#!/usr/bin/env bash
# Run the repaired closed-loop B0 baseline with five dense text memories per call.

set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/hpc2hdd/JH_DATA/jhai_data/yichenwang/projects/remembr}
RUNTIME_ROOT=${RUNTIME_ROOT:-/hpc2ssd/JH_DATA/spooler/yichenwang/projects/remembr/artifacts}

export PROJECT_ROOT
export INPUT_ROOT=${INPUT_ROOT:-$PROJECT_ROOT/artifacts}
export RESULT_ROOT=${RESULT_ROOT:-$RUNTIME_ROOT/eval_outs}
export RUNTIME_ROOT
export GPU_COUNT=${GPU_COUNT:-4}
export GPU_IDS=${GPU_IDS:-"0 1 2 3"}
export TEXT_RETRIEVER=dense
export RUN_TAG=dense_210_question_state_v4_top5
export REPORT_VERSION=v4
export REPORT_DIR=${REPORT_DIR:-$RUNTIME_ROOT/eval_reports/$RUN_TAG}
export TEXT_EPISODE_MODE=question
export QUESTION_TEXT_EVIDENCE_BUDGET=5
export UNIFIED_EVIDENCE_LEDGER=0
export NUMERIC_K=${NUMERIC_K:-4}
export DUPLICATE_REPLAN_LIMIT=2

exec bash "$PROJECT_ROOT/remembr/scripts/bash_scripts/run_navqa_7seq_retriever.sh"
