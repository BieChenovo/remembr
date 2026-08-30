#!/usr/bin/env bash
# Run the B1 ablation: original ReMEmbR controller/reader with GTE-base dense text retrieval.

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-/hpc2hdd/home/yichenwang/jhaidata/projects/remembr}

export PROJECT_ROOT
export TEXT_RETRIEVER=gte_dense
export GTE_MODEL=${GTE_MODEL:-$PROJECT_ROOT/third_party/gte-models/gte-multilingual-base}
export SEQUENCE_ID=${SEQUENCE_ID:-0}
export SMOKE_TAG=${SMOKE_TAG:-gte_dense_smoke}
export EVAL_TAG=${EVAL_TAG:-gte_dense_full}
export DISABLE_THINKING=${DISABLE_THINKING:-1}

exec "$SCRIPT_DIR/run_navqa_qwen.sh"
