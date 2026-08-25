#!/usr/bin/env bash

set -euo pipefail

HOME_ROOT=/hpc2hdd/home/yichenwang
PROJECT_ROOT="$HOME_ROOT/jhaidata/projects/remembr"
PYTHON="$HOME_ROOT/envs/remembr/bin/python"
VILA_MODEL="$HOME_ROOT/.cache/huggingface/hub/models--Efficient-Large-Model--VILA1.5-13b/snapshots/9c8575753b9376c86289b39cc839295c9899a753"
EMBEDDING_MODEL="$HOME_ROOT/.cache/huggingface/hub/models--mixedbread-ai--mxbai-embed-large-v1/snapshots/b33106f585b9ce46904ad7443a3b52b7a63e231c"
OUTPUT_ROOT="$PROJECT_ROOT/artifacts/captions"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU is not available"; print(torch.cuda.get_device_name(0), flush=True)'

cd "$PROJECT_ROOT/remembr"

for sequence_id in 0 3 4 6 16 21 22; do
    echo "Starting CODa sequence $sequence_id at $(date --iso-8601=seconds)"
    "$PYTHON" scripts/preprocess_captions.py \
        --seq_id "$sequence_id" \
        --seconds_per_caption 3 \
        --num-video-frames 6 \
        --model-path "$VILA_MODEL" \
        --embedding-model "$EMBEDDING_MODEL" \
        --captioner_name VILA1.5-13b \
        --conv-mode vicuna_v1 \
        --temperature 0 \
        --max_new_tokens 128 \
        --checkpoint-every 10 \
        --resume \
        --data_path "$PROJECT_ROOT/remembr/coda_data" \
        --out_path "$OUTPUT_ROOT/$sequence_id/captions"
done
