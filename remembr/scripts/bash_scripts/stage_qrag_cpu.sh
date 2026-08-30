#!/usr/bin/env bash
# Stage the official Q-RAG code and the most relevant open-domain checkpoint.
# This is CPU/network-only and is safe to run outside a rented GPU container.

set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/hpc2hdd/JH_DATA/jhai_data/yichenwang/projects/remembr}
THIRD_PARTY_ROOT=${THIRD_PARTY_ROOT:-$PROJECT_ROOT/third_party}
QRAG_ROOT=${QRAG_ROOT:-$THIRD_PARTY_ROOT/Q-RAG}
MODEL_ID=${MODEL_ID:-Q-RAG/qrag-ft-gte-on-hotpotqa_musique}
MODEL_NAME=${MODEL_NAME:-qrag-ft-gte-on-hotpotqa_musique}
MODEL_ROOT=${MODEL_ROOT:-$THIRD_PARTY_ROOT/qrag-models/$MODEL_NAME}

mkdir -p "$THIRD_PARTY_ROOT" "$MODEL_ROOT"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

step() {
    printf '[%s] %s\n' "$(timestamp)" "$*"
}

if [[ ! -d "$QRAG_ROOT/.git" ]]; then
    if [[ -e "$QRAG_ROOT" ]]; then
        step "$QRAG_ROOT exists but is not a Git checkout; refusing to overwrite it"
        exit 1
    fi
    step "Cloning the official Q-RAG repository"
    git clone --depth 1 https://github.com/griver/Q-RAG.git "$QRAG_ROOT"
else
    step "Q-RAG repository already exists; leaving its checked-out revision unchanged"
fi

git -C "$QRAG_ROOT" rev-parse HEAD > "$MODEL_ROOT/qrag_code_commit.txt"
printf '%s\n' "$MODEL_ID" > "$MODEL_ROOT/huggingface_model_id.txt"

download_file() {
    local filename=$1
    local destination="$MODEL_ROOT/$filename"
    local url="https://huggingface.co/$MODEL_ID/resolve/main/$filename?download=true"
    if [[ -s "$destination" && "$filename" != model_best.pt ]]; then
        step "$filename already exists; skipping"
        return
    fi
    step "Downloading $filename (resumable)"
    curl --fail --location \
        --retry 30 --retry-delay 10 \
        --connect-timeout 30 --continue-at - \
        --output "$destination" "$url"
}

download_file config.yaml
download_file config_orig.yaml
download_file model_best.pt

step "Computing SHA-256 for the completed checkpoint"
sha256sum "$MODEL_ROOT/model_best.pt" > "$MODEL_ROOT/model_best.pt.sha256"
du -sh "$MODEL_ROOT"
step "Q-RAG code and checkpoint staging completed"
