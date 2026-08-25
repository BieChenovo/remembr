#!/usr/bin/env bash
# Download Ollama and Qwen into shared storage without occupying a GPU.

set -Eeuo pipefail

USER_ROOT=/hpc2hdd/home/yichenwang
BOOTSTRAP_ROOT="$USER_ROOT/.local/lib/remembr-bootstrap"
OLLAMA_ROOT="$USER_ROOT/.local/opt/ollama"
OLLAMA_BIN="$OLLAMA_ROOT/bin/ollama"
OLLAMA_MODELS="$USER_ROOT/.local/share/ollama/models"
DOWNLOAD_CACHE="$USER_ROOT/.cache/remembr-downloads"
MODEL=${MODEL:-qwen3:8b}
# A per-process port avoids collisions with a stale local-port proxy left by an
# interrupted managed-terminal session.
OLLAMA_HOST=${OLLAMA_HOST:-127.0.0.1:$((20000 + $$ % 20000))}

export OLLAMA_MODELS OLLAMA_HOST
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
mkdir -p "$BOOTSTRAP_ROOT" "$OLLAMA_ROOT" "$OLLAMA_MODELS" "$DOWNLOAD_CACHE"

step() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

cleanup() {
    if [[ -n ${OLLAMA_PID:-} ]] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
        kill "$OLLAMA_PID" 2>/dev/null || true
        wait "$OLLAMA_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if ! PYTHONPATH="$BOOTSTRAP_ROOT" python3 -c 'import zstandard' >/dev/null 2>&1; then
    step "Installing the small zstandard bootstrap package"
    python3 -m pip install --target "$BOOTSTRAP_ROOT" zstandard==0.23.0
fi

if [[ ! -x "$OLLAMA_BIN" ]]; then
    archive="$DOWNLOAD_CACHE/ollama-linux-amd64.tar.zst"
    step "Downloading Ollama archive with resume support"
    curl --fail --location --retry 8 \
        --continue-at - \
        --output "$archive" \
        https://ollama.com/download/ollama-linux-amd64.tar.zst
    step "Extracting Ollama into shared home"
    PYTHONPATH="$BOOTSTRAP_ROOT" python3 -c \
        'import sys, zstandard; zstandard.ZstdDecompressor().copy_stream(sys.stdin.buffer, sys.stdout.buffer)' \
        < "$archive" | tar -xf - -C "$OLLAMA_ROOT"
    rm -f -- "$archive"
fi

mkdir -p "$USER_ROOT/.local/bin"
ln -sfn "$OLLAMA_BIN" "$USER_ROOT/.local/bin/ollama"

step "Starting download-only Ollama service at $OLLAMA_HOST"
"$OLLAMA_BIN" serve > "$DOWNLOAD_CACHE/ollama-cpu-stage-server.log" 2>&1 &
OLLAMA_PID=$!
for _ in $(seq 1 60); do
    if curl --noproxy '*' --silent --fail --connect-timeout 2 --max-time 5 \
        "http://$OLLAMA_HOST/api/tags" >/dev/null; then
        break
    fi
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        tail -100 "$DOWNLOAD_CACHE/ollama-cpu-stage-server.log"
        exit 1
    fi
    sleep 2
done
curl --noproxy '*' --silent --fail --connect-timeout 2 --max-time 5 \
    "http://$OLLAMA_HOST/api/tags" >/dev/null

if ! "$OLLAMA_BIN" list | awk 'NR > 1 {print $1}' | grep -Fxq "$MODEL"; then
    step "Downloading $MODEL into shared model storage"
    "$OLLAMA_BIN" pull "$MODEL"
fi

step "$MODEL is fully staged; no inference was run on the management node"
