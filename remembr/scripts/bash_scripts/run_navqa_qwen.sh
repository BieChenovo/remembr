#!/usr/bin/env bash
# Install a persistent minimal runtime, start local Qwen, then evaluate NaVQA.
# Run this script through gpu-guard; it deliberately never creates fake GPU load.

set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/hpc2hdd/home/yichenwang/jhaidata/projects/remembr}
REMEMBR_ROOT="$PROJECT_ROOT/remembr"
INPUT_ARTIFACT_ROOT=${INPUT_ARTIFACT_ROOT:-$PROJECT_ROOT/artifacts}
# The JH_DATA project mount is read-only inside Web GPU containers. Keep
# generated runtime files on the explicitly mounted, persistent SSD workspace.
RUNTIME_ROOT=${RUNTIME_ROOT:-/hpc2ssd/JH_DATA/spooler/yichenwang/projects/remembr/artifacts}
VENV_ROOT=${VENV_ROOT:-/hpc2hdd/home/yichenwang/.venvs/remembr-eval}
OLLAMA_ROOT=${OLLAMA_ROOT:-/hpc2hdd/home/yichenwang/.local/opt/ollama}
OLLAMA_MODELS=${OLLAMA_MODELS:-/hpc2hdd/home/yichenwang/.local/share/ollama/models}
DOWNLOAD_CACHE=${DOWNLOAD_CACHE:-/hpc2hdd/home/yichenwang/.cache/remembr-downloads}
OLLAMA_HOST=${OLLAMA_HOST:-127.0.0.1:11434}
MODEL=${MODEL:-qwen3:8b}
SEQUENCE_ID=${SEQUENCE_ID:-0}
SMOKE_EVAL=${SMOKE_EVAL:-1}
FULL_EVAL=${FULL_EVAL:-1}
SMOKE_TAG=${SMOKE_TAG:-smoke}
EVAL_TAG=${EVAL_TAG:-full}
NUM_PREDICT=${NUM_PREDICT:-2048}
DISABLE_THINKING=${DISABLE_THINKING:-0}
QUESTION_INDICES=${QUESTION_INDICES:-}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-/hpc2hdd/home/yichenwang/.cache/huggingface/hub/models--mixedbread-ai--mxbai-embed-large-v1/snapshots/b33106f585b9ce46904ad7443a3b52b7a63e231c}

export OLLAMA_MODELS OLLAMA_HOST
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_ROOT:$REMEMBR_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"

LOG_DIR="$RUNTIME_ROOT/logs"
OUT_DIR="$RUNTIME_ROOT/eval_outs"
mkdir -p "$LOG_DIR" "$OUT_DIR" "$VENV_ROOT" "$OLLAMA_MODELS" "$DOWNLOAD_CACHE"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

step() {
    printf '[%s] %s\n' "$(timestamp)" "$*"
}

cleanup() {
    if [[ -n ${OLLAMA_PID:-} ]] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
        kill "$OLLAMA_PID" 2>/dev/null || true
        wait "$OLLAMA_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

step "Checking persistent Python environment at $VENV_ROOT"
if [[ ! -x "$VENV_ROOT/bin/python" ]]; then
    python -m venv --copies "$VENV_ROOT"
fi

if ! "$VENV_ROOT/bin/python" -c 'import torch, langchain, langgraph, sentence_transformers, zstandard' >/dev/null 2>&1; then
    step "Installing pinned evaluation dependencies (persistent across containers)"
    "$VENV_ROOT/bin/python" -m pip install --upgrade 'pip<25' setuptools wheel
    "$VENV_ROOT/bin/python" -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r "$REMEMBR_ROOT/requirements-eval.txt"
fi

step "Verifying Python runtime"
"$VENV_ROOT/bin/python" -c 'import torch, langchain, langgraph, sentence_transformers; print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} langchain={langchain.__version__}")'

OLLAMA_BIN="$OLLAMA_ROOT/bin/ollama"
if [[ ! -x "$OLLAMA_BIN" ]]; then
    step "Installing Ollama in shared home"
    ollama_archive="$DOWNLOAD_CACHE/ollama-linux-amd64.tar.zst"
    curl --fail --location --retry 5 --retry-all-errors \
        --continue-at - \
        --output "$ollama_archive" \
        https://ollama.com/download/ollama-linux-amd64.tar.zst
    mkdir -p "$OLLAMA_ROOT"
    "$VENV_ROOT/bin/python" -c \
        'import sys, zstandard; zstandard.ZstdDecompressor().copy_stream(sys.stdin.buffer, sys.stdout.buffer)' \
        < "$ollama_archive" | tar -xf - -C "$OLLAMA_ROOT"
    rm -f -- "$ollama_archive"
fi

mkdir -p /hpc2hdd/home/yichenwang/.local/bin
ln -sfn "$OLLAMA_BIN" /hpc2hdd/home/yichenwang/.local/bin/ollama

step "Starting Ollama at $OLLAMA_HOST"
"$OLLAMA_BIN" serve >"$LOG_DIR/ollama.log" 2>&1 &
OLLAMA_PID=$!

for _ in $(seq 1 60); do
    if curl --noproxy '*' --silent --fail --connect-timeout 2 --max-time 5 \
        "http://$OLLAMA_HOST/api/tags" >/dev/null; then
        break
    fi
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        step "Ollama exited during startup"
        tail -100 "$LOG_DIR/ollama.log"
        exit 1
    fi
    sleep 2
done
curl --noproxy '*' --silent --fail --connect-timeout 2 --max-time 5 \
    "http://$OLLAMA_HOST/api/tags" >/dev/null

if ! "$OLLAMA_BIN" list | awk 'NR > 1 {print $1}' | grep -Fxq "$MODEL"; then
    step "Pulling $MODEL into persistent model storage"
    "$OLLAMA_BIN" pull "$MODEL"
fi

step "Warming up $MODEL on the A800"
warmup_prompt="Reply with only OK"
if [[ "$DISABLE_THINKING" == 1 ]]; then
    warmup_prompt="$warmup_prompt /no_think"
fi
curl --noproxy '*' --silent --fail "http://$OLLAMA_HOST/api/generate" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"$warmup_prompt\",\"stream\":false,\"keep_alive\":\"30m\",\"options\":{\"num_ctx\":32768,\"num_predict\":32,\"temperature\":0}}" \
    >"$LOG_DIR/qwen_warmup.json"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

run_eval() {
    local postfix=$1
    local question_limit=$2
    local -a limit_args=()
    local -a thinking_args=()
    if [[ "$question_limit" != all ]]; then
        limit_args=(--max_questions "$question_limit")
    elif [[ -n "$QUESTION_INDICES" ]]; then
        limit_args=(--question_indices "$QUESTION_INDICES")
    fi
    if [[ "$DISABLE_THINKING" == 1 ]]; then
        thinking_args=(--disable_thinking)
    fi

    step "Running sequence $SEQUENCE_ID evaluation ($postfix)"
    cd "$REMEMBR_ROOT/scripts"
    "$VENV_ROOT/bin/python" -u eval.py \
        --sequence_id "$SEQUENCE_ID" \
        --model "remembr+$MODEL" \
        --memory_backend local \
        --embedding_model "$EMBEDDING_MODEL" \
        --captions_dir "$INPUT_ARTIFACT_ROOT/captions" \
        --questions_dir "$INPUT_ARTIFACT_ROOT/questions" \
        --coda_dir "$PROJECT_ROOT/CODa" \
        --out_dir "$OUT_DIR" \
        --temperature 0 \
        --num_ctx 32768 \
        --num_predict "$NUM_PREDICT" \
        --max_retries 2 \
        --resume \
        --postfix "_$postfix" \
        "${thinking_args[@]}" \
        "${limit_args[@]}"
}

if [[ "$SMOKE_EVAL" == 1 ]]; then
    run_eval "$SMOKE_TAG" 1
fi

if [[ "$FULL_EVAL" == 1 ]]; then
    run_eval "$EVAL_TAG" all
fi

step "NaVQA evaluation completed successfully"
