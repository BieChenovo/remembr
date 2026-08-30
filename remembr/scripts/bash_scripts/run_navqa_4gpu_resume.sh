#!/usr/bin/env bash
# Finish the fixed 256-token NaVQA run using four GPU-pinned Ollama servers.
# Two evaluation workers share each server to overlap CPU retrieval with GPU work.

set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/hpc2hdd/JH_DATA/jhai_data/yichenwang/projects/remembr}
REMEMBR_ROOT="$PROJECT_ROOT/remembr"
RUNTIME_ROOT=${RUNTIME_ROOT:-/hpc2ssd/JH_DATA/spooler/yichenwang/projects/remembr/artifacts}
INPUT_ARTIFACT_ROOT=${INPUT_ARTIFACT_ROOT:-$PROJECT_ROOT/artifacts}
VENV_ROOT=${VENV_ROOT:-/hpc2hdd/home/yichenwang/.venvs/remembr-eval}
OLLAMA_ROOT=${OLLAMA_ROOT:-/hpc2hdd/home/yichenwang/.local/opt/ollama}
OLLAMA_MODELS=${OLLAMA_MODELS:-/hpc2hdd/home/yichenwang/.local/share/ollama/models}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-/hpc2hdd/home/yichenwang/.cache/huggingface/hub/models--mixedbread-ai--mxbai-embed-large-v1/snapshots/b33106f585b9ce46904ad7443a3b52b7a63e231c}
MODEL=${MODEL:-qwen3:8b}
SEQUENCES=${SEQUENCES:-"0 3 4 6 16 21 22"}
SOURCE_TAG=${SOURCE_TAG:-full_nothink_256_pst_descriptive_v1}
TARGET_TAG=${TARGET_TAG:-full_nothink_256_pst_descriptive_4gpu_resume_v1}
GPU_COUNT=${GPU_COUNT:-4}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
BASE_PORT=${BASE_PORT:-11434}
NAVQA_TIMEZONE=${NAVQA_TIMEZONE:-America/Los_Angeles}

PYTHON="$VENV_ROOT/bin/python"
OLLAMA_BIN="$OLLAMA_ROOT/bin/ollama"
WORKER_COUNT=$((GPU_COUNT * WORKERS_PER_GPU))
BATCH_DIR="$RUNTIME_ROOT/batches/$TARGET_TAG"
LOG_DIR="$BATCH_DIR/logs"
MANIFEST="$BATCH_DIR/shards.tsv"
PLAN_JSON="$BATCH_DIR/plan.json"
SUMMARY="$BATCH_DIR/summary.tsv"
REPORT_DIR="$RUNTIME_ROOT/eval_reports/$TARGET_TAG"
PLANNER="$REMEMBR_ROOT/scripts/merge_navqa_parallel.py"
mkdir -p "$LOG_DIR" "$REPORT_DIR"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

step() {
    printf '[%s] %s\n' "$(timestamp)" "$*"
}

server_pids=()
worker_pids=()
cleanup() {
    local pid
    for pid in "${worker_pids[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${server_pids[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${server_pids[@]:-}"; do
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

[[ -x "$PYTHON" ]] || {
    printf 'Missing prepared Python environment: %s\n' "$PYTHON" >&2
    exit 1
}
[[ -x "$OLLAMA_BIN" ]] || {
    printf 'Missing Ollama binary: %s\n' "$OLLAMA_BIN" >&2
    exit 1
}
[[ -d "$OLLAMA_MODELS" ]] || {
    printf 'Missing Ollama model directory: %s\n' "$OLLAMA_MODELS" >&2
    exit 1
}

mapfile -t visible_gpus < <(
    nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null
)
if ((${#visible_gpus[@]} < GPU_COUNT)); then
    printf 'Expected at least %d visible GPUs, found %d\n' \
        "$GPU_COUNT" "${#visible_gpus[@]}" >&2
    printf '%s\n' "${visible_gpus[@]:-no nvidia-smi output}" >&2
    exit 1
fi
step "Visible GPUs (${#visible_gpus[@]}): ${visible_gpus[*]}"

export OLLAMA_MODELS
export TZ="$NAVQA_TIMEZONE"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT:$REMEMBR_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"

step "Planning missing and failed questions across $WORKER_COUNT workers"
"$PYTHON" "$PLANNER" plan \
    --question-root "$INPUT_ARTIFACT_ROOT/questions" \
    --result-root "$RUNTIME_ROOT/eval_outs" \
    --source-tag "$SOURCE_TAG" \
    --target-tag "$TARGET_TAG" \
    --model "$MODEL" \
    --sequences "$SEQUENCES" \
    --manifest "$MANIFEST" \
    --plan-json "$PLAN_JSON" \
    --workers "$WORKER_COUNT" \
    --gpus "$GPU_COUNT" \
    --base-port "$BASE_PORT"

scheduled=$("$PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["questions_scheduled"])' \
    "$PLAN_JSON")
if [[ "$scheduled" == 0 ]]; then
    step "No questions require inference; merging reusable results only"
else
    step "Starting $GPU_COUNT GPU-pinned Ollama servers"
    for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
        port=$((BASE_PORT + gpu))
        env \
            CUDA_VISIBLE_DEVICES="$gpu" \
            OLLAMA_HOST="127.0.0.1:$port" \
            OLLAMA_MODELS="$OLLAMA_MODELS" \
            OLLAMA_NUM_PARALLEL="$WORKERS_PER_GPU" \
            OLLAMA_MAX_LOADED_MODELS=1 \
            OLLAMA_FLASH_ATTENTION=1 \
            "$OLLAMA_BIN" serve >"$LOG_DIR/ollama_gpu${gpu}.log" 2>&1 &
        server_pids+=("$!")
    done

    for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
        port=$((BASE_PORT + gpu))
        ready=0
        for _ in $(seq 1 90); do
            if curl --noproxy '*' --silent --fail --connect-timeout 2 --max-time 5 \
                "http://127.0.0.1:$port/api/tags" >/dev/null; then
                ready=1
                break
            fi
            sleep 2
        done
        if [[ "$ready" != 1 ]]; then
            step "Ollama on GPU $gpu did not become ready"
            tail -100 "$LOG_DIR/ollama_gpu${gpu}.log" || true
            exit 1
        fi
    done

    step "Warming Qwen on all four GPUs"
    warmup_pids=()
    for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
        port=$((BASE_PORT + gpu))
        curl --noproxy '*' --silent --show-error --fail \
            "http://127.0.0.1:$port/api/generate" \
            -H 'Content-Type: application/json' \
            -d "{\"model\":\"$MODEL\",\"prompt\":\"Reply only OK /no_think\",\"stream\":false,\"keep_alive\":\"30m\",\"options\":{\"num_ctx\":32768,\"num_predict\":16,\"temperature\":0}}" \
            >"$LOG_DIR/warmup_gpu${gpu}.json" &
        warmup_pids+=("$!")
    done
    for pid in "${warmup_pids[@]}"; do
        wait "$pid"
    done

    run_worker() {
        local worker=$1
        local gpu=$((worker % GPU_COUNT))
        local port=$((BASE_PORT + gpu))
        local found=0
        while IFS=$'\t' read -r row_worker row_gpu row_port sequence indices shard_tag; do
            [[ "$row_worker" == "$worker" ]] || continue
            found=1
            printf '[%s] worker=%s gpu=%s sequence=%s indices=%s\n' \
                "$(timestamp)" "$worker" "$gpu" "$sequence" "$indices"
            cd "$REMEMBR_ROOT/scripts"
            OLLAMA_HOST="127.0.0.1:$port" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
                "$PYTHON" -u eval.py \
                    --sequence_id "$sequence" \
                    --model "remembr+$MODEL" \
                    --memory_backend local \
                    --embedding_model "$EMBEDDING_MODEL" \
                    --captions_dir "$INPUT_ARTIFACT_ROOT/captions" \
                    --questions_dir "$INPUT_ARTIFACT_ROOT/questions" \
                    --coda_dir "$PROJECT_ROOT/CODa" \
                    --timezone "$NAVQA_TIMEZONE" \
                    --out_dir "$RUNTIME_ROOT/eval_outs" \
                    --temperature 0 \
                    --num_ctx 32768 \
                    --num_predict 256 \
                    --disable_thinking \
                    --text_judge_model "$MODEL" \
                    --text_judge_host "127.0.0.1:$port" \
                    --text_judge_num_predict 96 \
                    --max_retries 2 \
                    --resume \
                    --question_indices "$indices" \
                    --postfix "_$shard_tag"
        done < <(tail -n +2 "$MANIFEST")
        if [[ "$found" != 1 ]]; then
            printf '[%s] worker=%s has no scheduled questions\n' \
                "$(timestamp)" "$worker"
        fi
    }

    step "Launching $WORKER_COUNT evaluation workers"
    for ((worker = 0; worker < WORKER_COUNT; worker++)); do
        run_worker "$worker" >"$LOG_DIR/worker_${worker}.log" 2>&1 &
        worker_pids+=("$!")
    done

    worker_rc=0
    for index in "${!worker_pids[@]}"; do
        if ! wait "${worker_pids[$index]}"; then
            step "Worker $index failed; see $LOG_DIR/worker_${index}.log"
            worker_rc=1
        else
            step "Worker $index completed"
        fi
    done
    worker_pids=()
    if [[ "$worker_rc" != 0 ]]; then
        exit 1
    fi
fi

step "Merging shards into complete seven-sequence results"
"$PYTHON" "$PLANNER" merge \
    --question-root "$INPUT_ARTIFACT_ROOT/questions" \
    --result-root "$RUNTIME_ROOT/eval_outs" \
    --source-tag "$SOURCE_TAG" \
    --target-tag "$TARGET_TAG" \
    --model "$MODEL" \
    --sequences "$SEQUENCES" \
    --manifest "$MANIFEST" \
    --summary "$SUMMARY"

step "Generating per-sequence HTML reports"
for sequence in $SEQUENCES; do
    result="$RUNTIME_ROOT/eval_outs/$sequence/human_qa/remembr+${MODEL}__captions_VILA1.5-13b_3_secs_${TARGET_TAG}.json"
    "$PYTHON" "$REMEMBR_ROOT/scripts/visualize_eval_report.py" \
        --result "$result" \
        --questions "$INPUT_ARTIFACT_ROOT/questions/$sequence/human_qa.json" \
        --output "$REPORT_DIR/sequence_${sequence}.html"
done

step "Four-GPU NaVQA resume completed"
step "Summary: $SUMMARY"
step "Reports: $REPORT_DIR"
