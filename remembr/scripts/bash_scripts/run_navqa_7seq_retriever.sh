#!/usr/bin/env bash
# Run one text-retriever ablation on all seven NaVQA sequences (210 questions).

set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/hpc2hdd/JH_DATA/jhai_data/yichenwang/projects/remembr}
REMEMBR_ROOT="$PROJECT_ROOT/remembr"
INPUT_ROOT=${INPUT_ROOT:-$PROJECT_ROOT/artifacts}
RESULT_ROOT=${RESULT_ROOT:-$PROJECT_ROOT/artifacts/eval_outs}
RUNTIME_ROOT=${RUNTIME_ROOT:-/hpc2ssd/JH_DATA/spooler/yichenwang/projects/remembr/artifacts}
VENV_ROOT=${VENV_ROOT:-/hpc2hdd/home/yichenwang/.venvs/remembr-eval}
CUDA_TORCH_ROOT=${CUDA_TORCH_ROOT:-/opt/miniconda3/envs/pytorch/lib/python3.10/site-packages}
CUDA_TORCH_SHIM=${CUDA_TORCH_SHIM:-/tmp/remembr_cuda_torch}
OLLAMA_ROOT=${OLLAMA_ROOT:-/hpc2hdd/home/yichenwang/.local/opt/ollama}
OLLAMA_MODELS=${OLLAMA_MODELS:-/hpc2hdd/home/yichenwang/.local/share/ollama/models}
MODEL=${MODEL:-qwen3:8b}
CAPTION_FILE=${CAPTION_FILE:-captions_VILA1.5-13b_3_secs}
SEQUENCES=${SEQUENCES:-"0 3 4 6 16 21 22"}
TEXT_RETRIEVER=${TEXT_RETRIEVER:-gte_dense}
if [[ -z "${RUN_TAG:-}" ]]; then
    RUN_TAG="${TEXT_RETRIEVER}_210_question_state_v2"
fi
GTE_MODEL=${GTE_MODEL:-$PROJECT_ROOT/third_party/gte-models/gte-multilingual-base}
QRAG_CHECKPOINT=${QRAG_CHECKPOINT:-$PROJECT_ROOT/third_party/qrag-models/qrag-ft-gte-on-hotpotqa_musique/model_best.pt}
QRAG_INFERENCE_CHECKPOINT=${QRAG_INFERENCE_CHECKPOINT:-$PROJECT_ROOT/third_party/qrag-models/qrag-ft-gte-on-hotpotqa_musique/qrag_inference.pt}
QRAG_STEPS=${QRAG_STEPS:-5}
QRAG_STATE_FORMAT=${QRAG_STATE_FORMAT:-controller}
QRAG_EPISODE_MODE=${QRAG_EPISODE_MODE:-question}
QRAG_QUESTION_EVIDENCE_BUDGET=${QRAG_QUESTION_EVIDENCE_BUDGET:-5}
TEXT_EPISODE_MODE=${TEXT_EPISODE_MODE:-question}
QUESTION_TEXT_EVIDENCE_BUDGET=${QUESTION_TEXT_EVIDENCE_BUDGET:-5}
GPU_COUNT=${GPU_COUNT:-4}
BASE_PORT=${BASE_PORT:-12434}
NAVQA_TIMEZONE=${NAVQA_TIMEZONE:-America/Los_Angeles}

PYTHON="$VENV_ROOT/bin/python"
OLLAMA_BIN="$OLLAMA_ROOT/bin/ollama"
RUN_DIR="$RUNTIME_ROOT/retriever_210/$RUN_TAG"
LOG_DIR="$RUN_DIR/logs"
STATUS_FILE="$RUN_DIR/status.txt"
CACHE_DIR=${CACHE_DIR:-$RUNTIME_ROOT/embedding_cache/$RUN_TAG}
REPORT_DIR=${REPORT_DIR:-$PROJECT_ROOT/artifacts/eval_reports/$RUN_TAG}
mkdir -p "$LOG_DIR" "$CACHE_DIR" "$REPORT_DIR" "$RESULT_ROOT"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

step() {
    printf '[%s] %s\n' "$(timestamp)" "$*"
}

set_status() {
    printf '%s\t%s\n' "$(timestamp)" "$*" >"$STATUS_FILE"
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
on_error() {
    local rc=$?
    set_status "failed rc=$rc"
    exit "$rc"
}
trap cleanup EXIT INT TERM
trap on_error ERR

[[ "$TEXT_RETRIEVER" == gte_dense || "$TEXT_RETRIEVER" == qrag_static || "$TEXT_RETRIEVER" == qrag ]] || {
    printf 'TEXT_RETRIEVER must be gte_dense, qrag_static, or qrag; got %s\n' "$TEXT_RETRIEVER" >&2
    exit 2
}
[[ "$TEXT_EPISODE_MODE" == per_call || "$TEXT_EPISODE_MODE" == question ]] || {
    printf 'TEXT_EPISODE_MODE must be per_call or question; got %s\n' \
        "$TEXT_EPISODE_MODE" >&2
    exit 2
}
[[ "$QRAG_STATE_FORMAT" == native || "$QRAG_STATE_FORMAT" == controller ]] || {
    printf 'QRAG_STATE_FORMAT must be native or controller; got %s\n' \
        "$QRAG_STATE_FORMAT" >&2
    exit 2
}
[[ "$QRAG_EPISODE_MODE" == per_call || "$QRAG_EPISODE_MODE" == question ]] || {
    printf 'QRAG_EPISODE_MODE must be per_call or question; got %s\n' \
        "$QRAG_EPISODE_MODE" >&2
    exit 2
}
[[ "$QRAG_STEPS" == 1 || "$QRAG_STEPS" == 3 || "$QRAG_STEPS" == 5 ]] || {
    printf 'QRAG_STEPS must be 1, 3, or 5; got %s\n' "$QRAG_STEPS" >&2
    exit 2
}
[[ "$QUESTION_TEXT_EVIDENCE_BUDGET" =~ ^[1-9][0-9]*$ ]] || {
    printf 'QUESTION_TEXT_EVIDENCE_BUDGET must be positive; got %s\n' \
        "$QUESTION_TEXT_EVIDENCE_BUDGET" >&2
    exit 2
}
[[ "$QRAG_QUESTION_EVIDENCE_BUDGET" =~ ^[1-9][0-9]*$ ]] || {
    printf 'QRAG_QUESTION_EVIDENCE_BUDGET must be positive; got %s\n' \
        "$QRAG_QUESTION_EVIDENCE_BUDGET" >&2
    exit 2
}
[[ -x "$PYTHON" ]] || { printf 'Missing Python: %s\n' "$PYTHON" >&2; exit 1; }
[[ -x "$OLLAMA_BIN" ]] || { printf 'Missing Ollama: %s\n' "$OLLAMA_BIN" >&2; exit 1; }
[[ -d "$GTE_MODEL" ]] || { printf 'Missing GTE model: %s\n' "$GTE_MODEL" >&2; exit 1; }
if [[ "$TEXT_RETRIEVER" == qrag_static || "$TEXT_RETRIEVER" == qrag ]]; then
    [[ -f "$QRAG_CHECKPOINT" ]] || {
        printf 'Missing Q-RAG checkpoint: %s\n' "$QRAG_CHECKPOINT" >&2
        exit 1
    }
    [[ -f "$QRAG_INFERENCE_CHECKPOINT" ]] || {
        printf 'Missing exported Q-RAG inference checkpoint: %s\n' \
            "$QRAG_INFERENCE_CHECKPOINT" >&2
        exit 1
    }
fi

# Reuse the CUDA torch already provisioned by the GPU image while keeping all
# other Python packages from the prepared ReMEmbR evaluation environment.
mkdir -p "$CUDA_TORCH_SHIM"
ln -sfn "$CUDA_TORCH_ROOT/torch" "$CUDA_TORCH_SHIM/torch"
ln -sfn "$CUDA_TORCH_ROOT/torch-2.3.1.dist-info" \
    "$CUDA_TORCH_SHIM/torch-2.3.1.dist-info"

export OLLAMA_MODELS
export TZ="$NAVQA_TIMEZONE"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$CUDA_TORCH_SHIM:$PROJECT_ROOT:$REMEMBR_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"

mapfile -t visible_gpus < <(
    nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null
)
if ((${#visible_gpus[@]} < GPU_COUNT)); then
    printf 'Expected %d GPUs, found %d\n' "$GPU_COUNT" "${#visible_gpus[@]}" >&2
    exit 1
fi

set_status "starting retriever=$TEXT_RETRIEVER sequences=$SEQUENCES"
step "Run tag: $RUN_TAG"
step "Retriever: $TEXT_RETRIEVER"
step "Dense/GTE text episode: $TEXT_EPISODE_MODE, question budget: $QUESTION_TEXT_EVIDENCE_BUDGET"
if [[ "$TEXT_RETRIEVER" == qrag_static || "$TEXT_RETRIEVER" == qrag ]]; then
    step "Q-RAG state: $QRAG_STATE_FORMAT, episode: $QRAG_EPISODE_MODE, per-call max: $QRAG_STEPS, question budget: $QRAG_QUESTION_EVIDENCE_BUDGET"
fi
step "Visible GPUs: ${visible_gpus[*]}"

step "Starting $GPU_COUNT GPU-pinned Ollama servers"
for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
    port=$((BASE_PORT + gpu))
    env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        OLLAMA_HOST="127.0.0.1:$port" \
        OLLAMA_MODELS="$OLLAMA_MODELS" \
        OLLAMA_NUM_PARALLEL=2 \
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

step "Warming Qwen on all GPUs"
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

read -r -a sequence_array <<<"$SEQUENCES"
set_status "running 0/${#sequence_array[@]} sequences retriever=$TEXT_RETRIEVER"
step "Launching ${#sequence_array[@]} sequence workers"
for ordinal in "${!sequence_array[@]}"; do
    sequence=${sequence_array[$ordinal]}
    gpu=$((ordinal % GPU_COUNT))
    port=$((BASE_PORT + gpu))
    (
        cd "$REMEMBR_ROOT/scripts"
        extra_args=(
            --text_retriever "$TEXT_RETRIEVER"
            --gte_model "$GTE_MODEL"
            --embedding_device cuda:0
            --embedding_batch_size 32
            --embedding_cache_dir "$CACHE_DIR"
            --text_episode_mode "$TEXT_EPISODE_MODE"
            --question_text_evidence_budget "$QUESTION_TEXT_EVIDENCE_BUDGET"
        )
        if [[ "$TEXT_RETRIEVER" == qrag_static || "$TEXT_RETRIEVER" == qrag ]]; then
            extra_args+=(
                --qrag_checkpoint "$QRAG_CHECKPOINT"
                --qrag_inference_checkpoint "$QRAG_INFERENCE_CHECKPOINT"
                --qrag_steps "$QRAG_STEPS"
                --qrag_state_format "$QRAG_STATE_FORMAT"
                --qrag_episode_mode "$QRAG_EPISODE_MODE"
                --qrag_question_evidence_budget "$QRAG_QUESTION_EVIDENCE_BUDGET"
            )
        fi
        CUDA_VISIBLE_DEVICES="$gpu" \
        OLLAMA_HOST="127.0.0.1:$port" \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
            "$PYTHON" -u eval.py \
                --sequence_id "$sequence" \
                --model "remembr+$MODEL" \
                --memory_backend local \
                --captions_dir "$INPUT_ROOT/captions" \
                --questions_dir "$INPUT_ROOT/questions" \
                --coda_dir "$PROJECT_ROOT/CODa" \
                --caption_file "$CAPTION_FILE" \
                --timezone "$NAVQA_TIMEZONE" \
                --out_dir "$RESULT_ROOT" \
                --temperature 0 \
                --num_ctx 32768 \
                --num_predict 256 \
                --disable_thinking \
                --text_judge_model "$MODEL" \
                --text_judge_host "127.0.0.1:$port" \
                --text_judge_num_predict 96 \
                --max_retries 2 \
                --resume \
                --postfix "_$RUN_TAG" \
                "${extra_args[@]}"
    ) >"$LOG_DIR/sequence_${sequence}.log" 2>&1 &
    worker_pids+=("$!")
done

worker_rc=0
completed=0
for ordinal in "${!worker_pids[@]}"; do
    sequence=${sequence_array[$ordinal]}
    if wait "${worker_pids[$ordinal]}"; then
        completed=$((completed + 1))
        set_status "running $completed/${#sequence_array[@]} sequences retriever=$TEXT_RETRIEVER"
        step "Sequence $sequence completed ($completed/${#sequence_array[@]})"
    else
        step "Sequence $sequence failed; see $LOG_DIR/sequence_${sequence}.log"
        worker_rc=1
    fi
done
worker_pids=()
if [[ "$worker_rc" != 0 ]]; then
    exit 1
fi

step "Generating seven compact HTML reports"
for sequence in "${sequence_array[@]}"; do
    result="$RESULT_ROOT/$sequence/human_qa/remembr+${MODEL}__${CAPTION_FILE}__${TEXT_RETRIEVER}_${RUN_TAG}.json"
    "$PYTHON" "$REMEMBR_ROOT/scripts/visualize_eval_report.py" \
        --result "$result" \
        --questions "$INPUT_ROOT/questions/$sequence/human_qa.json" \
        --output "$REPORT_DIR/sequence_${sequence}.html"
done

set_status "complete retriever=$TEXT_RETRIEVER questions=210"
trap - ERR
step "Complete: $TEXT_RETRIEVER on 210 questions"
step "Results: $RESULT_ROOT"
step "Reports: $REPORT_DIR"
