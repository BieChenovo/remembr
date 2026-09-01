#!/usr/bin/env bash
# Run one text-retriever ablation on all seven NaVQA sequences (210 questions).

set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/hpc2hdd/JH_DATA/jhai_data/yichenwang/projects/remembr}
REMEMBR_ROOT="$PROJECT_ROOT/remembr"
INPUT_ROOT=${INPUT_ROOT:-$PROJECT_ROOT/artifacts}
RESULT_ROOT=${RESULT_ROOT:-$PROJECT_ROOT/artifacts/eval_outs}
RUNTIME_ROOT=${RUNTIME_ROOT:-/hpc2ssd/JH_DATA/spooler/yichenwang/projects/remembr/artifacts}
CODA_DIR=${CODA_DIR:-$REMEMBR_ROOT/coda_data}
VENV_ROOT=${VENV_ROOT:-/hpc2hdd/home/yichenwang/.venvs/remembr-eval}
CUDA_TORCH_ROOT=${CUDA_TORCH_ROOT:-/opt/miniconda3/envs/pytorch/lib/python3.10/site-packages}
CUDA_TORCH_SHIM=${CUDA_TORCH_SHIM:-/tmp/remembr_cuda_torch}
OLLAMA_ROOT=${OLLAMA_ROOT:-/hpc2hdd/home/yichenwang/.local/opt/ollama}
OLLAMA_MODELS=${OLLAMA_MODELS:-/hpc2hdd/home/yichenwang/.local/share/ollama/models}
MODEL=${MODEL:-qwen3:8b}
NUM_PREDICT=${NUM_PREDICT:-512}
CAPTION_FILE=${CAPTION_FILE:-captions_VILA1.5-13b_3_secs}
SEQUENCES=${SEQUENCES:-"0 3 4 6 16 21 22"}
TEXT_RETRIEVER=${TEXT_RETRIEVER:-gte_dense}
if [[ -z "${RUN_TAG:-}" ]]; then
    RUN_TAG="${TEXT_RETRIEVER}_210_question_state_v4_unified_top1"
fi
GTE_MODEL=${GTE_MODEL:-$PROJECT_ROOT/third_party/gte-models/gte-multilingual-base}
QRAG_CHECKPOINT=${QRAG_CHECKPOINT:-$PROJECT_ROOT/third_party/qrag-models/qrag-ft-gte-on-hotpotqa_musique/model_best.pt}
QRAG_INFERENCE_CHECKPOINT=${QRAG_INFERENCE_CHECKPOINT:-$PROJECT_ROOT/third_party/qrag-models/qrag-ft-gte-on-hotpotqa_musique/qrag_inference.pt}
if [[ -z "${QRAG_STEPS:-}" ]]; then
    if [[ "$RUN_TAG" == *question_state_v4_unified_top1* || "$TEXT_RETRIEVER" == qrag ]]; then
        QRAG_STEPS=1
    else
        QRAG_STEPS=5
    fi
fi
QRAG_STATE_FORMAT=${QRAG_STATE_FORMAT:-controller}
QRAG_EPISODE_MODE=${QRAG_EPISODE_MODE:-question}
QRAG_QUESTION_EVIDENCE_BUDGET=${QRAG_QUESTION_EVIDENCE_BUDGET:-5}
TEXT_EPISODE_MODE=${TEXT_EPISODE_MODE:-question}
QUESTION_TEXT_EVIDENCE_BUDGET=${QUESTION_TEXT_EVIDENCE_BUDGET:-5}
if [[ -z "${UNIFIED_EVIDENCE_LEDGER:-}" ]]; then
    if [[ "$RUN_TAG" == *question_state_v4_unified_top1* ]]; then
        UNIFIED_EVIDENCE_LEDGER=1
    else
        UNIFIED_EVIDENCE_LEDGER=0
    fi
fi
if [[ -z "${NUMERIC_K:-}" ]]; then
    if [[ "$UNIFIED_EVIDENCE_LEDGER" == 1 ]]; then
        NUMERIC_K=1
    else
        NUMERIC_K=4
    fi
fi
QUESTION_EVIDENCE_BUDGET=${QUESTION_EVIDENCE_BUDGET:-5}
MAX_RETRIEVAL_ROUNDS=${MAX_RETRIEVAL_ROUNDS:-5}
if [[ -z "${DUPLICATE_REPLAN_LIMIT:-}" ]]; then
    if [[ "$RUN_TAG" == *question_state_v4_unified_top1* ]]; then
        DUPLICATE_REPLAN_LIMIT=2
    else
        # One blocked duplicate reproduces v3's immediate reader fallback.
        DUPLICATE_REPLAN_LIMIT=1
    fi
fi
case "$RUN_TAG" in
    *question_state_v4_unified_top1*) TAG_REPORT_VERSION=v4 ;;
    *question_state_v3_interleaved*) TAG_REPORT_VERSION=v3 ;;
    *) TAG_REPORT_VERSION= ;;
esac
REPORT_VERSION=${REPORT_VERSION:-${TAG_REPORT_VERSION:-v2}}
GPU_COUNT=${GPU_COUNT:-4}
GPU_IDS=${GPU_IDS:-}
BASE_PORT=${BASE_PORT:-12434}
NAVQA_TIMEZONE=${NAVQA_TIMEZONE:-America/Los_Angeles}

PYTHON="$VENV_ROOT/bin/python"
OLLAMA_BIN="$OLLAMA_ROOT/bin/ollama"
RUN_DIR="$RUNTIME_ROOT/retriever_210/$RUN_TAG"
LOG_DIR="$RUN_DIR/logs"
STATUS_FILE="$RUN_DIR/status.txt"
CACHE_DIR=${CACHE_DIR:-$RUNTIME_ROOT/embedding_cache/$RUN_TAG}
case "$TEXT_RETRIEVER" in
    dense) REPORT_GROUP=b0 ;;
    gte_dense) REPORT_GROUP=b1 ;;
    qrag_static) REPORT_GROUP=b2 ;;
    qrag) REPORT_GROUP=b3 ;;
    *) REPORT_GROUP="$TEXT_RETRIEVER" ;;
esac
# Every implementation version owns a sibling report directory. Keep historical
# v1/v2 reports immutable and never nest v3/v4 below v2.
if [[ -z "${REPORT_DIR:-}" ]]; then
    REPORT_DIR="$PROJECT_ROOT/artifacts/eval_reports/$REPORT_GROUP/$REPORT_VERSION/sequences"
fi
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

[[ "$TEXT_RETRIEVER" == dense || "$TEXT_RETRIEVER" == gte_dense || "$TEXT_RETRIEVER" == qrag_static || "$TEXT_RETRIEVER" == qrag ]] || {
    printf 'TEXT_RETRIEVER must be dense, gte_dense, qrag_static, or qrag; got %s\n' "$TEXT_RETRIEVER" >&2
    exit 2
}
[[ "$REPORT_VERSION" =~ ^v[1-4]$ ]] || {
    printf 'REPORT_VERSION must be v1, v2, v3, or v4; got %s\n' "$REPORT_VERSION" >&2
    exit 2
}
if [[ -n "$TAG_REPORT_VERSION" && "$REPORT_VERSION" != "$TAG_REPORT_VERSION" ]]; then
    printf 'RUN_TAG %s requires REPORT_VERSION=%s; got %s\n' \
        "$RUN_TAG" "$TAG_REPORT_VERSION" "$REPORT_VERSION" >&2
    exit 2
fi
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
[[ "$UNIFIED_EVIDENCE_LEDGER" == 0 || "$UNIFIED_EVIDENCE_LEDGER" == 1 ]] || {
    printf 'UNIFIED_EVIDENCE_LEDGER must be 0 or 1; got %s\n' \
        "$UNIFIED_EVIDENCE_LEDGER" >&2
    exit 2
}
[[ "$QRAG_STEPS" == 1 || "$QRAG_STEPS" == 3 || "$QRAG_STEPS" == 5 ]] || {
    printf 'QRAG_STEPS must be 1, 3, or 5; got %s\n' "$QRAG_STEPS" >&2
    exit 2
}
if [[ "$RUN_TAG" == *question_state_v3_interleaved* ]]; then
    if [[ "$TEXT_RETRIEVER" == qrag && "$QRAG_STEPS" != 1 ]]; then
        printf 'B3 v3 interleaved requires QRAG_STEPS=1; got %s\n' \
            "$QRAG_STEPS" >&2
        exit 2
    fi
    if [[ "$TEXT_RETRIEVER" == qrag_static && "$QRAG_STEPS" != 5 ]]; then
        printf 'B2 v3 static requires QRAG_STEPS=5; got %s\n' \
            "$QRAG_STEPS" >&2
        exit 2
    fi
fi
if [[ "$RUN_TAG" == *question_state_v4_unified_top1* ]]; then
    [[ "$UNIFIED_EVIDENCE_LEDGER" == 1 ]] || {
        printf 'v4 unified top-1 requires UNIFIED_EVIDENCE_LEDGER=1\n' >&2
        exit 2
    }
    [[ "$NUMERIC_K" == 1 ]] || {
        printf 'v4 unified top-1 requires NUMERIC_K=1; got %s\n' "$NUMERIC_K" >&2
        exit 2
    }
    if [[ "$TEXT_RETRIEVER" == qrag_static || "$TEXT_RETRIEVER" == qrag ]]; then
        [[ "$QRAG_STEPS" == 1 ]] || {
            printf 'v4 unified top-1 requires QRAG_STEPS=1; got %s\n' "$QRAG_STEPS" >&2
            exit 2
        }
    fi
fi
[[ "$MAX_RETRIEVAL_ROUNDS" =~ ^[1-9][0-9]*$ ]] || {
    printf 'MAX_RETRIEVAL_ROUNDS must be positive; got %s\n' \
        "$MAX_RETRIEVAL_ROUNDS" >&2
    exit 2
}
[[ "$NUM_PREDICT" =~ ^[1-9][0-9]*$ ]] || {
    printf 'NUM_PREDICT must be positive; got %s\n' "$NUM_PREDICT" >&2
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
[[ "$NUMERIC_K" =~ ^[1-9][0-9]*$ ]] || {
    printf 'NUMERIC_K must be positive; got %s\n' "$NUMERIC_K" >&2
    exit 2
}
[[ "$QUESTION_EVIDENCE_BUDGET" =~ ^[1-9][0-9]*$ ]] || {
    printf 'QUESTION_EVIDENCE_BUDGET must be positive; got %s\n' \
        "$QUESTION_EVIDENCE_BUDGET" >&2
    exit 2
}
[[ "$DUPLICATE_REPLAN_LIMIT" =~ ^[1-9][0-9]*$ ]] || {
    printf 'DUPLICATE_REPLAN_LIMIT must be positive; got %s\n' \
        "$DUPLICATE_REPLAN_LIMIT" >&2
    exit 2
}
[[ -x "$PYTHON" ]] || { printf 'Missing Python: %s\n' "$PYTHON" >&2; exit 1; }
[[ -x "$OLLAMA_BIN" ]] || { printf 'Missing Ollama: %s\n' "$OLLAMA_BIN" >&2; exit 1; }
if [[ "$TEXT_RETRIEVER" != dense ]]; then
    [[ -d "$GTE_MODEL" ]] || { printf 'Missing GTE model: %s\n' "$GTE_MODEL" >&2; exit 1; }
fi
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

gpu_ids=()
if [[ -n "$GPU_IDS" ]]; then
    read -r -a gpu_ids <<<"$GPU_IDS"
else
    for ((slot = 0; slot < GPU_COUNT; slot++)); do
        gpu_ids+=("$slot")
    done
fi
if ((${#gpu_ids[@]} != GPU_COUNT)); then
    printf 'GPU_IDS must contain exactly %d IDs; got %d\n' \
        "$GPU_COUNT" "${#gpu_ids[@]}" >&2
    exit 1
fi
declare -A seen_gpu_ids=()
for gpu_id in "${gpu_ids[@]}"; do
    if ! [[ "$gpu_id" =~ ^[0-9]+$ ]] || ((gpu_id >= ${#visible_gpus[@]})); then
        printf 'Invalid GPU ID %s; visible GPU count is %d\n' \
            "$gpu_id" "${#visible_gpus[@]}" >&2
        exit 1
    fi
    if [[ -n "${seen_gpu_ids[$gpu_id]:-}" ]]; then
        printf 'GPU_IDS contains duplicate ID %s\n' "$gpu_id" >&2
        exit 1
    fi
    seen_gpu_ids[$gpu_id]=1
done

set_status "starting retriever=$TEXT_RETRIEVER sequences=$SEQUENCES"
step "Run tag: $RUN_TAG"
step "Retriever: $TEXT_RETRIEVER"
step "Controller: interleaved single-call, max rounds: $MAX_RETRIEVAL_ROUNDS"
step "Duplicate replans: $DUPLICATE_REPLAN_LIMIT"
step "Controller/reader output budget: $NUM_PREDICT tokens"
step "Dense/GTE text episode: $TEXT_EPISODE_MODE, question budget: $QUESTION_TEXT_EVIDENCE_BUDGET"
step "Unified evidence ledger: $UNIFIED_EVIDENCE_LEDGER, numeric k: $NUMERIC_K, global budget: $QUESTION_EVIDENCE_BUDGET"
if [[ "$TEXT_RETRIEVER" == qrag_static || "$TEXT_RETRIEVER" == qrag ]]; then
    step "Q-RAG state: $QRAG_STATE_FORMAT, episode: $QRAG_EPISODE_MODE, per-call max: $QRAG_STEPS, question budget: $QRAG_QUESTION_EVIDENCE_BUDGET"
fi
step "Visible GPUs: ${visible_gpus[*]}"
step "Selected GPU IDs: ${gpu_ids[*]}"

step "Starting $GPU_COUNT GPU-pinned Ollama servers"
for ((slot = 0; slot < GPU_COUNT; slot++)); do
    gpu=${gpu_ids[$slot]}
    port=$((BASE_PORT + slot))
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

for ((slot = 0; slot < GPU_COUNT; slot++)); do
    gpu=${gpu_ids[$slot]}
    port=$((BASE_PORT + slot))
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
for ((slot = 0; slot < GPU_COUNT; slot++)); do
    gpu=${gpu_ids[$slot]}
    port=$((BASE_PORT + slot))
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
    slot=$((ordinal % GPU_COUNT))
    gpu=${gpu_ids[$slot]}
    port=$((BASE_PORT + slot))
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
            --numeric_k "$NUMERIC_K"
            --question_evidence_budget "$QUESTION_EVIDENCE_BUDGET"
        )
        if [[ "$UNIFIED_EVIDENCE_LEDGER" == 1 ]]; then
            extra_args+=(--unified_evidence_ledger)
        fi
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
                --coda_dir "$CODA_DIR" \
                --caption_file "$CAPTION_FILE" \
                --timezone "$NAVQA_TIMEZONE" \
                --out_dir "$RESULT_ROOT" \
                --temperature 0 \
                --num_ctx 32768 \
                --num_predict "$NUM_PREDICT" \
                --disable_thinking \
                --max_retrieval_rounds "$MAX_RETRIEVAL_ROUNDS" \
                --duplicate_replan_limit "$DUPLICATE_REPLAN_LIMIT" \
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

question_count=$((30 * ${#sequence_array[@]}))
set_status "complete retriever=$TEXT_RETRIEVER questions=$question_count"
trap - ERR
step "Complete: $TEXT_RETRIEVER on $question_count questions"
step "Results: $RESULT_ROOT"
step "Reports: $REPORT_DIR"
