#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  USER_APPROVED_FULL_TRAIN=1 bash train_trace_ecg_gemlike.sh --full_train --frac 001|003|005|010
  USER_APPROVED_FULL_TRAIN=1 MAX_STEPS=20 bash train_trace_ecg_gemlike.sh --full_train --frac 005

Runs row-level TRACE-ECG:
  CE on all rows + lambda_margin * row-wise TargetMargin on bind-eligible rows.

Legacy TargetBind/LayoutBind and grouped triplet training are disabled.
EOF
}

MODE=""
FRAC=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full_train) MODE="full_train"; shift ;;
    --frac) FRAC="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ "${MODE}" != "full_train" || "${USER_APPROVED_FULL_TRAIN:-0}" != "1" ]]; then
  echo "Training is gated. Set USER_APPROVED_FULL_TRAIN=1 and pass --full_train."
  exit 1
fi
if [[ ! "${FRAC}" =~ ^(001|003|005|010)$ ]]; then
  echo "--frac must be one of 001, 003, 005, 010"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_GEM_ROOT="$(cd "${SCRIPT_DIR}/../../GEM" 2>/dev/null && pwd || true)"
GEM_ROOT="${GEM_ROOT:-${DEFAULT_GEM_ROOT}}"
if [[ -z "${GEM_ROOT}" || ! -d "${GEM_ROOT}/llava" ]]; then
  echo "GEM_ROOT is not set or does not point to a patched GEM checkout."
  echo "Set GEM_ROOT=/path/to/GEM after running scripts/apply_gem_patch.sh."
  exit 1
fi
GEMLIKE_DATA_ROOT="${GEMLIKE_DATA_ROOT:-/data/ecg_l3_gemlike_v1}"
ECG_FOLDER="${ECG_FOLDER:-/data/ecg_timeseries}"
BASE_MODEL="${LANSG_GEM_CHECKPOINT:-/data/model_checkpoints/LANSG_GEM}"
ECG_TOWER="${ECG_TOWER:-${GEM_ROOT}/ecg_coca/open_clip/checkpoint/cpt_wfep_epoch_20.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/model_checkpoints/trace_ecg_runs}"
CONDA_ENV="${CONDA_ENV:-gem}"

DATA_PATH="${DATA_PATH:-${GEMLIKE_DATA_ROOT}/gem_jsons/fractions/gem_train_l3_gemlike_frac${FRAC}.jsonl}"
RUN_NAME="${RUN_NAME:-TRACE_ECG_L3_frac${FRAC}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
LOG_DIR="${OUTPUT_DIR}/logs"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
TRAIN_LOG_CSV="${LOG_DIR}/trace_ecg_train_log.csv"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
EFFECTIVE_GLOBAL_BATCH="${EFFECTIVE_GLOBAL_BATCH:-256}"
if (( EFFECTIVE_GLOBAL_BATCH % (NPROC_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE) != 0 )); then
  echo "EFFECTIVE_GLOBAL_BATCH must be divisible by NPROC_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE"
  exit 1
fi
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$((EFFECTIVE_GLOBAL_BATCH / (NPROC_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE)))}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

MAX_STEPS_ARGS=()
if [[ -n "${MAX_STEPS:-}" ]]; then
  MAX_STEPS_ARGS=(--max_steps "${MAX_STEPS}")
fi

export HF_HOME="${HF_HOME:-/data/download_cache/hf}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED="1"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRACE_ECG_TRAIN_LOG_CSV="${TRAIN_LOG_CSV}"
# Internal compatibility: the GEM trainer currently uses anchor_ecg_* flag names.
export ANCHOR_ECG_TRAIN_LOG_CSV="${TRAIN_LOG_CSV}"

echo "RUN_NAME=${RUN_NAME}"
echo "DATA_PATH=${DATA_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "EFFECTIVE_GLOBAL_BATCH=${EFFECTIVE_GLOBAL_BATCH}"
echo "TRACE_ECG=True"
echo "TRACE_ECG_ROWWISE=True"
echo "TRACE_ECG_MEMORY_SAFE_BACKWARD=${TRACE_ECG_MEMORY_SAFE_BACKWARD:-True}"
echo "TRACE_ECG_MARGIN_CHUNK_BIND_ROWS=${TRACE_ECG_MARGIN_CHUNK_BIND_ROWS:-2}"
echo "INTERNAL_GEM_FLAG_PREFIX=anchor_ecg"
echo "TARGETMARGIN=True"
echo "LAYOUTBIND=False"
echo "GROUPED_TRIPLET_TRAINING=False"

cd "${GEM_ROOT}"
conda run -n "${CONDA_ENV}" --no-capture-output torchrun \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_addr "${MASTER_ADDR:-127.0.0.1}" \
  --node_rank 0 \
  --master_port "${MASTER_PORT:-1277}" \
  --nnodes 1 \
  "${GEM_ROOT}/llava/train/train_mem.py" \
  --deepspeed "${GEM_ROOT}/scripts/zero2.json" \
  --model_name_or_path "${BASE_MODEL}" \
  --version llava_v1 \
  --data_path "${DATA_PATH}" \
  --grouped_triplet_training False \
  --layoutbind_enable False \
  --anchor_ecg_enable True \
  --anchor_ecg_rowwise True \
  --anchor_ecg_lambda_margin "${TRACE_ECG_LAMBDA_MARGIN:-0.05}" \
  --anchor_ecg_margin_delta "${TRACE_ECG_MARGIN_DELTA:-0.5}" \
  --anchor_ecg_length_normalize True \
  --anchor_ecg_strict_eligibility True \
  --anchor_ecg_memory_safe_backward "${TRACE_ECG_MEMORY_SAFE_BACKWARD:-True}" \
  --anchor_ecg_margin_chunk_bind_rows "${TRACE_ECG_MARGIN_CHUNK_BIND_ROWS:-2}" \
  --targetmargin_enable True \
  --targetmargin_rowwise True \
  --targetmargin_lambda "${TRACE_ECG_LAMBDA_MARGIN:-0.05}" \
  --targetmargin_delta "${TRACE_ECG_MARGIN_DELTA:-0.5}" \
  --targetmargin_length_normalize True \
  --targetmargin_strict_eligibility True \
  --ecg_folder "${ECG_FOLDER}" \
  --ecg_tower "${ECG_TOWER}" \
  --open_clip_config coca_ViT-B-32 \
  --image_folder "${GEMLIKE_DATA_ROOT}" \
  --vision_tower "${VISION_TOWER:-openai/clip-vit-large-patch14-336}" \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio ori \
  --group_by_modality_length False \
  --bf16 True \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS:-0.5}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 4096 \
  --gradient_checkpointing True \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-8}" \
  --lazy_preprocess True \
  --lora_enable True \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --pure_lora_protocol True \
  --lora_target_modules llm_qkvo_mm_projector \
  --report_to "${REPORT_TO:-wandb}" \
  --run_name "${RUN_NAME}" \
  "${MAX_STEPS_ARGS[@]}"
