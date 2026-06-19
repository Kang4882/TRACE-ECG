#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  USER_APPROVED_FULL_TRAIN=1 bash train_full_l3_sft_gemlike.sh --full_train
  MAX_STEPS=100 bash train_full_l3_sft_gemlike.sh --preflight

Runs full-parameter row-level L3 SFT on GEM-like expanded L3 JSONL.
TRACE-ECG, TargetMargin, legacy LayoutBind, and grouped triplet training are
disabled.
EOF
}

MODE="${1:-}"
if [[ "${MODE}" != "--preflight" && "${MODE}" != "--full_train" ]]; then
  usage
  exit 1
fi
if [[ "${MODE}" == "--full_train" && "${USER_APPROVED_FULL_TRAIN:-0}" != "1" ]]; then
  echo "Full training is gated. Set USER_APPROVED_FULL_TRAIN=1."
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
INIT_CHECKPOINT="${GEM_RECIPE_INIT_CHECKPOINT:-/data/model_checkpoints/liuhaotian_llava-v1.6-vicuna-7b}"
ECG_TOWER="${ECG_TOWER:-${GEM_ROOT}/ecg_coca/open_clip/checkpoint/cpt_wfep_epoch_20.pt}"
CONDA_ENV="${CONDA_ENV:-gem}"

DATA_PATH="${DATA_PATH:-${GEMLIKE_DATA_ROOT}/gem_jsons/full/gem_train_l3_gemlike_full.jsonl}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"

if [[ "${MODE}" == "--preflight" ]]; then
  RUN_NAME="${RUN_NAME:-full_l3_sft_gemlike_preflight}"
  OUTPUT_DIR="${OUTPUT_DIR:-/data/model_checkpoints/${RUN_NAME}}"
  MAX_STEPS="${MAX_STEPS:-100}"
else
  RUN_NAME="${RUN_NAME:-full_l3_sft_gemlike}"
  OUTPUT_DIR="${OUTPUT_DIR:-/data/model_checkpoints/full_l3_sft_gemlike}"
  MAX_STEPS="${MAX_STEPS:--1}"
fi

MAX_STEPS_ARGS=()
if [[ "${MAX_STEPS}" != "-1" ]]; then
  MAX_STEPS_ARGS=(--max_steps "${MAX_STEPS}")
fi

mkdir -p "${OUTPUT_DIR}/logs"
LOG_FILE="${OUTPUT_DIR}/logs/train_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

export HF_HOME="${HF_HOME:-/data/download_cache/hf}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "RUN_NAME=${RUN_NAME}"
echo "DATA_PATH=${DATA_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "INIT_CHECKPOINT=${INIT_CHECKPOINT}"
echo "TRACE_ECG=False"
echo "INTERNAL_GEM_FLAG_PREFIX=anchor_ecg"
echo "TARGETMARGIN=False"
echo "LAYOUTBIND=False"
echo "GROUPED_TRIPLET_TRAINING=False"

cd "${GEM_ROOT}"
conda run -n "${CONDA_ENV}" --no-capture-output torchrun \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_addr "${MASTER_ADDR:-127.0.0.1}" \
  --node_rank 0 \
  --master_port "${MASTER_PORT:-14631}" \
  --nnodes 1 \
  "${GEM_ROOT}/llava/train/train_mem.py" \
  --deepspeed "${GEM_ROOT}/scripts/zero2.json" \
  --model_name_or_path "${INIT_CHECKPOINT}" \
  --version llava_v1 \
  --data_path "${DATA_PATH}" \
  --grouped_triplet_training False \
  --layoutbind_enable False \
  --anchor_ecg_enable False \
  --targetmargin_enable False \
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
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-64}" \
  --lazy_preprocess True \
  --report_to "${REPORT_TO:-wandb}" \
  --run_name "${RUN_NAME}" \
  "${MAX_STEPS_ARGS[@]}"
