#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/wq1880/Gen AI"
PYTHON="/home/wq1880/miniconda3/envs/myenv/bin/python"
NH="${PROJECT_DIR}/scripts/nh"
HF_HOME_DIR="/ceph_data/wq1880/Gen_AI/hf_cache"
MODEL_ID="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
MODEL_MARKER="${HF_HOME_DIR}/.exaone-3.5-7.8b-instruct.downloaded"

GPU_ID="${GPU_ID:-}"
TOP_K="${TOP_K:-5}"
EVAL_LIMIT="${EVAL_LIMIT:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
QUANTIZATION="${QUANTIZATION:-int8}"

if [[ "${QUANTIZATION}" != "fp16" && "${QUANTIZATION}" != "int8" && "${QUANTIZATION}" != "int4" ]]; then
  echo "QUANTIZATION must be one of: fp16, int8, int4" >&2
  exit 1
fi

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python not found: ${PYTHON}" >&2
  echo "Activate or create myenv first." >&2
  exit 1
fi

mkdir -p "${HF_HOME_DIR}"

export HF_HOME="${HF_HOME_DIR}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${PROJECT_DIR}"

if [[ -z "${GPU_ID}" ]]; then
  GPU_ID="$("${NH}" scripts/select_free_gpu.py)"
fi

if [[ -f "${MODEL_MARKER}" ]]; then
  echo "[cache] EXAONE cache marker exists. Skipping model download."
else
  echo "[cache] Missing marker: ${MODEL_MARKER}" >&2
  echo "[cache] Run scripts/run_exaone_fp16.sh once with network access to populate the cache." >&2
  exit 1
fi

if [[ "${EVAL_LIMIT}" == "0" ]]; then
  RESULT_SUFFIX="${QUANTIZATION}_all"
else
  RESULT_SUFFIX="${QUANTIZATION}_${EVAL_LIMIT}"
fi

echo "[eval] GPU_ID=${GPU_ID}"
echo "[eval] PROCESS_NAME=neahyuk"
echo "[eval] TOP_K=${TOP_K}"
echo "[eval] EVAL_LIMIT=${EVAL_LIMIT}"
echo "[eval] QUANTIZATION=${QUANTIZATION}"
echo "[eval] nvidia-smi before loading model"
nvidia-smi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${NH}" scripts/evaluate_rag.py \
  --device cuda \
  --top-k "${TOP_K}" \
  --limit "${EVAL_LIMIT}" \
  --embedding-cache-folder "${HF_HOME_DIR}" \
  --local-files-only \
  --llm-model "${MODEL_ID}" \
  --quantization "${QUANTIZATION}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output "data/eval/eval_results_exaone_${RESULT_SUFFIX}.jsonl" \
  --report "data/metadata/eval_report_exaone_${RESULT_SUFFIX}.json"
