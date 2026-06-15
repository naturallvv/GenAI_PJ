#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/wq1880/Gen AI"
PYTHON="/home/wq1880/miniconda3/envs/myenv/bin/python"
NH="${PROJECT_DIR}/scripts/nh"
HF_HOME_DIR="/ceph_data/wq1880/Gen_AI/hf_cache"
MODEL_ID="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
MODEL_MARKER="${HF_HOME_DIR}/.exaone-3.5-7.8b-instruct.downloaded"

GPU_ID="${GPU_ID:-}"
QUERY="${1:-학칙에서 장학금은 어떻게 규정되어 있나요?}"
TOP_K="${TOP_K:-5}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python not found: ${PYTHON}" >&2
  echo "Activate or create myenv first." >&2
  exit 1
fi

mkdir -p "${HF_HOME_DIR}"

export HF_HOME="${HF_HOME_DIR}"

cd "${PROJECT_DIR}"

if [[ -z "${GPU_ID}" ]]; then
  GPU_ID="$("${NH}" scripts/select_free_gpu.py)"
fi

if [[ -f "${MODEL_MARKER}" ]]; then
  echo "[cache] EXAONE cache marker exists. Skipping model download."
else
  echo "[cache] Downloading ${MODEL_ID} into ${HF_HOME_DIR}"
  "${PYTHON}" -m huggingface_hub.commands.huggingface_cli download "${MODEL_ID}"
  touch "${MODEL_MARKER}"
  echo "[cache] Download complete."
fi

echo "[run] GPU_ID=${GPU_ID}"
echo "[run] PROCESS_NAME=neahyuk"
echo "[run] QUERY=${QUERY}"
echo "[run] nvidia-smi before loading model"
nvidia-smi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${NH}" main.py "${QUERY}" \
  --device cuda \
  --top-k "${TOP_K}" \
  --llm-model "${MODEL_ID}" \
  --quantization fp16
