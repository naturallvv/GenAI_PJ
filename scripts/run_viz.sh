#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/wq1880/Gen AI"
PYTHON="/home/wq1880/miniconda3/envs/myenv/bin/python"
NH="${PROJECT_DIR}/scripts/nh"
PORT="${PORT:-8501}"
USE_GPU="${USE_GPU:-0}"
GPU_ID="${GPU_ID:-}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python not found: ${PYTHON}" >&2
  echo "Activate or create myenv first." >&2
  exit 1
fi

cd "${PROJECT_DIR}"

if [[ "${USE_GPU}" == "1" ]]; then
  if [[ -z "${GPU_ID}" ]]; then
    GPU_ID="$("${NH}" scripts/select_free_gpu.py)"
  fi
  echo "[viz] GPU_ID=${GPU_ID}"
  echo "[viz] PROCESS_NAME=neahyuk"
  echo "[viz] nvidia-smi before loading dashboard"
  nvidia-smi
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${NH}" -m streamlit run viz.py \
    --server.address 0.0.0.0 \
    --server.port "${PORT}" \
    --browser.gatherUsageStats false
else
  echo "[viz] USE_GPU=0; dashboard starts with CUDA hidden."
  CUDA_VISIBLE_DEVICES="" "${PYTHON}" -m streamlit run viz.py \
    --server.address 0.0.0.0 \
    --server.port "${PORT}" \
    --browser.gatherUsageStats false
fi
