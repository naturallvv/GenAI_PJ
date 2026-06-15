#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/wq1880/Gen AI"
PYTHON="/home/wq1880/miniconda3/envs/myenv/bin/python"
NH="${PROJECT_DIR}/scripts/nh"

GPU_ID="${GPU_ID:-}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python not found: ${PYTHON}" >&2
  echo "Activate or create myenv first." >&2
  exit 1
fi

cd "${PROJECT_DIR}"

"${PYTHON}" scripts/build_contextual_chunks.py

if [[ -z "${GPU_ID}" ]]; then
  GPU_ID="$("${NH}" scripts/select_free_gpu.py)"
fi

echo "[index] GPU_ID=${GPU_ID}"
echo "[index] PROCESS_NAME=neahyuk"
echo "[index] nvidia-smi before embedding"
nvidia-smi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${NH}" scripts/build_faiss_index.py \
  --articles data/processed/articles_contextual.jsonl \
  --index data/index_contextual/faiss.index \
  --metadata data/index_contextual/faiss_metadata.jsonl \
  --report data/metadata/index_contextual_report.json \
  --text-field embedding_text \
  --max-seq-length "${MAX_SEQ_LENGTH}" \
  --batch-size "${BATCH_SIZE}" \
  --device cuda

"${PYTHON}" scripts/build_bm25_index.py \
  --articles data/processed/articles_contextual.jsonl \
  --output data/index_contextual/bm25.json \
  --text-field embedding_text
