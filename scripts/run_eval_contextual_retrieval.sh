#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/wq1880/Gen AI"
PYTHON="/home/wq1880/miniconda3/envs/myenv/bin/python"
NH="${PROJECT_DIR}/scripts/nh"
EMBEDDING_CACHE="${EMBEDDING_CACHE:-/home/wq1880/.cache/huggingface}"

GPU_ID="${GPU_ID:-}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
TOP_K="${TOP_K:-5}"
RUN_RERANKER="${RUN_RERANKER:-0}"
RERANKER_MODEL="${RERANKER_MODEL:-BAAI/bge-reranker-v2-m3}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python not found: ${PYTHON}" >&2
  echo "Activate or create myenv first." >&2
  exit 1
fi

cd "${PROJECT_DIR}"

if [[ -z "${GPU_ID}" ]]; then
  GPU_ID="$("${NH}" scripts/select_free_gpu.py)"
fi

echo "[eval] GPU_ID=${GPU_ID}"
echo "[eval] PROCESS_NAME=neahyuk"
echo "[eval] TOP_K=${TOP_K}"
echo "[eval] EVAL_LIMIT=${EVAL_LIMIT}"
echo "[eval] nvidia-smi before retrieval evaluation"
nvidia-smi

COMMON_ARGS=(
  --qa data/eval/qa_seed.jsonl
  --device cuda
  --top-k "${TOP_K}"
  --limit "${EVAL_LIMIT}"
  --embedding-cache-folder "${EMBEDDING_CACHE}"
  --local-files-only
)

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${NH}" scripts/evaluate_rag.py \
  "${COMMON_ARGS[@]}" \
  --index data/index_contextual/faiss.index \
  --metadata data/index_contextual/faiss_metadata.jsonl \
  --retrieval-mode dense \
  --output data/eval/eval_results_contextual_dense_100.jsonl \
  --report data/metadata/eval_report_contextual_dense_100.json

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${NH}" scripts/evaluate_rag.py \
  "${COMMON_ARGS[@]}" \
  --index data/index_contextual/faiss.index \
  --metadata data/index_contextual/faiss_metadata.jsonl \
  --bm25-index data/index_contextual/bm25.json \
  --retrieval-mode hybrid \
  --dense-weight 0.9 \
  --candidate-k 50 \
  --output data/eval/eval_results_contextual_hybrid_100.jsonl \
  --report data/metadata/eval_report_contextual_hybrid_100.json

if [[ "${RUN_RERANKER}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${NH}" scripts/evaluate_rag.py \
    "${COMMON_ARGS[@]}" \
    --index data/index_contextual/faiss.index \
    --metadata data/index_contextual/faiss_metadata.jsonl \
    --bm25-index data/index_contextual/bm25.json \
    --retrieval-mode hybrid \
    --dense-weight 0.9 \
    --candidate-k 50 \
    --reranker-model "${RERANKER_MODEL}" \
    --output data/eval/eval_results_contextual_hybrid_reranker_100.jsonl \
    --report data/metadata/eval_report_contextual_hybrid_reranker_100.json
fi

"${PYTHON}" scripts/compare_retrieval_reports.py
