# Jeju National University Rules RAG Data

This project collects Jeju National University rule documents and prepares article-level JSONL data for a RAG pipeline.

## Environment

Use the existing conda environment:

```bash
source /home/wq1880/miniconda3/bin/activate
conda activate myenv
```

## Pipeline

```bash
python scripts/collect_rules.py
python scripts/extract_hwp_text.py
python scripts/chunk_articles.py
python scripts/build_faiss_index.py --device cpu
bash scripts/run_build_contextual_index.sh
bash scripts/run_exaone_fp16.sh "장학금 지급 기준은 어디에 있나요?"
```

Outputs:

- `data/raw_hwp/`: original HWP files
- `data/text/`: extracted text files
- `data/metadata/rules_manifest.json`: download metadata
- `data/metadata/extract_report.json`: extraction report
- `data/metadata/chunk_report.json`: article chunking report
- `data/processed/articles.jsonl`: article-level corpus
- `data/processed/articles_contextual.jsonl`: context-augmented article corpus
- `data/index/faiss.index`: FAISS dense retrieval index
- `data/index/faiss_metadata.jsonl`: metadata aligned with FAISS ids
- `data/index_contextual/faiss.index`: context-augmented FAISS dense index
- `data/index_contextual/bm25.json`: context-augmented BM25 index
- `viz.py`: Streamlit demo dashboard

## Context-Augmented Retrieval

Build context-augmented chunks, dense index, and BM25 index:

```bash
bash scripts/run_build_contextual_index.sh
```

Run a contextual hybrid retrieval query without EXAONE generation:

```bash
python main.py "외국인 유학생의 경우 기숙사를 제공받을 수 있나요?" \
  --device cpu \
  --embedding-cache-folder /home/wq1880/.cache/huggingface \
  --local-files-only \
  --index data/index_contextual/faiss.index \
  --metadata data/index_contextual/faiss_metadata.jsonl \
  --bm25-index data/index_contextual/bm25.json \
  --retrieval-mode hybrid \
  --dense-weight 0.9 \
  --candidate-k 50
```

Contextual chunk behavior:

- `embedding_text` starts with the original article and appends document context.
- `display_text` and `text` contain the original article used for evidence and LLM prompts.
- Document context includes rule name, category, chapter/section, article metadata, and purpose summary.

Retrieval comparison on the 100-question seed set:

| method | Hit@1 | Hit@3 | Hit@5 | MRR |
| --- | --- | --- | --- | --- |
| Baseline Dense | 0.88 | 0.97 | 1.00 | 0.9292 |
| Contextual Dense | 0.88 | 0.96 | 1.00 | 0.9262 |
| Contextual Hybrid `dense_weight=0.9` | 0.88 | 0.98 | 1.00 | 0.9300 |

Comparison files:

- `data/metadata/retrieval_comparison_contextual.json`
- `data/metadata/retrieval_comparison_contextual.md`

## RAG Runner

Retrieval-only answer:

```bash
python main.py "학칙에서 장학금은 어떻게 규정되어 있나요?"
```

With EXAONE FP16 and the model cache stored on Ceph:

```bash
bash scripts/run_exaone_fp16.sh "학칙에서 장학금은 어떻게 규정되어 있나요?"
```

GPU safety rule for LLM inference:

- If `GPU_ID` is not set, the runner checks `nvidia-smi` and selects the first free GPU.
- A GPU is treated as free only when it has no compute process, low memory use, and low utilization.
- The model process is launched through `scripts/nh`, so `nvidia-smi` shows the process name as `neahyuk`.
- If no free GPU is found, the runner exits before loading EXAONE.
- Training jobs are different: before any fine-tuning or long-running training starts, check `nvidia-smi` and confirm the GPU id/count with the user first.

Use a specific GPU only after checking it is free:

```bash
GPU_ID=8 TOP_K=5 bash scripts/run_exaone_fp16.sh "복수전공 신청 기준은 무엇인가요?"
```

## Evaluation

Retrieval evaluation with the 100-question seed set:

```bash
HF_HOME=/home/wq1880/.cache/huggingface \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python scripts/evaluate_rag.py \
  --device cpu \
  --top-k 5 \
  --local-files-only \
  --embedding-cache-folder /home/wq1880/.cache/huggingface
```

Current retrieval result:

- Hit@1: `0.88`
- Hit@3: `0.97`
- Hit@5: `1.00`
- MRR: `0.9292`
- Result files:
  - `data/eval/eval_results_retrieval_100.jsonl`
  - `data/metadata/eval_report_retrieval_100.json`
  - `data/eval/eval_results_retrieval_25.jsonl`
  - `data/metadata/eval_report_retrieval_25.json`

Contextual retrieval evaluation:

```bash
bash scripts/run_eval_contextual_retrieval.sh
```

EXAONE FP16 answer evaluation result on 25 QA pairs:

- Citation accuracy: `0.88`
- Keyword recall: `0.898`
- Answer pass rate: `0.84`
- Average generation time: `4.00156` seconds
- Max GPU memory allocated: `17.398` GB
- Result files:
  - `data/eval/eval_results_exaone_fp16_25.jsonl`
  - `data/metadata/eval_report_exaone_fp16_25.json`

Quantized EXAONE evaluation:

```bash
QUANTIZATION=int8 EVAL_LIMIT=25 MAX_NEW_TOKENS=256 bash scripts/run_eval_exaone_quant.sh
QUANTIZATION=int4 EVAL_LIMIT=25 MAX_NEW_TOKENS=256 bash scripts/run_eval_exaone_quant.sh
python scripts/compare_quantization_reports.py
```

Quantization comparison on 25 QA pairs:

| quantization | qa_count | Hit@1 | Hit@5 | MRR | citation accuracy | keyword recall | answer pass rate | avg generation seconds | tokens/sec | max GPU memory GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP16 | 25 | 0.88 | 1.00 | 0.94 | 0.88 | 0.898 | 0.84 | 4.0016 | 33.5938 | 17.398 |
| INT8 | 25 | 0.88 | 1.00 | 0.94 | 0.88 | 0.896 | 0.88 | 7.4312 | 19.1897 | 11.746 |
| INT4 | 25 | 0.88 | 1.00 | 0.94 | 0.80 | 0.870 | 0.76 | 2.5760 | 50.9247 | 8.996 |

Comparison files:

- `data/metadata/quantization_comparison.json`
- `data/metadata/quantization_comparison.md`

## Demo Dashboard

Install dashboard dependencies if needed:

```bash
pip install -r requirements.txt
```

Run the dashboard in retrieval-only mode:

```bash
bash scripts/run_viz.sh
```

The default dashboard setting hides CUDA, so accidental GPU model loading is blocked.
Use the sidebar search mode selector to compare `Baseline Dense`, `Contextual Dense`, and `Contextual Hybrid`.

For GPU generation, launch Streamlit through the GPU-safe wrapper:

```bash
USE_GPU=1 bash scripts/run_viz.sh
```

Use a specific GPU only after checking it is free:

```bash
USE_GPU=1 GPU_ID=8 bash scripts/run_viz.sh
```

Use another port if needed:

```bash
PORT=8502 bash scripts/run_viz.sh
```

See `NEXT_STEPS.md` for the final submission checklist.
