# 제주대학교 규정 RAG 시스템

Context-Augmented Chunking 기반 한국어 규정 질의응답(RAG) 시스템이다. 제주대학교 규정집 전체를 조항 단위로 구조화한 뒤, **BGE-M3 dense + BM25 hybrid 검색**으로 질문에 맞는 근거 조항을 찾고, **EXAONE-3.5-7.8B-Instruct**가 검색된 근거 조항만 사용해 한국어 답변을 생성한다. 양자화(FP16/INT8/INT4)에 따른 품질·속도·메모리도 비교한다.

- 규정 파일 275개 수집 → 조항 청크 5,317개 → FAISS 검색 인덱스 + BM25 인덱스
- 검색 평가 500문항, 생성 평가 100문항 (17개 카테고리·207개 규정 포괄)

---

## 1. 디렉터리 구조

| 경로 | 설명 |
| --- | --- |
| `main.py` | 검색·답변 통합 실행 스크립트 (CLI) |
| `viz.py` | Streamlit 데모 대시보드 |
| `requirements.txt` | 의존성 (python==3.10) |
| `data/raw_hwp/` | 원본 규정 파일 275개 (HWP/PDF) |
| `data/text/` | 추출 텍스트 275개 |
| `data/processed/` | 조항 코퍼스 — `articles.jsonl`(baseline), `articles_contextual.jsonl`(맥락 보강), 각 5,317조 |
| `data/index/` | Baseline dense FAISS 인덱스 |
| `data/index_contextual/` | Contextual dense FAISS + BM25 인덱스 |
| `data/eval/` | 평가셋 `qa_seed_v2.jsonl`(검색 500), `qa_gen_v2.jsonl`(생성 100) + 평가 결과 |
| `data/metadata/` | 수집·추출·청킹·인덱스·평가 리포트(JSON) |
| `scripts/` | 데이터 수집·구축·평가 파이프라인 스크립트 |

---

## 2. 환경 설정

Python 3.10 conda 환경을 권장한다.

```bash
conda create -n myenv python=3.10
conda activate myenv
pip install -r requirements.txt
```

`requirements.txt`는 CUDA 12.1용 `torch==2.5.1+cu121`을 설치한다(GPU 환경 기준). CPU만 사용한다면 torch는 CPU 빌드로 대체해도 검색은 동작한다.

---

## 3. 모델 준비

| 용도 | 모델 | 비고 |
| --- | --- | --- |
| 임베딩(검색) | `BAAI/bge-m3` | 약 2GB |
| 답변 생성 | `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct` | 약 15GB |

첫 실행 시 HuggingFace Hub에서 자동 다운로드된다. 캐시 위치를 지정해 두는 것을 권장한다.

```bash
export HF_HOME="$HOME/.cache/huggingface"
```

- **검색만** 쓸 경우 GPU 불필요(CPU로 동작).
- **EXAONE 답변 생성**은 GPU 필요 — INT4 ≈ 9GB, INT8 ≈ 12GB, FP16 ≈ 18GB VRAM.
- 오프라인 환경에서는 모델을 미리 받아둔 뒤 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 환경변수와 `--local-files-only` 옵션을 사용한다.

---

## 4. 빠른 시작

### (a) 검색만 — CPU 가능

```bash
python main.py "외국인 유학생은 학생생활관 입주 지원을 받을 수 있나요?" \
  --retrieval-mode hybrid \
  --index data/index_contextual/faiss.index \
  --metadata data/index_contextual/faiss_metadata.jsonl \
  --bm25-index data/index_contextual/bm25.json \
  --dense-weight 0.9 --candidate-k 50 \
  --device cpu
```

### (b) EXAONE 답변 생성 — GPU 필요

```bash
python main.py "장학금은 어떤 학생에게 지급하나요?" \
  --retrieval-mode hybrid \
  --index data/index_contextual/faiss.index \
  --metadata data/index_contextual/faiss_metadata.jsonl \
  --bm25-index data/index_contextual/bm25.json \
  --llm-model LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
  --quantization fp16 \
  --device cuda
```

`--quantization`은 `fp16`(기본)·`int8`·`int4` 중 선택한다.

### (c) 데모 대시보드 (Streamlit)

```bash
bash scripts/run_viz.sh              # 검색 전용 (CUDA 숨김, CPU)
USE_GPU=1 bash scripts/run_viz.sh    # EXAONE 생성 (빈 GPU 자동 선택)
PORT=8502 bash scripts/run_viz.sh    # 포트 변경
```

대시보드는 사이드바에서 검색 방식(Baseline Dense / Contextual Dense / Contextual Hybrid)과 EXAONE 생성 토글을 고를 수 있고, 화면에는 답변·검색 후보·근거 조항을 표시한다.

> `scripts/run_*.sh`는 작성자 서버 환경(conda `myenv` 절대경로, GPU 안전 래퍼)에 맞춰져 있다. 다른 환경에서는 스크립트 상단의 Python 경로·캐시 변수를 조정하거나 위 (a)/(b)의 직접 실행 명령을 사용한다.

---

## 5. 데이터 파이프라인 재현

```bash
python scripts/collect_rules.py                 # 규정 파일 수집 → data/raw_hwp/
python scripts/extract_hwp_text.py              # HWP/PDF → data/text/
python scripts/chunk_articles.py                # 제N조 단위 청킹 → data/processed/articles.jsonl
python scripts/build_faiss_index.py --device cpu        # Baseline FAISS
bash scripts/run_build_contextual_index.sh      # Contextual 청크 + FAISS + BM25
python scripts/build_eval_set.py --new 401 --seed 42    # 평가 후보 + 라벨 추출
python scripts/merge_eval_set.py                # 검색 500 / 생성 100 평가셋 생성
```

---

## 6. 평가 재현

### 검색 평가 (500문항)

```bash
python scripts/evaluate_rag.py \
  --qa data/eval/qa_seed_v2.jsonl \
  --index data/index_contextual/faiss.index \
  --metadata data/index_contextual/faiss_metadata.jsonl \
  --bm25-index data/index_contextual/bm25.json \
  --retrieval-mode hybrid --dense-weight 0.9 \
  --device cpu --top-k 5
```

### 생성 평가 (100문항, GPU)

```bash
python scripts/evaluate_rag.py \
  --qa data/eval/qa_gen_v2.jsonl \
  --index data/index_contextual/faiss.index \
  --metadata data/index_contextual/faiss_metadata.jsonl \
  --bm25-index data/index_contextual/bm25.json \
  --retrieval-mode hybrid --dense-weight 0.9 \
  --llm-model LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
  --quantization fp16 --device cuda --top-k 5
```

---

## 7. 성능 요약

### 검색 (500 QA)

| 검색 방식 | Hit@1 | Hit@3 | Hit@5 | MRR |
| --- | --- | --- | --- | --- |
| Baseline Dense | 0.900 | 0.988 | 0.998 | 0.9444 |
| Contextual Dense | 0.902 | 0.986 | 0.998 | 0.9435 |
| **Contextual Hybrid** (dense 0.9) | **0.906** | **0.992** | 0.998 | **0.9478** |

### 양자화별 생성 (100 QA, Contextual Hybrid 검색)

| 양자화 | 근거 인용 정확도 | 답변 통과율 | 평균 생성시간(s) | tokens/sec | VRAM(GB) |
| --- | --- | --- | --- | --- | --- |
| FP16 | 0.95 | 0.81 | 3.37 | 25.4 | 17.1 |
| INT8 | 0.92 | 0.77 | 11.74 | 7.8 | 11.7 |
| INT4 | 0.84 | 0.68 | 4.24 | 20.4 | 9.0 |

요약: Contextual Hybrid 검색이 가장 우수하며, FP16이 가장 안정적인 답변 품질을, INT4가 가장 낮은 VRAM을 제공한다(품질은 다소 하락).

---

## 8. GPU 안전 규칙 (공용 서버)

- `USE_GPU=1`로 실행하면 `scripts/select_free_gpu.py`가 **여유 메모리가 가장 많은 GPU**를 자동 선택한다.
- GPU 작업 프로세스는 `scripts/nh` 래퍼를 통해 `nvidia-smi`에서 프로세스명이 `neahyuk`으로 표시된다.
- 특정 GPU를 직접 지정하려면 비어 있는지 확인한 뒤 `GPU_ID`를 준다.

```bash
USE_GPU=1 GPU_ID=4 bash scripts/run_viz.sh
```
