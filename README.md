# 제주대학교 규정 RAG 시스템

제주대학교 규정집을 조항 단위로 검색하고, 검색된 근거 조항을 바탕으로 EXAONE-3.5 LLM이 한국어 답변을 생성하는 RAG 시스템임. (BGE-M3 dense + BM25 hybrid 검색)

## 1. 프로젝트 구성

```
.
├─ main.py               검색·답변 실행 스크립트 (CLI)
├─ viz.py                Streamlit 데모 대시보드
├─ requirements.txt      의존성 (python==3.10)
├─ data/                 데이터
│  ├─ raw_hwp/           원본 규정 파일 275개 (HWP/PDF)
│  ├─ text/              추출 텍스트
│  ├─ processed/         조항 코퍼스 (5,317조)
│  ├─ index/             Baseline FAISS 인덱스
│  ├─ index_contextual/  Contextual FAISS + BM25 인덱스
│  ├─ eval/              평가셋·평가 결과
│  └─ metadata/          파이프라인·평가 리포트
└─ scripts/              데이터 구축·평가 파이프라인 스크립트
```

## 2. 설치 방법

```bash
conda create -n myenv python=3.10
conda activate myenv
pip install -r requirements.txt
```

- 검색은 CPU로 동작하며, EXAONE 답변 생성은 GPU가 필요함 (INT4 ≈ 9GB, FP16 ≈ 18GB VRAM).
- 모델(`BAAI/bge-m3`, `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct`)은 첫 실행 시 HuggingFace에서 자동 다운로드된다.

## 3. 사용법

### 데모 대시보드

```bash
bash scripts/run_viz.sh              # 검색 전용 (CPU)
USE_GPU=1 bash scripts/run_viz.sh    # EXAONE 답변 생성 (GPU)
```

브라우저에서 질문을 입력하면 답변·검색 후보·근거 조항을 확인할 수 있음.

### 명령줄 (CLI)

검색만:

```bash
python main.py "외국인 유학생은 학생생활관 입주 지원을 받을 수 있나요?" \
  --retrieval-mode hybrid \
  --index data/index_contextual/faiss.index \
  --metadata data/index_contextual/faiss_metadata.jsonl \
  --bm25-index data/index_contextual/bm25.json \
  --device cpu
```

EXAONE 답변 생성:

```bash
python main.py "장학금은 어떤 학생에게 지급하나요?" \
  --retrieval-mode hybrid \
  --index data/index_contextual/faiss.index \
  --metadata data/index_contextual/faiss_metadata.jsonl \
  --bm25-index data/index_contextual/bm25.json \
  --llm-model LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
  --quantization fp16 --device cuda
```
