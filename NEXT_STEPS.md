# Next Steps

EXAONE FP16/INT8/INT4 평가, 100개 검색 평가셋 확장, 데모 대시보드 구성까지 완료했다. 남은 핵심 작업은 제출 구조 정리와 선택적인 추가 평가다.

## 완료된 상태

- 제주대학교 규정집 원문 수집 완료
- HWP/PDF 텍스트 추출 완료
- 조문 단위 청킹 완료
- `BAAI/bge-m3` 임베딩 기반 FAISS 인덱스 구축 완료
- EXAONE-3.5-7.8B-Instruct FP16 단일 질의 RAG 실행 확인
- 평가용 QA seed 100개 작성 완료
- 검색 평가 100문항 완료
- EXAONE FP16 답변 평가 25문항 완료
- EXAONE INT8/INT4 양자화 답변 평가 25문항 완료
- `viz.py` Streamlit 데모 대시보드 구현 완료

검색 평가 결과:

```json
{
  "qa_count": 100,
  "top_k": 5,
  "retrieval": {
    "hit_at_1": 0.88,
    "hit_at_3": 0.97,
    "hit_at_5": 1.0,
    "mrr": 0.9291666666666667
  }
}
```

EXAONE FP16 답변 평가 결과:

```json
{
  "qa_count": 25,
  "generation": {
    "citation_accuracy": 0.88,
    "keyword_recall": 0.898,
    "answer_pass_rate": 0.84,
    "generation_elapsed_seconds_avg": 4.00156
  },
  "cuda": {
    "max_memory_allocated_gb": 17.398
  }
}
```

양자화 비교 결과:

| quantization | qa_count | hit_at_1 | hit_at_5 | mrr | citation_accuracy | keyword_recall | answer_pass_rate | avg_generation_seconds | avg_generated_tokens | tokens_per_second | max_memory_allocated_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp16 | 25 | 0.8800 | 1.0000 | 0.9400 | 0.8800 | 0.8980 | 0.8400 | 4.0016 | 137.0800 | 33.5938 | 17.3980 |
| int8 | 25 | 0.8800 | 1.0000 | 0.9400 | 0.8800 | 0.8960 | 0.8800 | 7.4312 | 145.0000 | 19.1897 | 11.7460 |
| int4 | 25 | 0.8800 | 1.0000 | 0.9400 | 0.8000 | 0.8700 | 0.7600 | 2.5760 | 136.1200 | 50.9247 | 8.9960 |

평가 파일:

- `data/eval/qa_seed.jsonl`
- `data/eval/eval_results_retrieval_100.jsonl`
- `data/metadata/eval_report_retrieval_100.json`
- `data/eval/eval_results_retrieval_25.jsonl`
- `data/metadata/eval_report_retrieval_25.json`
- `data/eval/eval_results_exaone_fp16_25.jsonl`
- `data/metadata/eval_report_exaone_fp16_25.json`
- `data/eval/eval_results_exaone_int8_25.jsonl`
- `data/metadata/eval_report_exaone_int8_25.json`
- `data/eval/eval_results_exaone_int4_25.jsonl`
- `data/metadata/eval_report_exaone_int4_25.json`
- `data/metadata/quantization_comparison.json`
- `data/metadata/quantization_comparison.md`

## 대시보드 실행

기본 실행:

```bash
bash scripts/run_viz.sh
```

GPU 생성 실행:

```bash
USE_GPU=1 bash scripts/run_viz.sh
```

포트 변경:

```bash
PORT=8502 bash scripts/run_viz.sh
```

## 남은 일

1. 제출 구조 정리

- `Data/` 폴더 또는 현재 `data/` 폴더 제출용 정리
- `requirements.txt` 버전 확인
- `README.md` 설치 및 실행 가이드 보강
- `main.py` 전체 실행 스크립트 역할 확인
- `viz.py` 포함
- 최종 ZIP 이름 결정

2. 선택 작업

- 평가 QA를 100개에서 200개로 추가 확장
- 100개 QA 전체에 대해 FP16/INT8/INT4 답변 평가 재실행
- 보고서용 실패 사례 분석 표 작성

## 주의

- 추론/대시보드는 `USE_GPU=1` 실행 시 빈 GPU 자동 선택을 유지한다.
- 학습 작업은 GPU 사용 전 `nvidia-smi`를 확인하고, 사용할 GPU 번호와 개수를 사용자에게 먼저 확인한다.
- GPU 프로세스명은 `nvidia-smi`에서 `neahyuk`으로 표시되게 실행한다.
- Ceph 캐시는 `/ceph_data/wq1880/Gen_AI/hf_cache`만 사용한다.
- 다른 사용자 폴더와 시스템 설정 파일은 수정하지 않는다.
