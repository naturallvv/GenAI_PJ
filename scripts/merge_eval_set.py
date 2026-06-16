#!/usr/bin/env python3
"""후보(라벨) + 사람이 작성한 질문을 병합해 최종 평가셋을 만든다.

출력:
  data/eval/qa_seed_v2.jsonl      검색 평가셋 (기존 100 + 신규 400 = 500)
  data/eval/qa_gen_v2.jsonl       생성 평가셋 (gen_candidate 중 100개 선별)

병합 규칙:
  - eval_candidates.jsonl 의 라벨(expected_citations/keywords/reference_answer)에
    authored/batch*.jsonl 의 question 을 id 로 join 한다.
  - 기존 qa_seed.jsonl(100) 은 그대로 재사용해 검색셋 앞부분에 둔다.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval"
SCHEMA_FIELDS = ["id", "question", "expected_citations", "expected_keywords", "reference_answer"]


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="평가셋 병합")
    parser.add_argument("--candidates", default=str(EVAL / "eval_candidates.jsonl"))
    parser.add_argument("--authored-dir", default=str(EVAL / "authored"))
    parser.add_argument("--seed", default=str(EVAL / "qa_seed.jsonl"))
    parser.add_argument("--out-retrieval", default=str(EVAL / "qa_seed_v2.jsonl"))
    parser.add_argument("--out-generation", default=str(EVAL / "qa_gen_v2.jsonl"))
    parser.add_argument("--gen-count", type=int, default=100)
    args = parser.parse_args()

    candidates = {c["id"]: c for c in load_jsonl(Path(args.candidates))}

    # 작성된 질문 로드
    questions = {}
    for path in sorted(Path(args.authored_dir).glob("batch*.jsonl")):
        for row in load_jsonl(path):
            questions[row["id"]] = row["question"].strip()

    missing = [cid for cid in candidates if cid not in questions]
    if missing:
        raise SystemExit(f"질문이 없는 후보 {len(missing)}개: {missing[:10]}")
    empty = [cid for cid, q in questions.items() if cid in candidates and not q]
    if empty:
        raise SystemExit(f"빈 질문 {len(empty)}개: {empty[:10]}")

    # 신규 QA 레코드 구성 (스키마 통일)
    new_records = []
    gen_pool = []
    for cid, cand in candidates.items():
        record = {
            "id": cid,
            "question": questions[cid],
            "expected_citations": cand["expected_citations"],
            "expected_keywords": cand["expected_keywords"],
            "reference_answer": cand["reference_answer"],
        }
        new_records.append(record)
        if cand.get("gen_candidate"):
            gen_pool.append(record)

    # 검색셋: 기존 100 + 신규 400, 완전 중복 질문 제거(첫 등장 유지)
    seed = load_jsonl(Path(args.seed))
    combined = seed + new_records
    retrieval = []
    seen_q = {}
    dropped = []
    for r in combined:
        key = re.sub(r"\s+", "", r["question"])
        if key in seen_q:
            dropped.append((r["id"], seen_q[key]))
            continue
        seen_q[key] = r["id"]
        retrieval.append(r)
    if dropped:
        print(f"중복 질문 제거: {len(dropped)}건 {dropped}")
    Path(args.out_retrieval).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in retrieval) + "\n",
        encoding="utf-8",
    )

    # 생성셋: gen_candidate 중 키워드 3개 이상인 것에서 카테고리 다양성 고려해 선별
    gen_pool = [r for r in gen_pool if len(r["expected_keywords"]) >= 3]
    cat_of = {c["id"]: c.get("category", "") for c in candidates.values()}
    gen_pool.sort(key=lambda r: cat_of.get(r["id"], ""))
    # 카테고리 라운드로빈으로 다양성 확보
    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in gen_pool:
        by_cat[cat_of.get(r["id"], "")].append(r)
    selected = []
    cats = sorted(by_cat)
    while len(selected) < args.gen_count and any(by_cat.values()):
        for cat in cats:
            if by_cat[cat]:
                selected.append(by_cat[cat].pop(0))
                if len(selected) >= args.gen_count:
                    break
    Path(args.out_generation).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in selected) + "\n",
        encoding="utf-8",
    )

    print(f"검색셋(qa_seed_v2): {len(retrieval)}개 (기존 {len(seed)} + 신규 {len(new_records)})")
    print(f"생성셋(qa_gen_v2): {len(selected)}개")
    print(f"생성셋 카테고리 분포: {dict(Counter(cat_of.get(r['id'],'') for r in selected).most_common())}")


if __name__ == "__main__":
    main()
