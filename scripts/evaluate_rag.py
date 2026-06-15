#!/usr/bin/env python3
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import (  # noqa: E402
    build_prompt,
    generate_answer_with_metrics,
    load_bm25_index,
    load_llm,
    load_reranker,
    load_retriever,
    retrieve,
    set_process_name_from_env,
)


def load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return rows


def normalize_rule_name(name: str) -> str:
    name = re.sub(r"^\s*\d+\.\s*", "", name or "")
    return re.sub(r"\s+", "", name)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def expected_citations(qa: dict):
    citations = qa.get("expected_citations") or []
    if citations:
        return citations
    if qa.get("expected_rule_name") and qa.get("expected_article_no"):
        return [
            {
                "rule_name": qa["expected_rule_name"],
                "article_no": qa["expected_article_no"],
            }
        ]
    return []


def citation_matches(result: dict, expected: dict) -> bool:
    return (
        normalize_rule_name(result.get("rule_name", "")) == normalize_rule_name(expected.get("rule_name", ""))
        and result.get("article_no", "") == expected.get("article_no", "")
    )


def first_gold_rank(results: list[dict], citations: list[dict]):
    for result in results:
        if any(citation_matches(result, expected) for expected in citations):
            return result["rank"]
    return None


def answer_contains_citation(answer: str, citations: list[dict]) -> bool:
    compact_answer = normalize_text(answer)
    for expected in citations:
        article_no = normalize_text(expected.get("article_no", ""))
        rule_name = normalize_rule_name(expected.get("rule_name", ""))
        if article_no and rule_name and article_no in compact_answer and rule_name in compact_answer:
            return True
        short_rule_name = normalize_rule_name(re.sub(r"^\s*\d+\.\s*", "", expected.get("rule_name", "")))
        if article_no and short_rule_name and article_no in compact_answer and short_rule_name in compact_answer:
            return True
    return False


def keyword_metrics(answer: str, expected_keywords: list[str]):
    compact_answer = normalize_text(answer)
    present = []
    missing = []
    for keyword in expected_keywords:
        if normalize_text(keyword) in compact_answer:
            present.append(keyword)
        else:
            missing.append(keyword)
    total = len(expected_keywords)
    recall = len(present) / total if total else None
    return {
        "present_keywords": present,
        "missing_keywords": missing,
        "keyword_recall": recall,
    }


def top_result_summary(results: list[dict]):
    return [
        {
            "rank": item["rank"],
            "score": item["score"],
            "rule_name": item["rule_name"],
            "article_no": item["article_no"],
            "article_title": item.get("article_title", ""),
            "chunk_id": item.get("chunk_id", ""),
            "dense_score": item.get("dense_score"),
            "bm25_score": item.get("bm25_score"),
            "rerank_score": item.get("rerank_score"),
        }
        for item in results
    ]


def average(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def summarize(rows: list[dict], top_k: int, llm_model: str, quantization: str, retrieval_mode: str):
    n = len(rows)
    retrieval = {
        "hit_at_1": average([1.0 if row["gold_rank"] == 1 else 0.0 for row in rows]),
        "hit_at_3": average([1.0 if row["gold_rank"] and row["gold_rank"] <= 3 else 0.0 for row in rows]),
        f"hit_at_{top_k}": average([1.0 if row["gold_rank"] else 0.0 for row in rows]),
        "mrr": average([1.0 / row["gold_rank"] if row["gold_rank"] else 0.0 for row in rows]),
    }
    report = {
        "qa_count": n,
        "top_k": top_k,
        "retrieval_mode": retrieval_mode,
        "retrieval": retrieval,
        "llm_model": llm_model or None,
        "quantization": quantization if llm_model else None,
    }

    generated_rows = [row for row in rows if row.get("answer") is not None]
    if generated_rows:
        report["generation"] = {
            "evaluated_count": len(generated_rows),
            "citation_accuracy": average([1.0 if row["citation_in_answer"] else 0.0 for row in generated_rows]),
            "keyword_recall": average([row["keyword_recall"] for row in generated_rows]),
            "answer_pass_rate": average([1.0 if row["answer_pass"] else 0.0 for row in generated_rows]),
            "generation_elapsed_seconds_avg": average([row["generation_elapsed_seconds"] for row in generated_rows]),
            "generated_tokens_avg": average([row["generated_tokens"] for row in generated_rows]),
            "tokens_per_second_avg": average([row["tokens_per_second"] for row in generated_rows]),
        }

    return report


def main():
    set_process_name_from_env()
    parser = argparse.ArgumentParser(description="Evaluate retrieval and optional LLM answer quality for the RAG system.")
    parser.add_argument("--qa", default="data/eval/qa_seed.jsonl")
    parser.add_argument("--index", default="data/index/faiss.index")
    parser.add_argument("--metadata", default="data/index/faiss_metadata.jsonl")
    parser.add_argument("--bm25-index", default="")
    parser.add_argument("--retrieval-mode", choices=["dense", "hybrid"], default="dense")
    parser.add_argument("--dense-weight", type=float, default=0.9)
    parser.add_argument("--candidate-k", type=int, default=0)
    parser.add_argument("--reranker-model", default="")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-cache-folder", default="")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0, help="0이면 전체 QA를 평가합니다.")
    parser.add_argument("--output", default="data/eval/eval_results.jsonl")
    parser.add_argument("--report", default="data/metadata/eval_report.json")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--quantization", choices=["fp16", "int8", "int4"], default="fp16")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--keyword-threshold", type=float, default=0.5)
    args = parser.parse_args()

    qa_rows = load_jsonl(Path(args.qa))
    if args.limit > 0:
        qa_rows = qa_rows[: args.limit]
    if not qa_rows:
        raise SystemExit("No QA rows to evaluate.")

    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    index, metadata, embedder = load_retriever(
        Path(args.index),
        Path(args.metadata),
        args.embedding_model,
        args.device,
        args.max_seq_length,
        cache_folder=args.embedding_cache_folder or None,
        local_files_only=args.local_files_only,
    )
    bm25_index = load_bm25_index(args.bm25_index) if args.retrieval_mode == "hybrid" else None
    reranker = (
        load_reranker(
            args.reranker_model,
            args.device,
            cache_folder=args.embedding_cache_folder or None,
            local_files_only=args.local_files_only,
        )
        if args.reranker_model
        else None
    )

    tokenizer = None
    llm = None
    if args.llm_model:
        tokenizer, llm = load_llm(args.llm_model, args.quantization)

    evaluated = []
    started = time.time()
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, qa in enumerate(qa_rows, start=1):
            question = qa["question"]
            citations = expected_citations(qa)
            results = retrieve(
                question,
                index,
                metadata,
                embedder,
                args.top_k,
                retrieval_mode=args.retrieval_mode,
                bm25_index=bm25_index,
                dense_weight=args.dense_weight,
                candidate_k=args.candidate_k or None,
                reranker=reranker,
            )
            gold_rank = first_gold_rank(results, citations)

            row = {
                "id": qa.get("id", f"qa_{idx:04d}"),
                "question": question,
                "expected_citations": citations,
                "expected_keywords": qa.get("expected_keywords", []),
                "gold_rank": gold_rank,
                "retrieval_hit": gold_rank is not None,
                "top_results": top_result_summary(results),
            }

            if llm is not None and tokenizer is not None:
                prompt = build_prompt(question, results)
                generation_started = time.time()
                answer, generated_tokens = generate_answer_with_metrics(
                    prompt,
                    tokenizer,
                    llm,
                    args.max_new_tokens,
                )
                generation_elapsed = time.time() - generation_started
                tokens_per_second = generated_tokens / generation_elapsed if generation_elapsed > 0 else None
                keyword_result = keyword_metrics(answer, qa.get("expected_keywords", []))
                citation_in_answer = answer_contains_citation(answer, citations)
                keyword_recall = keyword_result["keyword_recall"]
                answer_pass = citation_in_answer and (
                    keyword_recall is None or keyword_recall >= args.keyword_threshold
                )
                row.update(
                    {
                        "answer": answer,
                        "citation_in_answer": citation_in_answer,
                        "keyword_recall": keyword_recall,
                        "present_keywords": keyword_result["present_keywords"],
                        "missing_keywords": keyword_result["missing_keywords"],
                        "answer_pass": answer_pass,
                        "generation_elapsed_seconds": round(generation_elapsed, 3),
                        "generated_tokens": generated_tokens,
                        "tokens_per_second": round(tokens_per_second, 3) if tokens_per_second is not None else None,
                    }
                )

            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            evaluated.append(row)
            print(
                f"[{idx}/{len(qa_rows)}] {row['id']} "
                f"gold_rank={gold_rank or '-'} "
                f"hit={row['retrieval_hit']}"
            )

    report = summarize(evaluated, args.top_k, args.llm_model, args.quantization, args.retrieval_mode)
    report["index"] = args.index
    report["metadata"] = args.metadata
    report["bm25_index"] = args.bm25_index or None
    report["dense_weight"] = args.dense_weight
    report["candidate_k"] = args.candidate_k or max(args.top_k * 10, args.top_k)
    report["reranker_model"] = args.reranker_model or None
    report["elapsed_seconds"] = round(time.time() - started, 3)
    if torch.cuda.is_available():
        report["cuda"] = {
            "device": torch.cuda.get_device_name(0),
            "max_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
            "max_memory_reserved_gb": round(torch.cuda.max_memory_reserved() / (1024**3), 3),
        }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n=== Evaluation Report ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nresults={output_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
