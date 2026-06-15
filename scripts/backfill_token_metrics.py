#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import resolve_cached_hf_snapshot  # noqa: E402


def average(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def update_report(report_path: Path, rows: list[dict]):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    generated_rows = [row for row in rows if row.get("answer") is not None]
    if generated_rows:
        report.setdefault("generation", {})
        report["generation"]["generated_tokens_avg"] = average(
            [row.get("generated_tokens") for row in generated_rows]
        )
        report["generation"]["tokens_per_second_avg"] = average(
            [row.get("tokens_per_second") for row in generated_rows]
        )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Backfill generated token counts and tokens/sec in eval results.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--model", default="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
    parser.add_argument("--hf-home", default="/ceph_data/wq1880/Gen_AI/hf_cache")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    model_path = resolve_cached_hf_snapshot(args.model, args.hf_home)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    results_path = Path(args.results)
    report_path = Path(args.report)
    rows = load_jsonl(results_path)
    for row in rows:
        answer = row.get("answer")
        elapsed = row.get("generation_elapsed_seconds")
        if answer is None or elapsed in (None, 0):
            continue
        token_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        generated_tokens = len(token_ids)
        row["generated_tokens"] = generated_tokens
        row["tokens_per_second"] = round(generated_tokens / elapsed, 3)

    write_jsonl(results_path, rows)
    update_report(report_path, rows)
    print(f"updated_results={results_path}")
    print(f"updated_report={report_path}")


if __name__ == "__main__":
    main()
