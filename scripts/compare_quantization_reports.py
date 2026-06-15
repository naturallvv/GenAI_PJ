#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_report(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric(report: dict, *keys, default=None):
    value = report
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def row_from_report(name: str, path: Path, report: dict | None):
    if report is None:
        return {
            "quantization": name,
            "status": "missing",
            "report_path": str(path),
        }
    return {
        "quantization": name,
        "status": "ok",
        "qa_count": report.get("qa_count"),
        "hit_at_1": metric(report, "retrieval", "hit_at_1"),
        "hit_at_5": metric(report, "retrieval", "hit_at_5"),
        "mrr": metric(report, "retrieval", "mrr"),
        "citation_accuracy": metric(report, "generation", "citation_accuracy"),
        "keyword_recall": metric(report, "generation", "keyword_recall"),
        "answer_pass_rate": metric(report, "generation", "answer_pass_rate"),
        "avg_generation_seconds": metric(report, "generation", "generation_elapsed_seconds_avg"),
        "avg_generated_tokens": metric(report, "generation", "generated_tokens_avg"),
        "tokens_per_second": metric(report, "generation", "tokens_per_second_avg"),
        "max_memory_allocated_gb": metric(report, "cuda", "max_memory_allocated_gb"),
        "max_memory_reserved_gb": metric(report, "cuda", "max_memory_reserved_gb"),
        "elapsed_seconds": report.get("elapsed_seconds"),
        "report_path": str(path),
    }


def format_float(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(rows: list[dict], path: Path):
    headers = [
        "quantization",
        "qa_count",
        "hit_at_1",
        "hit_at_5",
        "mrr",
        "citation_accuracy",
        "keyword_recall",
        "answer_pass_rate",
        "avg_generation_seconds",
        "avg_generated_tokens",
        "tokens_per_second",
        "max_memory_allocated_gb",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_float(row.get(header)) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Compare EXAONE quantization evaluation reports.")
    parser.add_argument("--fp16", default="data/metadata/eval_report_exaone_fp16_25.json")
    parser.add_argument("--int8", default="data/metadata/eval_report_exaone_int8_25.json")
    parser.add_argument("--int4", default="data/metadata/eval_report_exaone_int4_25.json")
    parser.add_argument("--output-json", default="data/metadata/quantization_comparison.json")
    parser.add_argument("--output-md", default="data/metadata/quantization_comparison.md")
    args = parser.parse_args()

    report_paths = {
        "fp16": Path(args.fp16),
        "int8": Path(args.int8),
        "int4": Path(args.int4),
    }
    rows = [
        row_from_report(name, path, load_report(path))
        for name, path in report_paths.items()
    ]

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(rows, output_md)

    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"json={output_json}")
    print(f"markdown={output_md}")


if __name__ == "__main__":
    main()
