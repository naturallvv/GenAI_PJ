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
            "name": name,
            "status": "missing",
            "report_path": str(path),
        }
    retrieval_mode = report.get("retrieval_mode")
    return {
        "name": name,
        "status": "ok",
        "qa_count": report.get("qa_count"),
        "top_k": report.get("top_k"),
        "retrieval_mode": retrieval_mode,
        "hit_at_1": metric(report, "retrieval", "hit_at_1"),
        "hit_at_3": metric(report, "retrieval", "hit_at_3"),
        "hit_at_5": metric(report, "retrieval", "hit_at_5"),
        "mrr": metric(report, "retrieval", "mrr"),
        "dense_weight": report.get("dense_weight") if retrieval_mode == "hybrid" else None,
        "candidate_k": report.get("candidate_k") if retrieval_mode in {"dense", "hybrid"} else None,
        "reranker_model": report.get("reranker_model"),
        "elapsed_seconds": report.get("elapsed_seconds"),
        "report_path": str(path),
    }


def format_value(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(rows: list[dict], path: Path):
    headers = [
        "name",
        "qa_count",
        "retrieval_mode",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "mrr",
        "dense_weight",
        "candidate_k",
        "elapsed_seconds",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(header)) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Compare retrieval evaluation reports.")
    parser.add_argument("--baseline", default="data/metadata/eval_report_retrieval_100.json")
    parser.add_argument("--contextual-dense", default="data/metadata/eval_report_contextual_dense_100.json")
    parser.add_argument("--contextual-hybrid", default="data/metadata/eval_report_contextual_hybrid_100.json")
    parser.add_argument(
        "--contextual-hybrid-reranker",
        default="data/metadata/eval_report_contextual_hybrid_reranker_100.json",
    )
    parser.add_argument("--output-json", default="data/metadata/retrieval_comparison_contextual.json")
    parser.add_argument("--output-md", default="data/metadata/retrieval_comparison_contextual.md")
    args = parser.parse_args()

    report_paths = {
        "Baseline Dense": Path(args.baseline),
        "Contextual Dense": Path(args.contextual_dense),
        "Contextual Hybrid": Path(args.contextual_hybrid),
        "Contextual Hybrid + Reranker": Path(args.contextual_hybrid_reranker),
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
