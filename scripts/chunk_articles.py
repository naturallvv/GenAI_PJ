#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm


ARTICLE_RE = re.compile(
    r"(?m)^\s*(제\s*\d+\s*조(?:\s*의\s*\d+)?)\s*(?:[（(]\s*([^)\n）]{1,80})\s*[）)])?"
)


def normalize_article_no(value: str) -> str:
    return re.sub(r"\s+", "", value)


def normalize_body(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str):
    matches = list(ARTICLE_RE.finditer(text))
    chunks = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = normalize_body(text[start:end])
        if not body:
            continue
        chunks.append(
            {
                "article_no": normalize_article_no(match.group(1)),
                "article_title": (match.group(2) or "").strip(),
                "text": body,
            }
        )
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Chunk extracted rule text by article.")
    parser.add_argument("--manifest", default="data/metadata/rules_manifest.json")
    parser.add_argument("--extract-report", default="data/metadata/extract_report.json")
    parser.add_argument("--output", default="data/processed/articles.jsonl")
    parser.add_argument("--report", default="data/metadata/chunk_report.json")
    args = parser.parse_args()

    manifest = {
        record["rule_id"]: record
        for record in json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    }
    extract_report = json.loads(Path(args.extract_report).read_text(encoding="utf-8"))

    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    summary = []
    total_articles = 0
    with output_path.open("w", encoding="utf-8") as out:
        for extracted in tqdm(extract_report, desc="Chunking articles"):
            rule_id = extracted["rule_id"]
            record = manifest.get(rule_id, {})
            text_path = extracted.get("text_path", "")
            if not text_path or extracted.get("status") != "extracted":
                summary.append(
                    {
                        "rule_id": rule_id,
                        "source_file": extracted.get("source_file", ""),
                        "status": "skipped",
                        "articles": 0,
                    }
                )
                continue

            text = Path(text_path).read_text(encoding="utf-8")
            chunks = chunk_text(text)
            for chunk_index, chunk in enumerate(chunks, start=1):
                item = {
                    "rule_id": rule_id,
                    "rule_name": record.get("rule_name", ""),
                    "article_no": chunk["article_no"],
                    "article_title": chunk["article_title"],
                    "text": chunk["text"],
                    "source_file": record.get("filename", extracted.get("source_file", "")),
                    "download_url": record.get("download_url", ""),
                    "chunk_id": f"{rule_id}_{chunk_index:04d}",
                }
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
            total_articles += len(chunks)
            summary.append(
                {
                    "rule_id": rule_id,
                    "source_file": extracted.get("source_file", ""),
                    "status": "chunked" if chunks else "no_articles",
                    "articles": len(chunks),
                }
            )

    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"files={len(summary)}")
    print(f"total_articles={total_articles}")
    print(f"no_articles={sum(1 for r in summary if r['status'] == 'no_articles')}")
    print(f"output={output_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
