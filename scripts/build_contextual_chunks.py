#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm


ARTICLE_RE = re.compile(
    r"(?m)^\s*(제\s*\d+\s*조(?:\s*의\s*\d+)?)\s*(?:[（(]\s*([^)\n）]{1,80})\s*[）)])?"
)
HEADING_RE = re.compile(r"(?m)^\s*(제\s*\d+\s*(장|절|관))\s*([^\n]{0,100})$")
ADDENDUM_RE = re.compile(r"(?m)^\s*부\s*칙\b.*$")


def normalize_spaces(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_article_no(value: str) -> str:
    return re.sub(r"\s+", "", value)


def clean_title(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def heading_events(text: str):
    events = []
    for match in HEADING_RE.finditer(text):
        heading_no = normalize_article_no(match.group(1))
        kind = match.group(2)
        title = clean_title(match.group(3))
        events.append(
            {
                "start": match.start(),
                "kind": kind,
                "value": f"{heading_no} {title}".strip(),
            }
        )
    return events


def active_headings(events: list[dict], position: int) -> tuple[str, str, str]:
    chapter = ""
    section = ""
    subsection = ""
    for event in events:
        if event["start"] >= position:
            break
        if event["kind"] == "장":
            chapter = event["value"]
            section = ""
            subsection = ""
        elif event["kind"] == "절":
            section = event["value"]
            subsection = ""
        elif event["kind"] == "관":
            subsection = event["value"]
    return chapter, section, subsection


def remove_heading_lines(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if HEADING_RE.match(line):
            continue
        lines.append(line)
    return normalize_spaces("\n".join(lines))


def split_addendum(text: str) -> tuple[str, str]:
    match = ADDENDUM_RE.search(text)
    if not match:
        return normalize_spaces(text), ""
    return normalize_spaces(text[: match.start()]), normalize_spaces(text[match.start() :])


def article_header(article_no: str, article_title: str) -> str:
    if article_title:
        return f"{article_no}({article_title})"
    return article_no


def body_without_header(text: str, article_no: str, article_title: str) -> str:
    header = re.escape(article_no)
    if article_title:
        pattern = rf"^\s*{header}\s*[（(]\s*{re.escape(article_title)}\s*[）)]\s*"
    else:
        pattern = rf"^\s*{header}\s*"
    return normalize_spaces(re.sub(pattern, "", text, count=1))


def purpose_summary(rule_name: str, category: str, articles: list[dict]) -> str:
    purpose = None
    for article in articles:
        title = article.get("article_title", "")
        if title == "목적" or "목적" in title:
            purpose = body_without_header(article["display_text"], article["article_no"], title)
            break

    if purpose:
        purpose = re.sub(r"\s+", " ", purpose).strip()
        if len(purpose) > 260:
            purpose = purpose[:260].rstrip() + "..."
        return purpose

    titles = [article.get("article_title", "") for article in articles[:5] if article.get("article_title")]
    title_hint = ", ".join(titles)
    if title_hint:
        return f"이 규정은 {rule_name}의 {title_hint} 등에 관한 사항을 다룬다."
    if category:
        return f"이 규정은 제주대학교 {category} 분야의 {rule_name}에 관한 사항을 다룬다."
    return f"이 규정은 제주대학교 {rule_name}에 관한 사항을 다룬다."


def make_embedding_text(item: dict) -> str:
    context_lines = [
        item.get("display_text", ""),
        "",
        "[문서 맥락]",
        f"규정명: {item.get('rule_name', '')}",
        f"상위분류: {item.get('category', '')}",
        f"장/절/관: {' > '.join(part for part in [item.get('chapter', ''), item.get('section', ''), item.get('subsection', '')] if part)}",
        f"조문 메타데이터: {item.get('article_no', '')} {item.get('article_title', '')}".strip(),
        f"규정 목적 요약: {item.get('purpose_summary', '')}",
    ]
    return "\n".join(line for line in context_lines if line is not None).strip()


def parse_articles(text: str):
    events = heading_events(text)
    matches = list(ARTICLE_RE.finditer(text))
    articles = []
    addenda = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        raw_body = remove_heading_lines(text[start:end])
        article_body, addendum_body = split_addendum(raw_body)
        if not article_body:
            continue

        chapter, section, subsection = active_headings(events, start)
        article = {
            "article_no": normalize_article_no(match.group(1)),
            "article_title": clean_title(match.group(2) or ""),
            "chapter": chapter,
            "section": section,
            "subsection": subsection,
            "display_text": article_body,
        }
        articles.append(article)

        if addendum_body:
            addenda.append(
                {
                    "article_no": "부칙",
                    "article_title": "",
                    "chapter": chapter,
                    "section": section,
                    "subsection": subsection,
                    "display_text": addendum_body,
                    "source_article_no": article["article_no"],
                }
            )
    return articles, addenda


def main():
    parser = argparse.ArgumentParser(description="Build context-augmented article chunks.")
    parser.add_argument("--manifest", default="data/metadata/rules_manifest.json")
    parser.add_argument("--extract-report", default="data/metadata/extract_report.json")
    parser.add_argument("--output", default="data/processed/articles_contextual.jsonl")
    parser.add_argument("--report", default="data/metadata/contextual_chunk_report.json")
    parser.add_argument("--include-addenda", action="store_true")
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
    total_addenda = 0
    with output_path.open("w", encoding="utf-8") as out:
        for extracted in tqdm(extract_report, desc="Contextual chunking"):
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
                        "addenda": 0,
                    }
                )
                continue

            text = Path(text_path).read_text(encoding="utf-8")
            articles, addenda = parse_articles(text)
            summary_text = purpose_summary(
                record.get("rule_name", ""),
                record.get("category", ""),
                articles,
            )

            rows = list(articles)
            if args.include_addenda:
                rows.extend(addenda)

            for chunk_index, chunk in enumerate(rows, start=1):
                item = {
                    "rule_id": rule_id,
                    "rule_name": record.get("rule_name", ""),
                    "category": record.get("category", ""),
                    "chapter": chunk.get("chapter", ""),
                    "section": chunk.get("section", ""),
                    "subsection": chunk.get("subsection", ""),
                    "article_no": chunk["article_no"],
                    "article_title": chunk.get("article_title", ""),
                    "purpose_summary": summary_text,
                    "display_text": chunk["display_text"],
                    "text": chunk["display_text"],
                    "source_file": record.get("filename", extracted.get("source_file", "")),
                    "download_url": record.get("download_url", ""),
                    "source_url": record.get("download_url", ""),
                    "chunk_id": f"{rule_id}_ctx_{chunk_index:04d}",
                    "chunk_type": "addendum" if chunk["article_no"] == "부칙" else "article",
                }
                if chunk.get("source_article_no"):
                    item["source_article_no"] = chunk["source_article_no"]
                item["embedding_text"] = make_embedding_text(item)
                out.write(json.dumps(item, ensure_ascii=False) + "\n")

            total_articles += len(articles)
            total_addenda += len(addenda)
            summary.append(
                {
                    "rule_id": rule_id,
                    "source_file": extracted.get("source_file", ""),
                    "status": "chunked" if articles else "no_articles",
                    "articles": len(articles),
                    "addenda": len(addenda),
                    "purpose_summary": summary_text,
                }
            )

    report = {
        "files": len(summary),
        "total_articles": total_articles,
        "total_addenda": total_addenda,
        "include_addenda": args.include_addenda,
        "output": str(output_path),
        "items": summary,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "items"}, ensure_ascii=False, indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
