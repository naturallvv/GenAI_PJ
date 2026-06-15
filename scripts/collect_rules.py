#!/usr/bin/env python3
import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import unquote, urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


BASE_URL = "https://www.jejunu.ac.kr"
RULES_URL = "https://www.jejunu.ac.kr/schoolinfo/statusAll/rule.htm"


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def safe_filename(value: str, fallback: str) -> str:
    value = unquote(value or "").strip().strip('"')
    value = re.sub(r"[\\/:*?\"<>|]", "_", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or fallback


def filename_from_content_disposition(header: str, fallback: str) -> str:
    if not header:
        return fallback
    match = re.search(r"filename\*=UTF-8''([^;]+)", header, re.IGNORECASE)
    if match:
        return safe_filename(match.group(1), fallback)
    match = re.search(r"filename=([^;]+)", header, re.IGNORECASE)
    if match:
        return safe_filename(match.group(1), fallback)
    return fallback


def nearest_category(node) -> str:
    current = node
    while current:
        for prev in current.find_all_previous(["h2", "h3", "h4", "strong"], limit=20):
            text = clean_text(prev.get_text(" "))
            if text and text not in {"대학규정", "본문 바로가기[15]"}:
                return text
        current = current.parent
    return ""


def rule_name_for_link(link) -> str:
    td = link.find_parent("td")
    if not td:
        return ""

    prev_td = td.find_previous_sibling("td")
    if prev_td:
        text = clean_text(prev_td.get_text(" "))
        text = re.sub(r"\s*다운로드\s*$", "", text).strip()
        if text:
            return text

    row = link.find_parent("tr")
    if row:
        cells = row.find_all("td")
        try:
            idx = cells.index(td)
        except ValueError:
            idx = -1
        if idx > 0:
            text = clean_text(cells[idx - 1].get_text(" "))
            text = re.sub(r"\s*다운로드\s*$", "", text).strip()
            if text:
                return text
    return ""


def parse_manifest(html: str):
    soup = BeautifulSoup(html, "html.parser")
    records = []
    seen = set()

    for link in soup.select('a[href*="/cs/download.htm"][href*="act=download"]'):
        href = link.get("href", "")
        url = urljoin(BASE_URL, href.replace("&amp;", "&"))
        if url in seen:
            continue
        seen.add(url)

        match = re.search(r"seq=(\d+).*?[&?]no=(\d+)", url)
        seq = match.group(1) if match else ""
        no = match.group(2) if match else ""
        rule_name = rule_name_for_link(link) or f"rule_{seq}_{no}"
        category = nearest_category(link)

        records.append(
            {
                "rule_id": f"{seq}_{no}",
                "rule_name": rule_name,
                "category": category,
                "seq": seq,
                "no": no,
                "download_url": url,
                "source_page": RULES_URL,
                "filename": "",
                "path": "",
                "status": "pending",
                "error": "",
            }
        )
    return records


def download_file(session, record, raw_dir: Path, timeout: int, sleep_seconds: float):
    fallback = f"{record['rule_id']}.hwp"
    response = session.get(record["download_url"], stream=True, timeout=timeout)
    response.raise_for_status()

    filename = filename_from_content_disposition(
        response.headers.get("Content-Disposition", ""), fallback
    )
    if not filename.lower().endswith(".hwp"):
        content_type = response.headers.get("Content-Type", "")
        if "hwp" in content_type.lower():
            filename += ".hwp"

    target = raw_dir / filename
    if target.exists() and target.stat().st_size > 0:
        record.update(
            {
                "filename": filename,
                "path": str(target),
                "status": "skipped_existing",
                "error": "",
            }
        )
        time.sleep(sleep_seconds)
        return record

    with target.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 128):
            if chunk:
                handle.write(chunk)

    record.update(
        {
            "filename": filename,
            "path": str(target),
            "status": "downloaded",
            "error": "",
        }
    )
    time.sleep(sleep_seconds)
    return record


def main():
    parser = argparse.ArgumentParser(description="Collect Jeju National University rule HWP files.")
    parser.add_argument("--url", default=RULES_URL)
    parser.add_argument("--raw-dir", default="data/raw_hwp")
    parser.add_argument("--manifest", default="data/metadata/rules_manifest.json")
    parser.add_argument("--failures", default="data/metadata/download_failures.json")
    parser.add_argument("--sleep", type=float, default=0.4)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    manifest_path = Path(args.manifest)
    failures_path = Path(args.failures)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "Mozilla/5.0 (compatible; jejunu-rules-rag-collector/1.0)"}
    )

    html = session.get(args.url, timeout=args.timeout).text
    records = parse_manifest(html)

    for record in tqdm(records, desc="Downloading rules"):
        try:
            download_file(session, record, raw_dir, args.timeout, args.sleep)
        except Exception as exc:
            record.update({"status": "failed", "error": repr(exc)})
            time.sleep(args.sleep)

    manifest_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    failures = [record for record in records if record["status"] == "failed"]
    failures_path.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"manifest_count={len(records)}")
    print(f"downloaded={sum(1 for r in records if r['status'] == 'downloaded')}")
    print(f"skipped_existing={sum(1 for r in records if r['status'] == 'skipped_existing')}")
    print(f"failed={len(failures)}")
    print(f"manifest={manifest_path}")
    print(f"failures={failures_path}")


if __name__ == "__main__":
    main()
