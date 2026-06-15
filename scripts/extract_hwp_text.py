#!/usr/bin/env python3
import argparse
import json
import re
import struct
import zlib
from pathlib import Path

import olefile
from pypdf import PdfReader
from tqdm import tqdm


PARA_TEXT_TAG_ID = 67


def safe_stem(filename: str) -> str:
    return re.sub(r"[\\/:*?\"<>|]", "_", Path(filename).stem)


def is_compressed(ole: olefile.OleFileIO) -> bool:
    header = ole.openstream("FileHeader").read()
    flags = struct.unpack_from("<I", header, 36)[0]
    return bool(flags & 1)


def section_streams(ole: olefile.OleFileIO):
    streams = []
    for path in ole.listdir(streams=True, storages=False):
        if len(path) == 2 and path[0] == "BodyText" and path[1].startswith("Section"):
            streams.append(path)
    return sorted(streams, key=lambda p: int(re.sub(r"\D", "", p[1]) or 0))


def read_record_header(data: bytes, offset: int):
    header = struct.unpack_from("<I", data, offset)[0]
    tag_id = header & 0x3FF
    level = (header >> 10) & 0x3FF
    size = (header >> 20) & 0xFFF
    offset += 4
    if size == 0xFFF:
        size = struct.unpack_from("<I", data, offset)[0]
        offset += 4
    return tag_id, level, size, offset


def clean_extracted_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b-\x1f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_hwp(path: Path) -> str:
    with olefile.OleFileIO(str(path)) as ole:
        compressed = is_compressed(ole)
        chunks = []
        for stream_path in section_streams(ole):
            data = ole.openstream(stream_path).read()
            if compressed:
                data = zlib.decompress(data, -15)

            offset = 0
            while offset + 4 <= len(data):
                tag_id, _level, size, payload_offset = read_record_header(data, offset)
                payload_end = payload_offset + size
                if payload_end > len(data):
                    break
                payload = data[payload_offset:payload_end]
                if tag_id == PARA_TEXT_TAG_ID and payload:
                    try:
                        text = payload.decode("utf-16le", errors="ignore")
                    except UnicodeDecodeError:
                        text = ""
                    if text:
                        chunks.append(text)
                offset = payload_end

    return clean_extracted_text("\n".join(chunks))


def extract_text_from_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    chunks = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text:
            chunks.append(text)
    return clean_extracted_text("\n".join(chunks))


def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(path)
    return extract_text_from_hwp(path)


def load_manifest(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Extract text from HWP files listed in manifest.")
    parser.add_argument("--manifest", default="data/metadata/rules_manifest.json")
    parser.add_argument("--text-dir", default="data/text")
    parser.add_argument("--report", default="data/metadata/extract_report.json")
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    text_dir = Path(args.text_dir)
    report_path = Path(args.report)
    text_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = []
    for record in tqdm(manifest, desc="Extracting HWP text"):
        if record.get("status") == "failed" or not record.get("path"):
            report.append(
                {
                    "rule_id": record.get("rule_id", ""),
                    "source_file": record.get("filename", ""),
                    "text_path": "",
                    "status": "skipped_no_file",
                    "chars": 0,
                    "has_article_1": False,
                    "error": record.get("error", ""),
                }
            )
            continue

        source_path = Path(record["path"])
        text_path = text_dir / f"{safe_stem(record['filename'])}.txt"
        try:
            text = extract_text(source_path)
            text_path.write_text(text + "\n", encoding="utf-8")
            report.append(
                {
                    "rule_id": record["rule_id"],
                    "source_file": record["filename"],
                    "text_path": str(text_path),
                    "status": "extracted" if text else "empty_text",
                    "chars": len(text),
                    "has_article_1": bool(re.search(r"제\s*1\s*조", text)),
                    "error": "",
                }
            )
        except Exception as exc:
            report.append(
                {
                    "rule_id": record["rule_id"],
                    "source_file": record.get("filename", ""),
                    "text_path": str(text_path),
                    "status": "failed",
                    "chars": 0,
                    "has_article_1": False,
                    "error": repr(exc),
                }
            )

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"total={len(report)}")
    print(f"extracted={sum(1 for r in report if r['status'] == 'extracted')}")
    print(f"empty_text={sum(1 for r in report if r['status'] == 'empty_text')}")
    print(f"failed={sum(1 for r in report if r['status'] == 'failed')}")
    print(f"has_article_1={sum(1 for r in report if r['has_article_1'])}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
