#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_metadata(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Search the FAISS article index.")
    parser.add_argument("query", nargs="?", default="장학금 지급 기준은 어디에 있나요?")
    parser.add_argument("--index", default="data/index/faiss.index")
    parser.add_argument("--metadata", default="data/index/faiss_metadata.jsonl")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    args = parser.parse_args()

    index = faiss.read_index(str(Path(args.index)))
    metadata = load_metadata(Path(args.metadata))

    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = args.max_seq_length
    query_embedding = model.encode(
        [args.query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, ids = index.search(query_embedding, args.top_k)
    for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
        item = metadata[int(idx)]
        preview = item["text"].replace("\n", " ")[:220]
        print(f"[{rank}] score={float(score):.4f}")
        print(f"규정명: {item['rule_name']}")
        print(f"조문: {item['article_no']} {item.get('article_title', '')}")
        print(f"본문: {preview}")
        print(f"source_file: {item['source_file']}")
        print()


if __name__ == "__main__":
    main()
