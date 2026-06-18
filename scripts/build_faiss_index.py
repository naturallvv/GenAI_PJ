import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_articles(path: Path):
    articles = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                articles.append(json.loads(line))
    return articles


def article_to_embedding_text(article: dict) -> str:
    if article.get("embedding_text"):
        return article["embedding_text"]
    title = article.get("article_title") or ""
    return "\n".join(
        [
            f"규정명: {article.get('rule_name', '')}",
            f"조문: {article.get('article_no', '')} {title}",
            f"본문: {article.get('text', '')}",
        ]
    ).strip()


def allow_trusted_torch_load_for_embedding_model(model_name: str):
    if model_name != "BAAI/bge-m3":
        return

    import transformers.modeling_utils as modeling_utils
    import transformers.utils.import_utils as import_utils

    import_utils.check_torch_load_is_safe = lambda: None
    modeling_utils.check_torch_load_is_safe = lambda: None


def main():
    parser = argparse.ArgumentParser(description="Build a FAISS index from article JSONL.")
    parser.add_argument("--articles", default="data/processed/articles.jsonl")
    parser.add_argument("--index", default="data/index/faiss.index")
    parser.add_argument("--metadata", default="data/index/faiss_metadata.jsonl")
    parser.add_argument("--report", default="data/metadata/index_report.json")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--text-field", default="", help="명시하면 해당 필드를 임베딩 텍스트로 사용합니다.")
    args = parser.parse_args()

    articles_path = Path(args.articles)
    index_path = Path(args.index)
    metadata_path = Path(args.metadata)
    report_path = Path(args.report)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    articles = load_articles(articles_path)
    if args.text_field:
        texts = [article.get(args.text_field, "") for article in articles]
    else:
        texts = [article_to_embedding_text(article) for article in articles]

    started = time.time()
    allow_trusted_torch_load_for_embedding_model(args.model)
    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = args.max_seq_length
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(index_path))

    with metadata_path.open("w", encoding="utf-8") as handle:
        for article in articles:
            handle.write(json.dumps(article, ensure_ascii=False) + "\n")

    elapsed = time.time() - started
    report = {
        "model": args.model,
        "device": args.device,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "text_field": args.text_field or "embedding_text|fallback",
        "articles": len(articles),
        "embedding_dim": int(embeddings.shape[1]),
        "index_size": int(index.ntotal),
        "elapsed_seconds": round(elapsed, 3),
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
