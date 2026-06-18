import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm


TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "") if len(token) > 1]


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description="Build a pure-Python BM25 index from article JSONL.")
    parser.add_argument("--articles", default="data/processed/articles_contextual.jsonl")
    parser.add_argument("--output", default="data/index_contextual/bm25.json")
    parser.add_argument("--text-field", default="embedding_text")
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    articles_path = Path(args.articles)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    documents = []
    document_frequency = defaultdict(int)
    total_length = 0

    for doc_id, article in enumerate(tqdm(load_jsonl(articles_path), desc="Building BM25")):
        text = article.get(args.text_field) or article.get("text", "")
        term_counts = Counter(tokenize(text))
        length = sum(term_counts.values())
        total_length += length
        for term in term_counts:
            document_frequency[term] += 1
        documents.append(
            {
                "doc_id": doc_id,
                "chunk_id": article.get("chunk_id", ""),
                "length": length,
                "term_counts": dict(term_counts),
            }
        )

    doc_count = len(documents)
    avgdl = total_length / doc_count if doc_count else 0.0
    idf = {
        term: math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
        for term, df in document_frequency.items()
    }

    index = {
        "articles": str(articles_path),
        "text_field": args.text_field,
        "doc_count": doc_count,
        "avgdl": avgdl,
        "k1": args.k1,
        "b": args.b,
        "idf": idf,
        "documents": documents,
    }
    output_path.write_text(json.dumps(index, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "articles": str(articles_path),
                "output": str(output_path),
                "doc_count": doc_count,
                "vocab_size": len(idf),
                "avgdl": round(avgdl, 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
