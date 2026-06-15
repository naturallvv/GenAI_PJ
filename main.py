#!/usr/bin/env python3
import argparse
import ctypes
import json
import math
import os
import re
import time
from pathlib import Path

import faiss
import torch
from sentence_transformers import SentenceTransformer


SYSTEM_INSTRUCTION = """당신은 제주대학교 규정 질의응답 도우미입니다.
반드시 제공된 근거 조항만 사용해 한국어로 답하세요.
근거에 없는 내용은 추측하지 말고 "찾을 수 없음"이라고 답하세요.
답변에는 관련 규정명과 조문 번호를 함께 표시하세요."""

TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")


def set_process_name_from_env(default: str = "neahyuk"):
    name = os.environ.get("GENAI_PROCESS_NAME", default)
    if not name:
        return
    safe_name = (name or default)[:15].encode("utf-8", errors="ignore")
    try:
        libc = ctypes.CDLL(None)
        libc.prctl(15, ctypes.c_char_p(safe_name), 0, 0, 0)
    except Exception:
        pass


def load_metadata(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tokenize_for_bm25(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "") if len(token) > 1]


def load_bm25_index(path: Path | str | None):
    if not path:
        return None
    with Path(path).open(encoding="utf-8") as handle:
        return json.loads(handle.read())


def load_reranker(
    model_name: str,
    device: str,
    cache_folder: str | None = None,
    local_files_only: bool = False,
):
    if not model_name:
        return None
    from sentence_transformers import CrossEncoder

    model_kwargs = {}
    if cache_folder:
        model_kwargs["cache_folder"] = cache_folder
    if local_files_only:
        model_kwargs["local_files_only"] = True
    return CrossEncoder(model_name, device=device, **model_kwargs)


def load_retriever(
    index_path: Path,
    metadata_path: Path,
    model_name: str,
    device: str,
    max_seq_length: int,
    cache_folder: str | None = None,
    local_files_only: bool = False,
):
    index = faiss.read_index(str(index_path))
    metadata = load_metadata(metadata_path)
    allow_trusted_torch_load_for_embedding_model(model_name)
    resolved_model_name = model_name
    if local_files_only:
        resolved_model_name = resolve_cached_hf_snapshot(model_name, cache_folder)
    model_kwargs = {}
    if cache_folder:
        model_kwargs["cache_folder"] = cache_folder
    if local_files_only:
        model_kwargs["local_files_only"] = True
    model = SentenceTransformer(resolved_model_name, device=device, **model_kwargs)
    model.max_seq_length = max_seq_length
    return index, metadata, model


def resolve_cached_hf_snapshot(model_name: str, cache_folder: str | None = None) -> str:
    if "/" not in model_name:
        return model_name

    cache_roots = []
    if cache_folder:
        cache_roots.append(Path(cache_folder))
    if os.environ.get("HF_HOME"):
        cache_roots.append(Path(os.environ["HF_HOME"]))
    cache_roots.append(Path.home() / ".cache" / "huggingface")

    repo_cache_name = f"models--{model_name.replace('/', '--')}"
    seen = set()
    for root in cache_roots:
        for repo_cache in (root / "hub" / repo_cache_name, root / repo_cache_name):
            if repo_cache in seen:
                continue
            seen.add(repo_cache)
            snapshots_dir = repo_cache / "snapshots"
            if not snapshots_dir.exists():
                continue
            snapshots = sorted(
                [path for path in snapshots_dir.iterdir() if path.is_dir()],
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            config_candidates = [path for path in snapshots if (path / "config.json").exists()]
            for snapshot in config_candidates:
                if (snapshot / "modules.json").exists():
                    return str(snapshot)
            if config_candidates:
                return str(config_candidates[0])

    return model_name


def allow_trusted_torch_load_for_embedding_model(model_name: str):
    """BAAI/bge-m3 is trusted and cached as pytorch_model.bin in this environment.

    transformers>=4.57 blocks torch.load with torch<2.6 because arbitrary .bin files
    can be unsafe. We only relax the check for the known embedding model used to
    build this FAISS index.
    """
    if model_name != "BAAI/bge-m3":
        return

    import transformers.modeling_utils as modeling_utils
    import transformers.utils.import_utils as import_utils

    import_utils.check_torch_load_is_safe = lambda: None
    modeling_utils.check_torch_load_is_safe = lambda: None


def dense_score_map(query: str, index, embedder, candidate_k: int):
    query_embedding = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    scores, ids = index.search(query_embedding, candidate_k)
    return {
        int(idx): float(score)
        for score, idx in zip(scores[0], ids[0])
        if idx >= 0
    }


def bm25_score_map(query: str, bm25_index: dict | None, candidate_k: int):
    if not bm25_index:
        return {}

    query_terms = tokenize_for_bm25(query)
    if not query_terms:
        return {}

    k1 = float(bm25_index.get("k1", 1.5))
    b = float(bm25_index.get("b", 0.75))
    avgdl = float(bm25_index.get("avgdl") or 0.0)
    idf = bm25_index.get("idf", {})
    scores = {}
    for document in bm25_index.get("documents", []):
        doc_id = int(document["doc_id"])
        doc_len = float(document.get("length") or 0.0)
        term_counts = document.get("term_counts", {})
        score = 0.0
        for term in query_terms:
            tf = float(term_counts.get(term, 0.0))
            if tf <= 0.0:
                continue
            denominator = tf + k1 * (1.0 - b + b * doc_len / avgdl) if avgdl > 0 else tf + k1
            score += float(idf.get(term, 0.0)) * (tf * (k1 + 1.0)) / denominator
        if score > 0.0:
            scores[doc_id] = score

    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:candidate_k])


def normalize_score_map(scores: dict[int, float]):
    if not scores:
        return {}
    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)
    if math.isclose(max_score, min_score):
        return {idx: 1.0 for idx in scores}
    return {idx: (score - min_score) / (max_score - min_score) for idx, score in scores.items()}


def build_ranked_results(
    metadata,
    ranked_indices: list[int],
    combined_scores: dict[int, float],
    dense_scores: dict[int, float] | None = None,
    bm25_scores: dict[int, float] | None = None,
    top_k: int = 5,
):
    results = []
    for rank, idx in enumerate(ranked_indices[:top_k], start=1):
        if idx < 0 or idx >= len(metadata):
            continue
        item = dict(metadata[int(idx)])
        item["rank"] = rank
        item["score"] = float(combined_scores.get(idx, 0.0))
        if dense_scores is not None:
            item["dense_score"] = float(dense_scores.get(idx, 0.0))
        if bm25_scores is not None:
            item["bm25_score"] = float(bm25_scores.get(idx, 0.0))
        results.append(item)
    return results


def apply_reranker(query: str, results: list[dict], reranker, top_k: int):
    if reranker is None or not results:
        return results[:top_k]
    pairs = [
        [
            query,
            item.get("embedding_text") or item.get("display_text") or item.get("text", ""),
        ]
        for item in results
    ]
    scores = reranker.predict(pairs)
    scored = []
    for item, score in zip(results, scores):
        updated = dict(item)
        updated["rerank_score"] = float(score)
        scored.append(updated)
    scored.sort(key=lambda item: item["rerank_score"], reverse=True)
    for rank, item in enumerate(scored[:top_k], start=1):
        item["rank"] = rank
        item["score"] = item["rerank_score"]
    return scored[:top_k]


def retrieve(
    query: str,
    index,
    metadata,
    embedder,
    top_k: int,
    retrieval_mode: str = "dense",
    bm25_index: dict | None = None,
    dense_weight: float = 0.9,
    candidate_k: int | None = None,
    reranker=None,
):
    candidate_k = candidate_k or max(top_k * 10, top_k)
    dense_scores = dense_score_map(query, index, embedder, candidate_k)

    if retrieval_mode == "dense":
        ranked_indices = [idx for idx, _score in sorted(dense_scores.items(), key=lambda item: item[1], reverse=True)]
        results = build_ranked_results(metadata, ranked_indices, dense_scores, dense_scores=dense_scores, top_k=candidate_k)
        return apply_reranker(query, results, reranker, top_k)

    if retrieval_mode != "hybrid":
        raise ValueError("retrieval_mode must be one of: dense, hybrid")

    bm25_scores = bm25_score_map(query, bm25_index, candidate_k)
    dense_norm = normalize_score_map(dense_scores)
    bm25_norm = normalize_score_map(bm25_scores)
    combined_scores = {}
    for idx in set(dense_scores) | set(bm25_scores):
        combined_scores[idx] = (
            dense_weight * dense_norm.get(idx, 0.0)
            + (1.0 - dense_weight) * bm25_norm.get(idx, 0.0)
        )
    ranked_indices = [idx for idx, _score in sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)]
    results = build_ranked_results(
        metadata,
        ranked_indices,
        combined_scores,
        dense_scores=dense_scores,
        bm25_scores=bm25_scores,
        top_k=candidate_k,
    )
    return apply_reranker(query, results, reranker, top_k)


def format_context(results):
    blocks = []
    for item in results:
        title = item.get("article_title") or ""
        citation = f"{item['rule_name']} {item['article_no']}"
        if title:
            citation += f"({title})"
        blocks.append(
            "\n".join(
                [
                    f"[근거 {item['rank']}] {citation}",
                    f"유사도: {item['score']:.4f}",
                    f"본문: {item.get('display_text') or item['text']}",
                    f"출처파일: {item['source_file']}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_prompt(query: str, results):
    context = format_context(results)
    return f"""{SYSTEM_INSTRUCTION}

[질문]
{query}

[검색된 근거 조항]
{context}

[답변 작성 규칙]
1. 위 근거 조항에 있는 내용만 사용하세요.
2. 답변 끝에 "근거: 규정명 조문" 형식으로 인용하세요.
3. 근거가 부족하면 "찾을 수 없음"이라고 답하세요.

[답변]
"""


def simple_grounded_answer(query: str, results):
    if not results:
        return "찾을 수 없음"
    first = results[0]
    first_title = first.get("article_title") or ""
    first_citation = f"{first['rule_name']} {first['article_no']}"
    if first_title:
        first_citation += f"({first_title})"

    lines = [
        f"검색된 규정 근거에 따르면, 핵심 조항은 {first_citation}입니다.",
        "관련 근거 조항은 아래와 같습니다.",
        "",
    ]
    for item in results[:3]:
        title = item.get("article_title") or ""
        citation = f"{item['rule_name']} {item['article_no']}"
        if title:
            citation += f"({title})"
        preview = (item.get("display_text") or item["text"]).replace("\n", " ")
        if len(preview) > 500:
            preview = preview[:500] + "..."
        lines.append(f"- {citation}: {preview}")
    lines.append("")
    lines.append("정리된 자연어 답변은 --llm-model 옵션으로 지시튜닝 LLM을 연결하면 생성됩니다.")
    return "\n".join(lines)


def load_llm(model_name: str, quantization: str):
    import transformers.integrations as integrations
    import transformers.modeling_utils as modeling_utils
    import transformers.modeling_rope_utils as rope_utils
    import transformers.utils as transformers_utils
    import transformers.utils.generic as generic_utils
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if not hasattr(rope_utils, "RopeParameters"):
        rope_utils.RopeParameters = dict
    patch_missing_kernel_integrations(integrations)
    patch_missing_generic_utils(generic_utils)
    patch_docstring_utils(transformers_utils)
    patch_attention_interface(modeling_utils)

    local_files_only = is_hf_offline()
    resolved_model_name = model_name
    if local_files_only:
        resolved_model_name = resolve_cached_hf_snapshot(model_name, os.environ.get("HF_HOME"))

    quant_config = None
    if quantization in {"int8", "int4"}:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("INT8/INT4 양자화에는 bitsandbytes 설치가 필요합니다.") from exc

        if quantization == "int8":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    config = AutoConfig.from_pretrained(
        resolved_model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    patch_exaone_rope_config(config)
    if quantization == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_name,
            config=config,
            trust_remote_code=True,
            dtype=torch.float16,
            local_files_only=local_files_only,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_name,
            config=config,
            trust_remote_code=True,
            quantization_config=quant_config,
            device_map="auto",
            local_files_only=local_files_only,
        )
    model.eval()
    return tokenizer, model


def is_hf_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def patch_missing_kernel_integrations(integrations):
    def class_decorator_factory(*_args, **_kwargs):
        def decorator(obj):
            return obj

        return decorator

    def func_decorator_factory(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

    if not hasattr(integrations, "use_kernel_forward_from_hub"):
        integrations.use_kernel_forward_from_hub = class_decorator_factory
    if not hasattr(integrations, "use_kernel_func_from_hub"):
        integrations.use_kernel_func_from_hub = func_decorator_factory
    if not hasattr(integrations, "use_kernelized_func"):
        integrations.use_kernelized_func = class_decorator_factory


def patch_missing_generic_utils(generic_utils):
    from contextlib import nullcontext

    if not hasattr(generic_utils, "maybe_autocast"):
        def maybe_autocast(*_args, **_kwargs):
            return nullcontext()

        generic_utils.maybe_autocast = maybe_autocast

    def check_model_inputs(func):
        return func

    generic_utils.check_model_inputs = check_model_inputs


def patch_docstring_utils(transformers_utils):
    def auto_docstring(obj=None, *args, **kwargs):
        if obj is not None and callable(obj):
            return obj

        def decorator(inner):
            return inner

        return decorator

    transformers_utils.auto_docstring = auto_docstring


def patch_attention_interface(modeling_utils):
    all_attention_functions = getattr(modeling_utils, "ALL_ATTENTION_FUNCTIONS", None)
    if all_attention_functions is None or hasattr(all_attention_functions, "get_interface"):
        return

    def get_interface(attn_implementation, default):
        if attn_implementation == "eager":
            return default
        return all_attention_functions.get(attn_implementation, default)

    all_attention_functions.get_interface = get_interface


def patch_exaone_rope_config(config):
    if getattr(config, "model_type", "") != "exaone":
        return
    if getattr(config, "rope_parameters", None) is not None:
        return

    rope_scaling = getattr(config, "rope_scaling", None)
    rope_theta = getattr(config, "rope_theta", None)

    if isinstance(rope_scaling, dict):
        rope_parameters = dict(rope_scaling)
        if rope_theta is not None:
            rope_parameters["rope_theta"] = rope_theta
    else:
        rope_parameters = {"rope_type": "default"}
        if rope_theta is not None:
            rope_parameters["rope_theta"] = rope_theta

    config.rope_parameters = rope_parameters


def generate_answer(prompt: str, model_name: str, quantization: str, max_new_tokens: int):
    tokenizer, model = load_llm(model_name, quantization)
    return generate_answer_with_model(prompt, tokenizer, model, max_new_tokens)


def generate_answer_with_model(prompt: str, tokenizer, model, max_new_tokens: int):
    generated, _generated_tokens = generate_answer_with_metrics(prompt, tokenizer, model, max_new_tokens)
    return generated


def generate_answer_with_metrics(prompt: str, tokenizer, model, max_new_tokens: int):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_token_count = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][input_token_count:]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated.strip(), int(generated_ids.shape[-1])


def main():
    set_process_name_from_env()
    parser = argparse.ArgumentParser(description="Jeju National University rule RAG runner.")
    parser.add_argument("query", nargs="?", help="질문. 생략하면 대화형 입력을 사용합니다.")
    parser.add_argument("--index", default="data/index/faiss.index")
    parser.add_argument("--metadata", default="data/index/faiss_metadata.jsonl")
    parser.add_argument("--bm25-index", default="")
    parser.add_argument("--retrieval-mode", choices=["dense", "hybrid"], default="dense")
    parser.add_argument("--dense-weight", type=float, default=0.9)
    parser.add_argument("--candidate-k", type=int, default=0)
    parser.add_argument("--reranker-model", default="")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-cache-folder", default="")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--llm-model", default="", help="예: LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
    parser.add_argument("--quantization", choices=["fp16", "int8", "int4"], default="fp16")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--show-prompt", action="store_true")
    args = parser.parse_args()

    query = args.query or input("질문: ").strip()
    if not query:
        raise SystemExit("질문이 비어 있습니다.")

    started = time.time()
    index, metadata, embedder = load_retriever(
        Path(args.index),
        Path(args.metadata),
        args.embedding_model,
        args.device,
        args.max_seq_length,
        cache_folder=args.embedding_cache_folder or None,
        local_files_only=args.local_files_only,
    )
    bm25_index = load_bm25_index(args.bm25_index) if args.retrieval_mode == "hybrid" else None
    reranker = (
        load_reranker(
            args.reranker_model,
            args.device,
            cache_folder=args.embedding_cache_folder or None,
            local_files_only=args.local_files_only,
        )
        if args.reranker_model
        else None
    )
    results = retrieve(
        query,
        index,
        metadata,
        embedder,
        args.top_k,
        retrieval_mode=args.retrieval_mode,
        bm25_index=bm25_index,
        dense_weight=args.dense_weight,
        candidate_k=args.candidate_k or None,
        reranker=reranker,
    )
    prompt = build_prompt(query, results)

    if args.llm_model:
        answer = generate_answer(prompt, args.llm_model, args.quantization, args.max_new_tokens)
    else:
        answer = simple_grounded_answer(query, results)

    print("\n=== 질문 ===")
    print(query)
    print("\n=== 답변 ===")
    print(answer)
    print("\n=== 검색 근거 ===")
    for item in results:
        title = item.get("article_title") or ""
        print(f"[{item['rank']}] score={item['score']:.4f} | {item['rule_name']} | {item['article_no']} {title}")
        print(item["text"].replace("\n", " ")[:260])
        print(f"source_file: {item['source_file']}")
        print()

    if args.show_prompt:
        print("\n=== LLM Prompt ===")
        print(prompt)

    print(f"elapsed_seconds={time.time() - started:.3f}")


if __name__ == "__main__":
    main()
