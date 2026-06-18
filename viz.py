#!/usr/bin/env python3
import os
import time
from pathlib import Path

import streamlit as st
import torch

from main import (
    build_prompt,
    generate_answer_with_metrics,
    load_bm25_index,
    load_llm,
    load_retriever,
    retrieve,
    set_process_name_from_env,
    simple_grounded_answer,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INDEX = PROJECT_ROOT / "data" / "index" / "faiss.index"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "index" / "faiss_metadata.jsonl"
DEFAULT_CONTEXTUAL_INDEX = PROJECT_ROOT / "data" / "index_contextual" / "faiss.index"
DEFAULT_CONTEXTUAL_METADATA = PROJECT_ROOT / "data" / "index_contextual" / "faiss_metadata.jsonl"
DEFAULT_CONTEXTUAL_BM25 = PROJECT_ROOT / "data" / "index_contextual" / "bm25.json"
DEFAULT_EMBEDDING_CACHE = Path("/home/wq1880/.cache/huggingface")
DEFAULT_LLM_CACHE = Path("/ceph_data/wq1880/Gen_AI/hf_cache")
DEFAULT_LLM_MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"


set_process_name_from_env()


st.set_page_config(
    page_title="제주대학교 규정 RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_style():
    st.markdown(
        """
        <style>
        :root {
            --accent: #0f766e;
            --accent-soft: #ccfbf1;
            --ink: #172033;
            --muted: #667085;
            --line: #d0d5dd;
            --surface: #f8fafc;
        }
        .block-container {
            padding-top: 1.3rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--ink);
        }
        .metric-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.65rem;
            margin: 0.6rem 0 1rem;
        }
        .metric-box {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: white;
            padding: 0.7rem 0.8rem;
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.78rem;
            margin-bottom: 0.2rem;
        }
        .metric-value {
            color: var(--ink);
            font-size: 1.05rem;
            font-weight: 700;
        }
        .answer-box {
            border-left: 4px solid var(--accent);
            background: #f0fdfa;
            padding: 0.9rem 1rem;
            border-radius: 0 8px 8px 0;
            line-height: 1.65;
            white-space: pre-wrap;
        }
        .evidence {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            margin-bottom: 0.7rem;
            background: white;
        }
        .evidence-title {
            font-weight: 700;
            color: var(--ink);
            margin-bottom: 0.3rem;
        }
        .evidence-meta {
            color: var(--muted);
            font-size: 0.82rem;
            margin-bottom: 0.5rem;
        }
        .evidence-text {
            color: #344054;
            line-height: 1.58;
            white-space: pre-wrap;
        }
        .score-track {
            height: 0.6rem;
            background: #eaecf0;
            border-radius: 999px;
            overflow: hidden;
            margin: 0.3rem 0 0.6rem;
        }
        .score-fill {
            height: 100%;
            background: var(--accent);
            border-radius: 999px;
        }
        @media (max-width: 780px) {
            .metric-row {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def clamp_score(score: float) -> float:
    return max(0.0, min(float(score), 1.0))


def html_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def setup_hf_cache(cache_dir: str, offline: bool):
    cache_dir = cache_dir.strip()
    if cache_dir:
        os.environ["HF_HOME"] = cache_dir
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)


def cuda_available() -> bool:
    return torch.cuda.is_available()


@st.cache_resource(show_spinner="검색 인덱스를 불러오는 중")
def get_retriever(
    index_path: str,
    metadata_path: str,
    embedding_model: str,
    device: str,
    max_seq_length: int,
    embedding_cache_folder: str,
    local_files_only: bool,
):
    return load_retriever(
        Path(index_path),
        Path(metadata_path),
        embedding_model,
        device,
        max_seq_length,
        cache_folder=embedding_cache_folder or None,
        local_files_only=local_files_only,
    )


@st.cache_resource(show_spinner="BM25 인덱스를 불러오는 중")
def get_bm25_index(path: str):
    if not path:
        return None
    return load_bm25_index(path)


@st.cache_resource(show_spinner="LLM을 불러오는 중")
def get_llm(model_name: str, quantization: str, cache_dir: str, offline: bool):
    setup_hf_cache(cache_dir, offline)
    return load_llm(model_name, quantization)


def render_answer(answer: str):
    st.markdown(
        f"<div class='answer-box'>{html_escape(answer)}</div>",
        unsafe_allow_html=True,
    )


def render_evidence(results: list[dict]):
    for item in results:
        title = item.get("article_title") or ""
        citation = f"{item['rule_name']} {item['article_no']}"
        if title:
            citation += f"({title})"
        score = clamp_score(item["score"])
        preview = item["text"].replace("\n", " ")
        st.markdown(
            f"""
            <div class="evidence">
                <div class="evidence-title">[{item['rank']}] {html_escape(citation)}</div>
                <div class="evidence-meta">score={item['score']:.4f} · {html_escape(item.get('source_file', ''))}</div>
                <div class="score-track"><div class="score-fill" style="width: {score * 100:.1f}%"></div></div>
                <div class="evidence-text">{html_escape(preview)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_candidates(results: list[dict]):
    rows = []
    for item in results:
        rows.append(
            {
                "rank": item["rank"],
                "score": round(item["score"], 4),
                "rule_name": item["rule_name"],
                "article_no": item["article_no"],
                "article_title": item.get("article_title", ""),
                "source_file": item.get("source_file", ""),
                "dense_score": round(item["dense_score"], 4) if item.get("dense_score") is not None else None,
                "bm25_score": round(item["bm25_score"], 4) if item.get("bm25_score") is not None else None,
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)


def main():
    apply_style()

    st.title("제주대학교 규정 RAG")
    has_cuda = cuda_available()

    with st.sidebar:
        st.subheader("검색 방식")
        retrieval_profile = st.selectbox(
            "검색 방식",
            [
                "Baseline Dense",
                "Contextual Dense",
                "Contextual Hybrid",
            ],
            index=2,
            help="규정 검색 방식. 평가에서 Contextual Hybrid가 가장 우수했습니다.",
        )

        st.subheader("답변 생성")
        use_llm = st.toggle("EXAONE 답변 생성", value=False, disabled=not has_cuda)
        if use_llm:
            quantization = st.selectbox("양자화", ["fp16", "int8", "int4"], index=0)
        else:
            quantization = "fp16"
        if not has_cuda:
            st.info("GPU가 보이지 않아 EXAONE 생성은 비활성화됩니다. `USE_GPU=1 bash scripts/run_viz.sh`로 실행하세요.")

        # 검색 방식에서 파생되는 설정(위젯 아님)
        use_contextual = retrieval_profile.startswith("Contextual")
        retrieval_mode = "hybrid" if "Hybrid" in retrieval_profile else "dense"
        index_default = DEFAULT_CONTEXTUAL_INDEX if use_contextual else DEFAULT_INDEX
        metadata_default = DEFAULT_CONTEXTUAL_METADATA if use_contextual else DEFAULT_METADATA

        with st.expander("고급 설정", expanded=False):
            top_k = st.slider("Top-k", min_value=3, max_value=10, value=5, step=1)
            dense_weight = st.slider("Dense 가중치", min_value=0.1, max_value=0.9, value=0.9, step=0.1)
            candidate_k = st.select_slider("후보 수", options=[20, 50, 100], value=50)
            max_new_tokens = st.slider("최대 생성 토큰", min_value=128, max_value=768, value=384, step=64)
            embedding_devices = ["cpu", "cuda"] if has_cuda else ["cpu"]
            embedding_device = st.selectbox("임베딩 장치", embedding_devices, index=len(embedding_devices) - 1)
            max_seq_length = st.select_slider("임베딩 길이", options=[512, 1024, 2048], value=1024)
            embedding_model = st.text_input("임베딩 모델", value="BAAI/bge-m3")
            embedding_cache = st.text_input("임베딩 캐시", value=str(DEFAULT_EMBEDDING_CACHE))
            local_files_only = st.checkbox("로컬 캐시만 사용", value=True)
            llm_model = st.text_input("LLM 모델", value=DEFAULT_LLM_MODEL)
            llm_cache = st.text_input("LLM 캐시", value=str(DEFAULT_LLM_CACHE))
            llm_offline = st.checkbox("LLM 로컬 캐시만 사용", value=True)
            index_path = st.text_input("FAISS index", value=str(index_default))
            metadata_path = st.text_input("FAISS metadata", value=str(metadata_default))
            bm25_path = st.text_input("BM25 index", value=str(DEFAULT_CONTEXTUAL_BM25) if retrieval_mode == "hybrid" else "")

    query = st.text_area(
        "질문",
        value="외국인 유학생의 경우 기숙사를 제공받을 수 있나요?",
        height=90,
    )
    run_clicked = st.button("검색 및 답변", type="primary", use_container_width=True)

    if not run_clicked:
        return
    if not query.strip():
        st.warning("질문이 비어 있습니다.")
        return

    setup_hf_cache(embedding_cache, local_files_only)

    try:
        retrieval_started = time.time()
        index, metadata, embedder = get_retriever(
            index_path,
            metadata_path,
            embedding_model,
            embedding_device,
            int(max_seq_length),
            embedding_cache,
            local_files_only,
        )
        bm25_index = get_bm25_index(bm25_path) if retrieval_mode == "hybrid" else None
        results = retrieve(
            query.strip(),
            index,
            metadata,
            embedder,
            int(top_k),
            retrieval_mode=retrieval_mode,
            bm25_index=bm25_index,
            dense_weight=float(dense_weight),
            candidate_k=int(candidate_k),
        )
        retrieval_elapsed = time.time() - retrieval_started
    except Exception as exc:
        st.error(f"검색 인덱스 로드 또는 검색 중 오류가 발생했습니다: {exc}")
        return

    answer = ""
    generated_tokens = None
    generation_elapsed = None
    tokens_per_second = None

    if use_llm:
        try:
            prompt = build_prompt(query.strip(), results)
            tokenizer, model = get_llm(llm_model, quantization, llm_cache, llm_offline)
            generation_started = time.time()
            answer, generated_tokens = generate_answer_with_metrics(
                prompt,
                tokenizer,
                model,
                int(max_new_tokens),
            )
            generation_elapsed = time.time() - generation_started
            if generation_elapsed > 0 and generated_tokens is not None:
                tokens_per_second = generated_tokens / generation_elapsed
        except Exception as exc:
            st.error(f"LLM 답변 생성 중 오류가 발생했습니다: {exc}")
            answer = simple_grounded_answer(query.strip(), results)
    else:
        answer = simple_grounded_answer(query.strip(), results)

    left, right = st.columns([1.05, 1.2], gap="large")
    with left:
        st.subheader("답변")
        render_answer(answer)
    with right:
        st.subheader("검색 후보")
        render_candidates(results)

    st.subheader("근거 조항")
    render_evidence(results)


if __name__ == "__main__":
    main()
