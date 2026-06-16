#!/usr/bin/env python3
"""보고서용 시각화 그림 생성 스크립트 (500 검색 / 100 생성 평가 기준).

파일 번호는 보고서 등장 순서와 일치한다:
    figure1_corpus_stats        (3장 데이터)
    figure2_retrieval_hit       (9.1 검색 Hit@k, 500 QA)
    figure3_mrr                 (9.1 검색 MRR, 500 QA)
    figure4_gold_rank           (9.1 Contextual Hybrid 정답 순위 분포, 500 QA)
    figure5_rank_tradeoff       (9.1 Baseline→Hybrid 질문별 순위 변화, 500 QA)
    figure6_quant_quality       (9.2 양자화 품질, 100 QA)
    figure7_quant_efficiency    (9.2 양자화 효율, 100 QA)

사용법:
    /home/wq1880/miniconda3/envs/myenv/bin/python scripts/make_report_figures.py
산출물: reports/figures/figureN_*.png (300 dpi)
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rcParams


def set_korean_font():
    try:
        import koreanize_matplotlib  # noqa: F401

        rcParams["axes.unicode_minus"] = False
        return
    except Exception:
        pass
    for path in [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    rcParams["axes.unicode_minus"] = False


set_korean_font()

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"
EVAL = ROOT / "data" / "eval"
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C_BASE, C_CTX, C_HYB = "#9fb3c8", "#6b8cae", "#1f4e79"
C_ACC = "#e07b39"


def loadj(path):
    return json.load(open(path, encoding="utf-8"))


def load_results(path):
    return {json.loads(l)["id"]: json.loads(l) for l in open(path, encoding="utf-8") if l.strip()}


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path.relative_to(ROOT))


def figure1_corpus_stats():
    rep = loadj(META / "contextual_chunk_report.json")
    labels = ["규정 파일", "조항 청크", "부칙 후보"]
    vals = [rep["files"], rep["total_articles"], rep["total_addenda"]]
    fig, ax = plt.subplots(figsize=(6.2, 4))
    bars = ax.bar(labels, vals, color=[C_BASE, C_HYB, C_ACC], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.01, f"{v:,}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("개수")
    ax.set_title("그림 1. 말뭉치 구축 결과")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "figure1_corpus_stats.png")


def _retrieval_data():
    return [d for d in loadj(META / "retrieval_comparison_contextual_500.json") if d.get("status") == "ok"]


def figure2_retrieval_hit():
    data = _retrieval_data()
    ks = ["hit_at_1", "hit_at_3", "hit_at_5"]
    klabels = ["Hit@1", "Hit@3", "Hit@5"]
    colors = [C_BASE, C_CTX, C_HYB]
    x = np.arange(len(ks))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, d in enumerate(data):
        vals = [d[k] for k in ks]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=d["name"], color=colors[i % len(colors)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(klabels)
    ax.set_ylim(0.85, 1.01)
    ax.set_ylabel("Hit@k")
    ax.set_title("그림 2. 검색 방식별 Hit@k 비교 (500 QA)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "figure2_retrieval_hit.png")


def figure3_mrr():
    data = _retrieval_data()
    names = [d["name"] for d in data]
    mrr = [d["mrr"] for d in data]
    fig, ax = plt.subplots(figsize=(6.2, 4))
    bars = ax.bar(names, mrr, color=[C_BASE, C_CTX, C_HYB], width=0.55)
    for b, v in zip(bars, mrr):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0002, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0.940, 0.950)
    ax.set_ylabel("MRR")
    ax.set_title("그림 3. 검색 방식별 MRR 비교 (500 QA, y축 확대)")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=8)
    save(fig, "figure3_mrr.png")


def figure4_gold_rank():
    rows = load_results(EVAL / "eval_results_contextual_hybrid_w09_500.jsonl")
    counts = Counter()
    for r in rows.values():
        gr = r.get("gold_rank")
        counts["5위 밖" if not gr else str(gr)] += 1
    order = ["1", "2", "3", "4", "5", "5위 밖"]
    labels = [o for o in order if o in counts]
    vals = [counts[o] for o in labels]
    colors = [C_HYB if l != "5위 밖" else C_ACC for l in labels]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("정답 조항이 검색된 순위")
    ax.set_ylabel("질문 수")
    ax.set_title("그림 4. Contextual Hybrid 정답 조항 순위 분포 (500 QA)")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "figure4_gold_rank.png")


def figure5_rank_tradeoff():
    base = load_results(EVAL / "eval_results_baseline_dense_500.jsonl")
    hyb = load_results(EVAL / "eval_results_contextual_hybrid_w09_500.jsonl")
    MISS = 6  # top-5 밖은 6위로 간주
    deltas = []
    for i in base:
        if i not in hyb:
            continue
        b = base[i].get("gold_rank") or MISS
        h = hyb[i].get("gold_rank") or MISS
        if b != h:
            deltas.append(b - h)  # 양수 = Hybrid가 더 상위(개선)
    counter = Counter(deltas)
    xs = sorted(counter)
    vals = [counter[x] for x in xs]
    colors = [C_HYB if x > 0 else C_ACC for x in xs]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar([f"{x:+d}" for x in xs], vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, str(v), ha="center", va="bottom", fontsize=8)
    up = sum(v for x, v in counter.items() if x > 0)
    down = sum(v for x, v in counter.items() if x < 0)
    ax.set_xlabel("순위 변화량 (Baseline 순위 - Hybrid 순위, 양수=개선)")
    ax.set_ylabel("질문 수")
    ax.set_title(f"그림 5. Baseline→Hybrid 질문별 순위 변화 (500 QA, 개선 {up} · 하락 {down})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_HYB), plt.Rectangle((0, 0), 1, 1, color=C_ACC)]
    ax.legend(handles, ["개선", "하락"], fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "figure5_rank_tradeoff.png")


def _quant_data():
    return [d for d in loadj(META / "quantization_comparison_gen100.json") if d.get("status") == "ok"]


def figure6_quant_quality():
    data = _quant_data()
    labels = [d["quantization"].upper() for d in data]
    metrics = ["citation_accuracy", "keyword_recall", "answer_pass_rate"]
    mlabels = ["근거 인용 정확도", "키워드 재현율", "답변 통과율"]
    colors = [C_CTX, C_HYB, C_ACC]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, metric in enumerate(metrics):
        vals = [d[metric] for d in data]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=mlabels[i], color=colors[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.55, 1.0)
    ax.set_ylabel("점수")
    ax.set_title("그림 6. 양자화 수준별 답변 품질 (100 QA)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "figure6_quant_quality.png")


def figure7_quant_efficiency():
    data = _quant_data()
    labels = [d["quantization"].upper() for d in data]
    tps = [d["tokens_per_second"] for d in data]
    vram = [d["max_memory_allocated_gb"] for d in data]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    bars = ax1.bar(x - 0.2, tps, 0.4, label="tokens/sec", color=C_HYB)
    ax1.set_ylabel("tokens/sec", color=C_HYB)
    ax1.tick_params(axis="y", labelcolor=C_HYB)
    for b, v in zip(bars, tps):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=C_HYB)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + 0.2, vram, 0.4, label="VRAM (GB)", color=C_ACC)
    ax2.set_ylabel("VRAM (GB)", color=C_ACC)
    ax2.tick_params(axis="y", labelcolor=C_ACC)
    for b, v in zip(bars2, vram):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=C_ACC)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("그림 7. 양자화 수준별 생성 속도와 VRAM 사용량 (100 QA)")
    save(fig, "figure7_quant_efficiency.png")


if __name__ == "__main__":
    figure1_corpus_stats()
    figure2_retrieval_hit()
    figure3_mrr()
    figure4_gold_rank()
    figure5_rank_tradeoff()
    figure6_quant_quality()
    figure7_quant_efficiency()
    print("\n모든 그림 생성 완료 ->", OUT)
