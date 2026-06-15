#!/usr/bin/env python3
import argparse
import html
import json
from pathlib import Path


COLORS = {
    "ink": "#172033",
    "muted": "#667085",
    "line": "#d0d5dd",
    "surface": "#f8fafc",
    "panel": "#ffffff",
    "teal": "#0f766e",
    "blue": "#2563eb",
    "amber": "#d97706",
    "rose": "#e11d48",
    "green": "#16a34a",
    "violet": "#7c3aed",
    "slate": "#475467",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value):
    return f"{value * 100:.1f}%"


def num(value, digits=2):
    return f"{value:.{digits}f}"


class Svg:
    def __init__(self, width=1280, height=920):
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            "<style>"
            "text{font-family:Inter,'Noto Sans KR','Apple SD Gothic Neo',Arial,sans-serif;fill:#172033}"
            ".title{font-size:30px;font-weight:800}"
            ".subtitle{font-size:15px;fill:#667085}"
            ".panel-title{font-size:20px;font-weight:800}"
            ".label{font-size:13px;fill:#344054}"
            ".small{font-size:12px;fill:#667085}"
            ".value{font-size:13px;font-weight:700;fill:#172033}"
            "</style>",
            f'<rect width="{width}" height="{height}" fill="{COLORS["surface"]}"/>',
        ]

    def rect(self, x, y, w, h, fill, stroke=None, rx=8, opacity=None):
        attrs = [f'x="{x}"', f'y="{y}"', f'width="{w}"', f'height="{h}"', f'rx="{rx}"', f'fill="{fill}"']
        if stroke:
            attrs.append(f'stroke="{stroke}"')
        if opacity is not None:
            attrs.append(f'opacity="{opacity}"')
        self.parts.append(f"<rect {' '.join(attrs)}/>")

    def line(self, x1, y1, x2, y2, stroke, width=1):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}"/>')

    def text(self, x, y, text, cls="", anchor="start", fill=None):
        fill_attr = f' fill="{fill}"' if fill else ""
        class_attr = f' class="{cls}"' if cls else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}"{class_attr}{fill_attr}>{html.escape(str(text))}</text>'
        )

    def panel(self, x, y, w, h, title, subtitle=None):
        self.rect(x, y, w, h, COLORS["panel"], COLORS["line"], rx=8)
        self.text(x + 24, y + 36, title, "panel-title")
        if subtitle:
            self.text(x + 24, y + 59, subtitle, "subtitle")

    def finish(self):
        self.parts.append("</svg>")
        return "\n".join(self.parts) + "\n"


def draw_horizontal_bars(svg, x, y, w, rows, max_value=1.0, value_formatter=pct, color=COLORS["teal"]):
    bar_x = x + 145
    bar_w = w - 235
    bar_h = 18
    gap = 38
    for idx, (label, value) in enumerate(rows):
        yy = y + idx * gap
        svg.text(x, yy + 14, label, "label")
        svg.rect(bar_x, yy, bar_w, bar_h, "#eaecf0", rx=999)
        fill_w = 0 if max_value == 0 else max(0, min(bar_w, bar_w * value / max_value))
        svg.rect(bar_x, yy, fill_w, bar_h, color, rx=999)
        svg.text(bar_x + bar_w + 12, yy + 14, value_formatter(value), "value")


def draw_grouped_quality(svg, x, y, w, h, rows):
    metrics = [
        ("citation_accuracy", "인용 정확도"),
        ("keyword_recall", "키워드 재현율"),
        ("answer_pass_rate", "답변 통과율"),
    ]
    quant_colors = {"fp16": COLORS["blue"], "int8": COLORS["amber"], "int4": COLORS["green"]}
    chart_x = x + 60
    chart_y = y + 42
    chart_w = w - 95
    chart_h = h - 100
    svg.line(chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h, COLORS["line"])
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        yy = chart_y + chart_h - tick * chart_h
        svg.line(chart_x, yy, chart_x + chart_w, yy, "#eef2f6")
        svg.text(chart_x - 10, yy + 4, f"{int(tick * 100)}", "small", anchor="end")

    group_w = chart_w / len(metrics)
    bar_w = 26
    for i, (key, label) in enumerate(metrics):
        gx = chart_x + i * group_w + group_w / 2
        svg.text(gx, chart_y + chart_h + 26, label, "small", anchor="middle")
        offsets = [-32, 0, 32]
        for offset, row in zip(offsets, rows):
            q = row["quantization"]
            value = row[key]
            bh = value * chart_h
            bx = gx + offset - bar_w / 2
            by = chart_y + chart_h - bh
            svg.rect(bx, by, bar_w, bh, quant_colors[q], rx=4)
            svg.text(bx + bar_w / 2, by - 7, pct(value), "small", anchor="middle")

    legend_x = x + 24
    legend_y = y + h - 24
    for i, q in enumerate(["fp16", "int8", "int4"]):
        lx = legend_x + i * 86
        svg.rect(lx, legend_y - 12, 14, 14, quant_colors[q], rx=3)
        svg.text(lx + 20, legend_y, q.upper(), "small")


def draw_efficiency(svg, x, y, w, rows):
    max_tps = max(row["tokens_per_second"] for row in rows)
    max_mem = max(row["max_memory_allocated_gb"] for row in rows)
    max_sec = max(row["avg_generation_seconds"] for row in rows)

    svg.text(x, y, "속도: tokens/sec 높을수록 좋음", "label")
    draw_horizontal_bars(
        svg,
        x,
        y + 16,
        w,
        [(row["quantization"].upper(), row["tokens_per_second"]) for row in rows],
        max_value=max_tps,
        value_formatter=lambda v: f"{v:.1f}",
        color=COLORS["violet"],
    )

    svg.text(x, y + 150, "VRAM: GB 낮을수록 가벼움", "label")
    draw_horizontal_bars(
        svg,
        x,
        y + 166,
        w,
        [(row["quantization"].upper(), row["max_memory_allocated_gb"]) for row in rows],
        max_value=max_mem,
        value_formatter=lambda v: f"{v:.1f} GB",
        color=COLORS["rose"],
    )

    svg.text(x, y + 300, "평균 생성 시간: 초 낮을수록 빠름", "label")
    draw_horizontal_bars(
        svg,
        x,
        y + 316,
        w,
        [(row["quantization"].upper(), row["avg_generation_seconds"]) for row in rows],
        max_value=max_sec,
        value_formatter=lambda v: f"{v:.2f}s",
        color=COLORS["slate"],
    )


def write_markdown(path: Path, retrieval_report: dict, quant_rows: list[dict], svg_path: Path):
    retrieval = retrieval_report["retrieval"]
    lines = [
        "# Evaluation Performance Visualization",
        "",
        f"![Evaluation Performance]({svg_path.name})",
        "",
        "## Retrieval Evaluation",
        "",
        f"- QA count: `{retrieval_report['qa_count']}`",
        f"- Hit@1: `{retrieval['hit_at_1']:.4f}`",
        f"- Hit@3: `{retrieval['hit_at_3']:.4f}`",
        f"- Hit@5: `{retrieval['hit_at_5']:.4f}`",
        f"- MRR: `{retrieval['mrr']:.4f}`",
        "",
        "## Quantization Evaluation",
        "",
        "| quantization | answer pass | citation | keyword recall | tokens/sec | generation sec | max VRAM GB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in quant_rows:
        lines.append(
            "| {q} | {pass_rate:.4f} | {citation:.4f} | {keyword:.4f} | {tps:.4f} | {sec:.4f} | {mem:.4f} |".format(
                q=row["quantization"],
                pass_rate=row["answer_pass_rate"],
                citation=row["citation_accuracy"],
                keyword=row["keyword_recall"],
                tps=row["tokens_per_second"],
                sec=row["avg_generation_seconds"],
                mem=row["max_memory_allocated_gb"],
            )
        )
    lines += [
        "",
        "## Note",
        "",
        "- Retrieval was evaluated on 100 QA pairs.",
        "- EXAONE generation and quantization were evaluated on 25 QA pairs.",
        "- INT8 reduces VRAM but is slower in this batch-1 RAG setting; INT4 is fastest and lightest but has lower answer quality.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_svg(retrieval_report: dict, quant_rows: list[dict]) -> str:
    svg = Svg()
    svg.text(48, 56, "제주대학교 규정 RAG 평가 성능 요약", "title")
    svg.text(
        48,
        84,
        "검색 평가는 100개 QA, 생성/양자화 평가는 25개 QA 기준",
        "subtitle",
    )

    retrieval = retrieval_report["retrieval"]
    svg.panel(48, 116, 520, 270, "검색 성능", "BGE-M3 + FAISS, 100 QA")
    draw_horizontal_bars(
        svg,
        72,
        190,
        460,
        [
            ("Hit@1", retrieval["hit_at_1"]),
            ("Hit@3", retrieval["hit_at_3"]),
            ("Hit@5", retrieval["hit_at_5"]),
            ("MRR", retrieval["mrr"]),
        ],
        max_value=1.0,
        color=COLORS["teal"],
    )

    svg.panel(604, 116, 628, 270, "생성 품질 비교", "EXAONE FP16 / INT8 / INT4, 25 QA")
    draw_grouped_quality(svg, 628, 176, 580, 190, quant_rows)

    svg.panel(48, 418, 1184, 420, "양자화 효율 비교", "속도, VRAM, 평균 생성 시간")
    draw_efficiency(svg, 84, 492, 1040, quant_rows)

    svg.rect(48, 858, 1184, 36, "#ecfeff", "#99f6e4", rx=8)
    svg.text(
        72,
        881,
        "해석: INT4는 가장 빠르고 VRAM이 낮지만 답변 통과율이 하락했고, INT8은 VRAM은 줄었지만 생성 속도는 FP16보다 느렸다.",
        "label",
    )
    return svg.finish()


def main():
    parser = argparse.ArgumentParser(description="Create a one-page SVG visualization for RAG evaluation metrics.")
    parser.add_argument("--retrieval-report", default="data/metadata/eval_report_retrieval_100.json")
    parser.add_argument("--quantization-report", default="data/metadata/quantization_comparison.json")
    parser.add_argument("--output-svg", default="data/metadata/evaluation_performance.svg")
    parser.add_argument("--output-md", default="data/metadata/evaluation_performance.md")
    args = parser.parse_args()

    retrieval_report = load_json(Path(args.retrieval_report))
    quant_rows = [row for row in load_json(Path(args.quantization_report)) if row.get("status") == "ok"]
    order = {"fp16": 0, "int8": 1, "int4": 2}
    quant_rows.sort(key=lambda row: order.get(row["quantization"], 99))

    svg_path = Path(args.output_svg)
    md_path = Path(args.output_md)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(build_svg(retrieval_report, quant_rows), encoding="utf-8")
    write_markdown(md_path, retrieval_report, quant_rows, svg_path)
    print(f"svg={svg_path}")
    print(f"markdown={md_path}")


if __name__ == "__main__":
    main()
