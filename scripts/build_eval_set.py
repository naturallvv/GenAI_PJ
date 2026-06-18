#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / "data" / "processed" / "articles_contextual.jsonl"
SEED_QA = ROOT / "data" / "eval" / "qa_seed.jsonl"

TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")

# 규정 텍스트 기능어/형식어 — 키워드로 부적절한 토큰
STOPWORDS = {
    "이다", "한다", "있다", "없다", "된다", "하는", "하여", "하고", "하지", "또는", "및", "그", "이",
    "등", "각", "수", "것", "바", "자", "중", "때", "해당", "다음", "경우", "관한", "대한", "의한",
    "위한", "따라", "따른", "이내", "이상", "이하", "까지", "부터", "으로", "에서", "에게", "같은",
    "모든", "별표", "별지", "서식", "제호", "규정", "규칙", "조항", "조의", "본문", "단서", "신설",
    "개정", "삭제", "공포", "시행", "부칙", "총장", "대학교", "제주", "제주대학교", "위원회", "위원",
    "위원장", "사항", "기준", "관리", "운영", "업무", "처리", "필요", "정한", "정하는", "정하여",
    "둘", "이를", "그에", "이에", "대하여", "관하여", "있어서", "아니", "아니한", "아니하는",
}

# 질문화에 부적절한 제네릭 제목
GENERIC_TITLES = {
    "목적", "정의", "적용범위", "시행", "경과조치", "삭제", "비밀유지", "준용", "위임", "기타",
    "재검토", "해석", "효력", "적용", "용어의 정의", "용어의정의",
}


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


# HWP 표/별표 추출 잔여물: CJK 한자 잔여, 박스드로잉, 사유표시 기호 등
MOJIBAKE_RE = re.compile(r"[㐀-鿿─-╿①-⓿�]+")


def clean_text(text: str) -> str:
    """표 추출 시 끼어든 깨진 문자(한자 잔여·박스문자)를 제거하고 공백 정리."""
    if not text:
        return ""
    text = MOJIBAKE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_header(text: str, article_no: str, article_title: str) -> str:
    header = re.escape(article_no)
    if article_title:
        pattern = rf"^\s*{header}\s*[（(]\s*{re.escape(article_title)}\s*[）)]\s*"
    else:
        pattern = rf"^\s*{header}\s*"
    body = re.sub(pattern, "", text, count=1)
    return clean_text(body)


def is_substantive(article: dict) -> bool:
    body = (article.get("text") or "").strip()
    if len(body) < 50:
        return False
    if "삭제" in body and len(body) < 25:
        return False
    return True


# 부칙류·개정/폐지 래퍼 제목 — 여러 규정에 반복되거나 내용이 없어 질문 대상으로 부적절
ADDENDUM_TITLES = {
    "시행일", "경과조치", "적용례", "유효기간", "재검토 기한",
}
# 본문이 사실상 비어있는 개정·폐지·생략 래퍼 조항 탐지
BOILERPLATE_BODY_RE = re.compile(r"다음과\s*같이\s*개정한다|\(\s*생략\s*\)|이를\s*폐지한다|를\s*폐지한다")


def is_clean_eval_article(article: dict, text_rule_count: dict, norm_text) -> bool:
    """평가셋에 부적절한 조항 제외: 폐지규정·중복본문·부칙/개정 래퍼."""
    if "폐지" in article.get("rule_name", ""):
        return False
    title = (article.get("article_title") or "").strip()
    if title in ADDENDUM_TITLES or "다른 규정" in title or "다른규정" in title:
        return False
    body = article.get("text") or ""
    if BOILERPLATE_BODY_RE.search(body) and len(body.strip()) < 160:
        return False
    # 동일 본문이 2개 이상 규정에 등장하면 검색 구분 불가 → 제외
    if text_rule_count.get(norm_text(body), 0) >= 2:
        return False
    return True


def article_quality(article: dict) -> tuple:
    """규정 내에서 어떤 조항을 우선 뽑을지 결정하는 점수 (높을수록 우선)."""
    title = (article.get("article_title") or "").strip()
    body_len = len((article.get("text") or "").strip())
    specific_title = 1 if (title and title not in GENERIC_TITLES and len(title) >= 3) else 0
    # 너무 긴 조항(별표 덩어리)은 약간 감점, 적당한 길이 선호
    length_score = min(body_len, 600)
    return (specific_title, length_score)


def build_idf(articles: list[dict]) -> dict:
    doc_freq = Counter()
    for article in articles:
        terms = {t for t in TOKEN_RE.findall(article.get("text") or "") if len(t) >= 2}
        doc_freq.update(terms)
    n = len(articles)
    return {term: math.log((n + 1) / (freq + 1)) + 1.0 for term, freq in doc_freq.items()}


# 활용형/조사로 끝나 키워드로 부적절한 토큰
BAD_SUFFIX_RE = re.compile(
    r"(하는|하여|하고|하며|하지|받아야|되는|된다|한다|이라|이며|에서|으로|에게|에는|까지|부터|"
    r"으며|로서|로써|토록|면서|에도|와의|과의|이고|라는|라고|에의|만을|만의|등을|등의|"
    r"하려는|하려|보다|위해|통해|따라|대해|관해|있어|없이)$"
)


def is_good_keyword(term: str) -> bool:
    if len(term) < 2 or term.isdigit():
        return False
    if term in STOPWORDS:
        return False
    if BAD_SUFFIX_RE.search(term):
        return False
    # 'A로', 'B의' 같은 단음절 라틴문자+조사 잔여물 제외
    if len(term) <= 2 and re.search(r"[A-Za-z]", term):
        return False
    return True


def strip_josa(term: str) -> str:
    """명사 내부 글자를 깨지 않는 범위에서 조사를 제거한다."""
    # 단음절 조사: 어간 2자 이상이면 제거(을/를/은/는/에/과/와/도/만)
    if len(term) >= 3 and term[-1] in "을를은는에과와도만":
        return term[:-1]
    # 의: 어간 2자 이상이면 제거(원장의→원장, 위원회의→위원회). 회의·정의(2자)는 보존
    if len(term) >= 3 and term[-1] == "의":
        return term[:-1]
    # 이/로: 짧은 명사(어린이·진로 등)를 깰 위험이 있어 어간 3자 이상일 때만 제거
    if len(term) >= 4 and term[-1] in "이로":
        return term[:-1]
    return term


def dedupe_keywords(keywords: list[str]) -> list[str]:
    """한 토큰이 다른 토큰+조사 형태면(예: 운영위원회 vs 운영위원회는) 짧은 명사형만 남긴다."""
    result = []
    for kw in keywords:
        redundant = False
        for other in keywords:
            if other != kw and kw.startswith(other) and len(kw) - len(other) <= 2:
                redundant = True
                break
        if not redundant and kw not in result:
            result.append(kw)
    return result


def extract_keywords(article: dict, idf: dict, top_n: int = 4) -> list[str]:
    title = (article.get("article_title") or "").strip()
    body = strip_header(article.get("text") or "", article.get("article_no", ""), title)
    tf = Counter(t for t in TOKEN_RE.findall(body) if is_good_keyword(t))
    scored = [(term, freq * idf.get(term, 1.0)) for term, freq in tf.items()]
    scored.sort(key=lambda item: item[1], reverse=True)
    keywords = [strip_josa(term) for term, _ in scored[: top_n + 3]]
    # 제목의 핵심어를 앞에 보강(제네릭 제외)
    if title and title not in GENERIC_TITLES:
        for token in TOKEN_RE.findall(title):
            if is_good_keyword(token) and strip_josa(token) not in keywords:
                keywords.insert(0, strip_josa(token))
                break
    return dedupe_keywords([k for k in keywords if len(k) >= 2])[:top_n]


def extract_reference(article: dict, max_len: int = 220) -> str:
    title = (article.get("article_title") or "").strip()
    body = strip_header(article.get("text") or "", article.get("article_no", ""), title)
    # 첫 문장 또는 max_len 까지
    sentences = re.split(r"(?<=[.다])\s+", body)
    ref = ""
    for sentence in sentences:
        if not sentence.strip():
            continue
        ref = (ref + " " + sentence).strip() if ref else sentence.strip()
        if len(ref) >= max_len:
            break
    if len(ref) > max_len + 60:
        ref = ref[: max_len + 60].rstrip() + "…"
    return ref


def gen_suitable(article: dict, keywords: list[str]) -> bool:
    """생성평가에 적합한가: 키워드 충분 + 적당한 길이 + 사실형."""
    body_len = len((article.get("text") or "").strip())
    title = (article.get("article_title") or "").strip()
    return (
        len(keywords) >= 3
        and 60 <= body_len <= 1200
        and title not in GENERIC_TITLES
    )


def balanced_sample(pool: list[dict], target: int, per_rule_cap: int, seed: int) -> list[dict]:
    """라운드로빈으로 규정 커버리지를 최대화하며 카테고리 균형을 맞춘다."""
    import random

    rng = random.Random(seed)

    # 규정별로 후보 정리 + 품질 정렬
    by_rule = defaultdict(list)
    for article in pool:
        by_rule[article["rule_name"]].append(article)
    for rule in by_rule:
        by_rule[rule].sort(key=article_quality, reverse=True)

    # 규정을 카테고리별로 묶어 카테고리 라운드로빈 → 그 안에서 규정 라운드로빈
    cat_of_rule = {a["rule_name"]: (a.get("category") or "기타") for a in pool}
    rules_by_cat = defaultdict(list)
    for rule in by_rule:
        rules_by_cat[cat_of_rule[rule]].append(rule)
    for cat in rules_by_cat:
        rng.shuffle(rules_by_cat[cat])
    categories = sorted(rules_by_cat)
    rng.shuffle(categories)

    selected = []
    cursor = {rule: 0 for rule in by_rule}
    taken_per_rule = Counter()

    # pass 1..per_rule_cap: 각 카테고리의 각 규정에서 한 개씩 순환
    for _pass in range(per_rule_cap):
        progressed = False
        for cat in categories:
            for rule in rules_by_cat[cat]:
                if len(selected) >= target:
                    return selected
                if taken_per_rule[rule] >= per_rule_cap:
                    continue
                idx = cursor[rule]
                if idx >= len(by_rule[rule]):
                    continue
                selected.append(by_rule[rule][idx])
                cursor[rule] = idx + 1
                taken_per_rule[rule] += 1
                progressed = True
        if not progressed:
            break
    return selected


def main():
    parser = argparse.ArgumentParser(description="평가셋 확장용 후보 샘플링 + 라벨 추출")
    parser.add_argument("--new", type=int, default=400, help="신규로 뽑을 후보 조항 수")
    parser.add_argument("--per-rule-cap", type=int, default=4)
    parser.add_argument("--gen-target", type=int, default=130, help="생성평가 후보로 표시할 수")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worksheet", default="data/eval/authoring_worksheet.md")
    parser.add_argument("--candidates", default="data/eval/eval_candidates.jsonl")
    args = parser.parse_args()

    articles = load_jsonl(CTX)
    idf = build_idf(articles)

    # 기존 qa_seed 이 다루는 (규정, 조) 제외
    covered = set()
    if SEED_QA.exists():
        for qa in load_jsonl(SEED_QA):
            for c in qa.get("expected_citations", []):
                covered.add((re.sub(r"\s+", "", c.get("rule_name", "")), c.get("article_no", "")))

    def key_of(a):
        return (re.sub(r"\s+", "", a["rule_name"]), a["article_no"])

    def norm_text(t):
        return re.sub(r"\s+", "", t or "")

    # 동일 본문이 몇 개 '규정'에 등장하는지 — 중복(구분불가) 본문 탐지
    text_rules = defaultdict(set)
    for a in articles:
        if is_substantive(a):
            text_rules[norm_text(a.get("text"))].add(a["rule_name"])
    text_rule_count = {t: len(rs) for t, rs in text_rules.items()}

    pool = [
        a
        for a in articles
        if is_substantive(a)
        and key_of(a) not in covered
        and is_clean_eval_article(a, text_rule_count, norm_text)
    ]
    # 동일 (규정,조) 중복 제거 — 첫 등장만
    seen = set()
    unique_pool = []
    for a in pool:
        k = key_of(a)
        if k in seen:
            continue
        seen.add(k)
        unique_pool.append(a)
    pool = unique_pool

    selected = balanced_sample(pool, args.new, args.per_rule_cap, args.seed)

    # 라벨 부착
    candidates = []
    for i, article in enumerate(selected, start=1):
        keywords = extract_keywords(article, idf)
        cand = {
            "id": f"qa_n{i:03d}",
            "rule_name": article["rule_name"],
            "article_no": article["article_no"],
            "article_title": article.get("article_title", ""),
            "category": article.get("category", ""),
            "expected_citations": [
                {"rule_name": article["rule_name"], "article_no": article["article_no"]}
            ],
            "expected_keywords": keywords,
            "reference_answer": extract_reference(article),
            "gen_candidate": gen_suitable(article, keywords),
            "source_text": clean_text(article.get("text") or "")[:600],
        }
        candidates.append(cand)

    # 생성 후보 상한 표시(품질 좋은 것 위주는 이미 article_quality로 앞쪽에 옴)
    gen_marked = 0
    for cand in candidates:
        if cand["gen_candidate"] and gen_marked < args.gen_target:
            gen_marked += 1
        else:
            cand["gen_candidate"] = False

    # 출력: 머신용 jsonl
    cand_path = ROOT / args.candidates
    cand_path.parent.mkdir(parents=True, exist_ok=True)
    with cand_path.open("w", encoding="utf-8") as handle:
        for cand in candidates:
            handle.write(json.dumps(cand, ensure_ascii=False) + "\n")

    # 출력: 작성 워크시트(md)
    ws_path = ROOT / args.worksheet
    cat_dist = Counter(c["category"] for c in candidates)
    rule_dist = Counter(c["rule_name"] for c in candidates)
    lines = [
        "# 질문 작성 워크시트",
        "",
        f"- 신규 후보: {len(candidates)}개  /  생성평가 표시: {gen_marked}개",
        f"- 다루는 규정 수: {len(rule_dist)}  /  카테고리: {len(cat_dist)}",
        "",
        "## 카테고리 분포",
        "",
    ]
    for cat, n in cat_dist.most_common():
        lines.append(f"- {cat}: {n}")
    lines += ["", "## 후보 (각 항목에 자연스러운 한국어 질문 1개씩 작성)", ""]
    for cand in candidates:
        gen = " [GEN]" if cand["gen_candidate"] else ""
        title = f"({cand['article_title']})" if cand["article_title"] else ""
        lines.append(f"### {cand['id']}{gen}  |  {cand['rule_name']} {cand['article_no']}{title}")
        lines.append(f"- 카테고리: {cand['category']}")
        lines.append(f"- 키워드: {', '.join(cand['expected_keywords'])}")
        lines.append(f"- 본문: {cand['source_text'][:300]}")
        lines.append("")
    ws_path.parent.mkdir(parents=True, exist_ok=True)
    ws_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"신규 후보: {len(candidates)}개")
    print(f"다루는 규정: {len(rule_dist)}개, 카테고리: {len(cat_dist)}개")
    print(f"생성평가 표시: {gen_marked}개")
    print(f"카테고리 분포: {dict(cat_dist.most_common())}")
    print(f"\n워크시트: {ws_path.relative_to(ROOT)}")
    print(f"머신용 후보: {cand_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
