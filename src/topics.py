"""Assign every article to one of the 5 topic categories of the brief.

The brief gives English indicative keywords per category. The corpus is Urdu,
so each seed keyword is translated and expanded with close Urdu synonyms used
by BBC Urdu. Every keyword is passed through the *Assignment 1* pipeline
(normalise -> tokenize -> lemmatize -> stem) so that it matches the stemmed
tokens of cleaned.txt exactly; multi-word keywords are matched as n-grams.

Score of category c for an article (all three signals come from the data):

    S_c = body_c + title_c + 0.5 * tags_c

    body_c  : keyword hits per 100 tokens of the cleaned article body
    title_c : keyword hits in the (A1-normalised) headline
    tags_c  : BBC's own topic tags (Metadata.json `category` + `topics`)
              mapped to c; the primary `category` tag counts twice

The article goes to argmax_c S_c (ties -> body_c); 9 headline-checked
corrections are listed in OVERRIDES with the reason for each. Geographic tags such as
"پاکستان" or "نائجیریا" are deliberately not mapped: where a story happens
does not say what it is about.
"""
from __future__ import annotations

import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from a1_bridge import a1_normalize  # noqa: E402
from corpus import load_cleaned_articles, load_metadata  # noqa: E402

# Seed keywords of the brief (first line of each list) + Urdu expansions.
KEYWORDS: dict[str, list[str]] = {
    "Politics": [
        # election, government, minister, parliament
        "انتخابات", "الیکشن", "حکومت", "حکومتی", "وزیر", "وزیراعظم", "وزیر اعظم",
        "پارلیمان", "پارلیمنٹ", "پارلیمانی",
        "انتخاب", "ووٹ", "ووٹر", "وزیراعلیٰ", "وزیر اعلیٰ", "وزارت", "اسمبلی",
        "سینیٹ", "سینیٹر", "کابینہ", "سیاست", "سیاسی", "سیاستدان", "اپوزیشن",
        "جماعت", "پارٹی", "تحریک انصاف", "مسلم لیگ", "پیپلز پارٹی", "آئین",
        "آئینی", "ترمیم", "اقتدار", "جلسہ", "لانگ مارچ", "رکن پارلیمان",
        "نظر بند", "بدعنوانی", "کرپشن", "قومی خزانے", "عدالت", "سپریم کورٹ",
        "ہائی کورٹ", "صحافت", "صحافی", "سینسر",
    ],
    "Sports": [
        # cricket, match, team, player, score
        "کرکٹ", "میچ", "ٹیم", "کھلاڑی", "سکور",
        "کھلاڑیوں", "رنز", "وکٹ", "وکٹیں", "بیٹنگ", "بولنگ", "بولر", "بیٹسمین",
        "بلے باز", "کپتان", "کپتانی", "ٹیسٹ", "ون ڈے", "ٹی ٹوئنٹی", "سیریز",
        "ٹورنامنٹ", "ورلڈ کپ", "ایشیا کپ", "فٹبال", "ہاکی", "اولمپکس", "گول",
        "کوچ", "سکواڈ", "اننگز", "سنچری", "چیمپیئن", "فائنل", "کھیل", "سٹیڈیم",
        "ایتھلیٹ", "تمغہ", "تمغے", "اوپنر", "کرکٹر",
    ],
    "Economy": [
        # inflation, trade, bank, GDP, budget
        "مہنگائی", "افراط زر", "تجارت", "تجارتی", "بینک", "جی ڈی پی", "بجٹ",
        "معیشت", "معاشی", "اقتصادی", "سرمایہ", "سرمایہ کاری", "سرمایہ کار",
        "قرض", "قرضہ", "آئی ایم ایف", "روپے", "ڈالر", "قیمت", "قیمتیں", "ٹیکس",
        "برآمد", "برآمدات", "درآمد", "درآمدات", "کاروبار", "کاروباری", "کمپنی",
        "مارکیٹ", "منڈی", "سٹاک", "حصص", "تیل", "پیٹرول", "پیٹرولیم", "ڈیزل",
        "ایندھن", "شرح سود", "کرنسی", "سبسڈی", "صنعت", "تاجر", "منافع",
        "آمدن", "لیوی", "ریفائنری", "بیرل", "سپلائی", "ارب پتی",
    ],
    "International": [
        # UN, treaty, foreign, bilateral, conflict
        "اقوام متحدہ", "معاہدہ", "معاہدے", "خارجہ", "غیر ملکی", "دو طرفہ",
        "تنازع", "تنازعہ",
        "دفتر خارجہ", "وزیر خارجہ", "جنگ", "جنگی", "کشیدگی", "سفارتی",
        "سفارتکار", "سفیر", "بین الاقوامی", "عالمی", "نیٹو", "حملہ", "حملے",
        "فوج", "فوجی", "میزائل", "ڈرون", "جوہری", "پابندیاں", "سرحد", "حوثی",
        "طالبان", "القاعدہ", "مذاکرات", "جنگ بندی", "اتحاد", "اتحادی", "اسلحہ",
        "طیارہ", "فضائیہ", "بحریہ", "پینٹاگون", "وائٹ ہاؤس", "دہشت گرد",
    ],
    "Health & Society": [
        # hospital, disease, vaccine, flood, education
        "ہسپتال", "بیماری", "ویکسین", "سیلاب", "تعلیم",
        "تعلیمی", "مرض", "امراض", "سکول", "کالج", "یونیورسٹی", "طلبہ",
        "طالبہ", "طالبعلم", "استاد", "اساتذہ", "صحت", "ڈاکٹر", "علاج", "مریض",
        "دوا", "ادویات", "کینسر", "حمل", "زچگی", "بچہ", "بچی", "بچے", "خواتین",
        "خاتون", "عورت", "معاشرہ", "سماجی", "تشدد", "جنسی", "ریپ", "ہراسانی",
        "شادی", "خاندان", "طلاق", "پولیس", "قتل", "ملزم", "فلم", "ڈرامہ",
        "موسیقی", "فنکار", "اداکار", "اداکارہ", "ثقافت", "مذہب", "سوشل میڈیا",
        "انٹرنیٹ", "مصنوعی ذہانت", "سائنس", "سائنسدان", "نرس",
    ],
}

# BBC topic tags (Metadata.json) that are about a subject, not a place.
TAG_MAP: dict[str, str] = {}
for _tag in ["سیاست", "پاکستان تحریکِ انصاف", "عمران خان", "پاکستان مسلم لیگ",
             "مریم نواز", "بھارتیہ جنتا پارٹی", "نریندر مودی", " شہباز شریف",
             "پرویز مشرف", "بدعنوانی", "احتجاج", "صحافتی آزادی", "سینسر شپ",
             "آزادئ اظہارِ رائے", "عدالتیں", "جج", "ڈونلڈ ٹرمپ", "اقلیتوں کے حقوق"]:
    TAG_MAP[_tag] = "Politics"
for _tag in ["کھیل", "کرکٹ", "پاکستانی کرکٹ ٹیم", "انگلش کرکٹ ٹیم", "انڈین کرکٹ ٹیم",
             "فٹبال", "ہاکی", "ایتھلیٹکس", "پاکستان بمقابلہ انڈیا", "جوا"]:
    TAG_MAP[_tag] = "Sports"
for _tag in ["معیشت", "افراطِ زر", "تیل و گیس کی صنعت", "کاروبار", "بازارِ حصص",
             "تجارت", "بینکاری", "اخراجاتِ زندگی", "توانائی", "کپڑوں کی صنعت",
             "ملازمتیں", "کاروبار اور انتظام", "ای کامرس"]:
    TAG_MAP[_tag] = "Economy"
for _tag in ["ایران - اسرائیل، امریکہ جنگ", "جنگ", "یوکرین روس تنازعہ", "مشرقِ وسطٰی",
             "القاعدہ", "گیارہ ستمبر 2001 کے حملے", "جوہری توانائی", "جوہری ہتھیار",
             "حماس", "غزہ", "فلسطینی علاقے", "فلسطین", "انسانی امداد", "طالبان",
             "پاسدارانِ انقلاب", "جنگِ عظیم دوم", "انڈیا اور چین کا سرحدی تنازع",
             "انڈیا پاکستان فضائی حملے", "2025 ایران اسرائیل تنازعہ", "دولتِ اسلامیہ",
             "ڈرون حملے", "جاسوسی", "خلیج فارس", "وزارتِ دفاع", "فوج", "پاکستانی فوج",
             "مسئلہ کشمیر", "ولادیمیر پوتن", "شی جن پنگ", "ایمانویل میکخواں",
             "باہری خلا", "خلائی تحقیق"]:
    TAG_MAP[_tag] = "International"
for _tag in ["صحت", "طب", "خواتین کی صحت", "کینسر", "طبی تحقیق",
             "اولاد پیدا کرنے کی صلاحیت", "موٹاپا", "غذائیت", "تعلیم", "سیلاب",
             "خراب موسم", "ماحولیاتی تبدیلی", "ماحولیات", "فطرت", "قدرت کا تحفظ",
             "بچے", "خواتین", "خواتین کے حقوق", "جنسی تشدد", "گھریلو تشدد", "جرم",
             "پولیس", "حقوقِ انسانی", "معاشرہ", "شادی", "خاندان", "ثقافت", "مذہب",
             "فن فنکار", "فلم", "موسیقی", "ٹی وی", "ادب", "کتب", "بالی وڈ", "آرٹ",
             "سوشل میڈیا", "سائبر کرائم", "ترکِ وطن", "پناہ گزین اور تارکینِ وطن", "غربت", "فیشن", "سائبر حملہ",
             "مصنوعی ذہانت", "ٹیکنالوجی", "سائنس", "کمپیوٹر سکیورٹی", "انٹرنیٹ",
             "جانور", "جانوروں کی فلاح", "نسل پرستی", "لائف سٹائل", "مسلمان",
             "عیسائیت", "رمضان", "نوجوان", "سیکس", "تشدد", "ٹرانسپورٹ", "سفر",
             "جعلی خبریں", "فوڈ", "فوڈ سیفٹی", "کریئر",
             "جیل", "اشتہارات", "موبائل فون، سمارٹ فون",
             "ایپل", "گوگل", "کمپیوٹنگ", "شیعہ"]:
    TAG_MAP[_tag] = "Health & Society"

TAG_WEIGHT = 0.5

# Manual corrections after reading the headline of every article. Each one is a
# keyword collision or a story whose keywords point away from its subject.
OVERRIDES: dict[int, tuple[str, str]] = {
    30: ("Health & Society", "royal-family human-interest story, not politics"),
    107: ("International", "Xi Jinping - Pezeshkian meeting: diplomacy, not oil trade"),
    135: ("Health & Society", "'negotiations' are ransom talks with kidnappers (crime)"),
    172: ("Politics", "'جنگ' is the newspaper Jang, not 'war'; press-freedom story"),
    199: ("International", "US visa-bond policy for foreign nationals; 'dollar' is the bond"),
    203: ("International", "Ahmadinejad accused of links with Israel (Iran-Israel war)"),
    207: ("Economy", "Google opens its first office in Karachi: investment/business"),
    211: ("Economy", "launch of Pakistan's largest data centre: investment/business"),
    231: ("International", "9/11 accused Ramzi al-Shibh before a US military court"),
}


@lru_cache(maxsize=1)
def normalized_keywords() -> dict[str, list[tuple[str, ...]]]:
    """Keywords after the A1 pipeline, as token tuples (deduplicated)."""
    out = {}
    for cat, words in KEYWORDS.items():
        seen, grams = set(), []
        for w in words:
            g = tuple(a1_normalize(w))
            if g and g not in seen:
                seen.add(g)
                grams.append(g)
        out[cat] = grams
    return out


def count_hits(tokens: list[str], grams: list[tuple[str, ...]]) -> int:
    """Number of (possibly overlapping-free, longest-first) keyword matches."""
    by_len = sorted({len(g) for g in grams}, reverse=True)
    gset = set(grams)
    i, hits = 0, 0
    while i < len(tokens):
        for n in by_len:
            if tuple(tokens[i:i + n]) in gset:
                hits += 1
                i += n
                break
        else:
            i += 1
    return hits


def score_article(tokens: list[str], title: str, tags: list[str],
                  primary: str) -> dict[str, dict[str, float]]:
    kw = normalized_keywords()
    title_toks = a1_normalize(title)
    n = max(len(tokens), 1)
    res = {}
    for cat in C.CATEGORIES:
        body = 100.0 * count_hits(tokens, kw[cat]) / n
        ttl = float(count_hits(title_toks, kw[cat]))
        tag = sum((2.0 if t == primary else 1.0) for t in tags if TAG_MAP.get(t) == cat)
        res[cat] = {"body": body, "title": ttl, "tags": tag,
                    "total": body + ttl + TAG_WEIGHT * tag}
    return res


@lru_cache(maxsize=1)
def assign_categories() -> list[dict]:
    """One record per article: number, label, label_id and the score table."""
    meta = load_metadata()
    arts = load_cleaned_articles()
    out = []
    for m, art in zip(meta, arts):
        toks = [t for s in art for t in s]
        tags = list(dict.fromkeys([m["category"]] + m["topics"]))
        sc = score_article(toks, m["title"], tags, m["category"])
        auto = max(C.CATEGORIES, key=lambda c: (sc[c]["total"], sc[c]["body"]))
        label, reason = OVERRIDES.get(m["article_number"], (auto, ""))
        out.append({"article_number": m["article_number"], "title": m["title"],
                    "bbc_category": m["category"], "auto_label": auto,
                    "label": label, "label_id": C.CATEGORIES.index(label),
                    "override_reason": reason, "scores": sc})
    return out


def labels() -> list[int]:
    return [r["label_id"] for r in assign_categories()]


def distribution() -> Counter:
    return Counter(r["label"] for r in assign_categories())


if __name__ == "__main__":
    recs = assign_categories()
    print(distribution())
    for r in recs:
        s = r["scores"][r["label"]]
        print(r["article_number"], r["label"], f"{s['total']:.2f}", r["title"][:60])
