# -*- coding: utf-8 -*-
"""
Custom Urdu preprocessing toolkit - built from scratch (no NLP libraries).

Pipeline order matters and is applied in exactly this sequence:

    normalize_unicode -> remove_diacritics -> remove_noise -> remove_non_urdu
    -> normalize_whitespace -> segment_sentences -> tokenize -> lemmatize -> stem

Every stage is a plain function so it can be demonstrated and graded in isolation
(see notebook.ipynb) and reused by build_dataset.py.
"""
import re
import unicodedata

# --------------------------------------------------------------------------
# 1. UNICODE NORMALIZATION
# --------------------------------------------------------------------------
# BBC Urdu mixes Arabic-script codepoints with the Urdu-specific ones. The same
# visual letter can therefore arrive as two different codepoints, which would
# split one word into two vocabulary entries. We fold everything onto the Urdu
# (Nastaliq) canonical form.
CHAR_MAP = {
    "ي": "ی",  # ARABIC YEH        -> FARSI YEH      ي -> ی
    "ى": "ی",  # ALEF MAKSURA      -> FARSI YEH      ى -> ی
    "ۍ": "ی",  # YEH WITH TAIL     -> FARSI YEH
    "ك": "ک",  # ARABIC KAF        -> KEHEH          ك -> ک
    "ڪ": "ک",  # SWASH KAF         -> KEHEH
    "ه": "ہ",  # ARABIC HEH        -> HEH GOAL       ه -> ہ
    "ۃ": "ہ",  # TEH MARBUTA GOAL  -> HEH GOAL
    "ة": "ہ",  # TEH MARBUTA       -> HEH GOAL       ة -> ہ
    "ۂ": "ہ",  # HEH GOAL + HAMZA  -> HEH GOAL
    "أ": "ا",  # ALEF WITH HAMZA ABOVE -> ALEF        أ -> ا
    "إ": "ا",  # ALEF WITH HAMZA BELOW -> ALEF        إ -> ا
    "ٱ": "ا",  # ALEF WASLA        -> ALEF
    "ؤ": "ؤ",  # WAW WITH HAMZA kept (distinct Urdu letter)
    "ھ": "ھ",  # DOACHASHMEE HEH kept (do-chashmi he is phonemic)
    "ء": "ء",  # HAMZA kept
}
# Arabic-Indic and extended Arabic-Indic digits -> ASCII digits.
for _i in range(10):
    CHAR_MAP[chr(0x0660 + _i)] = str(_i)   # ٠..٩
    CHAR_MAP[chr(0x06F0 + _i)] = str(_i)   # ۰..۹

# Invisible / formatting characters that survive a copy-paste from the web.
INVISIBLES = re.compile(r"[​‌‍‎‏⁠﻿­]")


def normalize_unicode(text: str) -> str:
    """NFC-compose, fold Arabic variants onto Urdu letters, strip invisibles."""
    text = unicodedata.normalize("NFC", text)
    text = INVISIBLES.sub("", text)
    return "".join(CHAR_MAP.get(ch, ch) for ch in text)


# --------------------------------------------------------------------------
# 2. DIACRITICS REMOVAL
# --------------------------------------------------------------------------
# Urdu is normally written without aerab. When they do appear they create
# duplicate surface forms of one word (عِلم vs علم), inflating data sparsity.
DIACRITICS = re.compile(
    "["
    "ً-ٟ"   # fathatan..wavy hamza below (zabar, zer, pesh, shadda...)
    "ٰ"          # superscript alef
    "ۖ-ۭ"   # Quranic annotation marks
    "ـ"          # tatweel / kashida (pure decoration)
    "]"
)


def remove_diacritics(text: str) -> str:
    """علم <- عِلمِ  : strip every vowel/annotation mark and the kashida."""
    return DIACRITICS.sub("", text)


# --------------------------------------------------------------------------
# 3. NOISE REMOVAL
# --------------------------------------------------------------------------
URL_RE = re.compile(r"(https?://\S+|www\.\S+|\S+\.(?:com|net|org|pk|co\.uk)\b\S*)")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
HTML_RE = re.compile(r"<[^>]+>|&[a-zA-Z#0-9]+;")
HASHTAG_RE = re.compile(r"[#@]\w+")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"   # pictographs, emoticons, symbols
    "\U00002600-\U000027BF"   # misc symbols + dingbats
    "\U0001F1E6-\U0001F1FF"   # regional indicators (flags)
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U00002190-\U000021FF"   # arrows
    "\U00002B00-\U00002BFF"
    "]+"
)
# Navigation / furniture strings BBC repeats on every page.
BOILERPLATE = [
    "،تصویر کا ذریعہ", "تصویر کا ذریعہ", "،تصویر کا کیپشن", "تصویر کا کیپشن",
    "ویڈیو کیپشن", "مواد پر جائیں", "اہم خبریں", "بی بی سی اردو",
    "بی بی سی نیوز", "بی بی سی", "سب سے زیادہ پڑھی جانے والی",
    "یہ بھی پڑھیے", "یہ بھی پڑھیں", "ویڈیو دیکھیے", "فوٹو کیپشن",
    "اپ ڈیٹ کی گئی", "شائع ہوا", "لائیو", "سوشل میڈیا",
]


def remove_noise(text: str) -> str:
    """Drop URLs, emails, emoji, HTML residue, hashtags and BBC page furniture."""
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    for phrase in BOILERPLATE:
        text = text.replace(phrase, " ")
    return text


# --------------------------------------------------------------------------
# 4. NON-URDU TEXT REMOVAL
# --------------------------------------------------------------------------
# Anything outside the Urdu block, ASCII digits and the punctuation we keep is
# discarded. This removes embedded English and Roman Urdu in one rule, because
# both are written in Latin script.
KEEP_PUNCT = "۔؟!،؛:()\"'“”‘’«»-—%"
URDU_RANGE = "؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿"
NON_URDU_RE = re.compile(rf"[^{URDU_RANGE}0-9\s{re.escape(KEEP_PUNCT)}]+")
LATIN_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z'’\-]*\b")


def remove_non_urdu(text: str) -> str:
    """Strip Latin-script words (English + Roman Urdu) and any other script."""
    text = LATIN_WORD_RE.sub(" ", text)
    return NON_URDU_RE.sub(" ", text)


# --------------------------------------------------------------------------
# 5. WHITESPACE AND FORMATTING NORMALIZATION
# --------------------------------------------------------------------------
def normalize_whitespace(text: str) -> str:
    """Collapse runs of space/newline and tidy spacing around punctuation."""
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = re.sub(r"\s+([۔؟!،؛:])", r"\1", text)        # no space before punct
    text = re.sub(r"([۔؟!،؛:])(?=\S)", r"\1 ", text)    # one space after punct
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text.strip()


# --------------------------------------------------------------------------
# 6. SENTENCE SEGMENTATION
# --------------------------------------------------------------------------
# Urdu sentences end with the danda (۔), the Urdu question mark (؟) or "!".
SENT_END = "۔؟!"
SENT_SPLIT_RE = re.compile(rf"(?<=[{SENT_END}])\s*")


# A closing quote that follows the danda belongs to the sentence that just
# ended ("...‘۔ ‘" ), not to the next one. Without this, every quoted paragraph
# starts a bogus sentence consisting of a lone quote mark.
_CLOSERS = "’‘”“»)"


def segment_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences, keeping the terminator attached."""
    parts = [s.strip() for s in SENT_SPLIT_RE.split(text.replace("\n", " "))]
    merged: list[str] = []
    for part in parts:
        if not part:
            continue
        lead = ""
        while part and part[0] in _CLOSERS:      # pull stray closers backwards
            lead += part[0]
            part = part[1:].lstrip()
        if lead and merged:
            merged[-1] += lead
        elif lead:
            merged.append(lead)
        if part:
            merged.append(part)
    return [s for s in merged if any(ch.isalnum() for ch in s)]


# --------------------------------------------------------------------------
# 7. CUSTOM URDU TOKENIZER
# --------------------------------------------------------------------------
NUM_TOKEN = "<NUM>"
PUNCT_CHARS = set("۔؟!،؛:()\"'“”‘’«»-—")

# Urdu postpositions (حروفِ جار). Normally space separated, but BBC copy often
# fuses them onto the preceding noun ("پاکستانمیں"), which would create a new
# vocabulary item for every noun. We split those back off.
POSTPOSITIONS = {
    "کا", "کی", "کے", "کو", "نے", "سے", "میں", "پر", "تک", "پہ",
    "والا", "والی", "والے", "کیلیے", "لیے", "لیئے", "بعد", "پاس", "ساتھ",
}
# Fused forms we split explicitly: written-together compounds that are two words.
FUSED = {
    "کیلیے": ("کے", "لیے"), "کیلئے": ("کے", "لیے"), "کےلیے": ("کے", "لیے"),
    "کےلئے": ("کے", "لیے"), "اسلیے": ("اس", "لیے"), "اسلئے": ("اس", "لیے"),
    "جسکے": ("جس", "کے"), "جسکی": ("جس", "کی"), "جسکا": ("جس", "کا"),
    "اسکے": ("اس", "کے"), "اسکی": ("اس", "کی"), "اسکا": ("اس", "کا"),
    "انکے": ("ان", "کے"), "انکی": ("ان", "کی"), "انکا": ("ان", "کا"),
    "جنکے": ("جن", "کے"), "کیساتھ": ("کے", "ساتھ"), "اسمیں": ("اس", "میں"),
    "انمیں": ("ان", "میں"), "جسمیں": ("جس", "میں"),
}
_NUM_RE = re.compile(r"^[0-9]+([.,٫][0-9]+)*$")
# Digits are often fused to Urdu letters in BBC copy (ٹی20, 19و, کہ9). A whole-token
# test would leave those digits in the corpus, so digit runs are extracted from
# inside the token as well.
_DIGIT_RUN_RE = re.compile(r"([0-9]+(?:[.,٫][0-9]+)*)")
_HAS_DIGIT_RE = re.compile(r"[0-9]")

# Only these postpositions are detached by the generic rule, and only when the
# remaining stem is >= _MIN_SPLIT_STEM characters.
#
# "نے" is deliberately NOT in this list. It is the ergative marker, but it is
# also the ending of every Urdu infinitive in the oblique case (بولنے، کرنے،
# ہونے، جانے). Splitting on it produced بول + نے for thousands of verbs, so the
# generic rule is restricted to postpositions that do not double as a verb
# ending. Genuinely fused forms are handled by the hand-checked FUSED table.
_SPLITTABLE = sorted(["میں", "سے", "کو", "پر", "تک", "کے", "کی", "کا"],
                     key=len, reverse=True)
_MIN_SPLIT_STEM = 4


def _split_fused_postposition(word: str) -> list[str]:
    """پاکستانمیں -> ['پاکستان', 'میں'] (only when the stem stays >= 3 chars)."""
    if word in FUSED:
        return list(FUSED[word])
    if word in POSTPOSITIONS:
        return [word]
    for post in _SPLITTABLE:
        if word.endswith(post) and len(word) - len(post) >= _MIN_SPLIT_STEM:
            return [word[: -len(post)], post]
    return [word]


def tokenize(text: str) -> list[str]:
    """
    Word-boundary tokenizer for Urdu.

    - separates punctuation into its own tokens
    - replaces every number with <NUM>
    - detaches fused postpositions
    """
    tokens: list[str] = []
    for chunk in text.split():
        buf = ""
        for ch in chunk:
            if ch in PUNCT_CHARS:
                if buf:
                    tokens.extend(_emit(buf))
                    buf = ""
                tokens.append(ch)
            else:
                buf += ch
        if buf:
            tokens.extend(_emit(buf))
    return tokens


def _emit(word: str) -> list[str]:
    if _NUM_RE.match(word):
        return [NUM_TOKEN]
    if _HAS_DIGIT_RE.search(word):
        # split the token into digit runs and letter runs, e.g. ٹی20 -> ٹی + <NUM>
        out: list[str] = []
        for piece in _DIGIT_RUN_RE.split(word):
            if not piece:
                continue
            if _HAS_DIGIT_RE.match(piece):
                if not out or out[-1] != NUM_TOKEN:
                    out.append(NUM_TOKEN)
            else:
                out.extend(_split_fused_postposition(piece))
        return out or [NUM_TOKEN]
    return _split_fused_postposition(word)


# --------------------------------------------------------------------------
# 8. CUSTOM URDU LEMMATIZER  (plural + gender normalization)
# --------------------------------------------------------------------------
# Dictionary first (irregular + Arabic broken plurals), rules second.
LEMMA_DICT = {
    # irregular / broken plurals
    "اساتذہ": "استاد", "علما": "عالم", "علماء": "عالم", "طلبہ": "طالبعلم",
    "طلباء": "طالبعلم", "امرا": "امیر", "وزرا": "وزیر", "وزراء": "وزیر",
    "سفرا": "سفیر", "شعرا": "شاعر", "ادبا": "ادیب", "قواعد": "قاعدہ",
    "حالات": "حالت", "واقعات": "واقعہ", "مقامات": "مقام", "معلومات": "معلومات",
    "اقدامات": "اقدام", "انتخابات": "انتخاب", "مذاکرات": "مذاکرہ",
    "احتجاجات": "احتجاج", "خدمات": "خدمت", "ضروریات": "ضرورت",
    "اخبارات": "اخبار", "مکانات": "مکان", "جذبات": "جذبہ", "خیالات": "خیال",
    # Arabic sound plurals whose singular ends in ـت (the "ات" rule alone
    # under-generates these, so they are listed explicitly)
    "ہدایات": "ہدایت", "سہولیات": "سہولت", "شکایات": "شکایت",
    "روایات": "روایت", "کیفیات": "کیفیت", "اطلاعات": "اطلاع",
    "تفصیلات": "تفصیل", "مشکلات": "مشکل", "امکانات": "امکان",
    "انکشافات": "انکشاف", "مطالبات": "مطالبہ", "معاملات": "معاملہ",
    "اعلانات": "اعلان", "الزامات": "الزام", "تعلقات": "تعلق",
    "کوششات": "کوشش", "سفارشات": "سفارش", "رجحانات": "رجحان",
    "الفاظ": "لفظ", "کتب": "کتاب", "اولاد": "اولاد", "عوام": "عوام",
    # gender pairs -> masculine singular base
    "لڑکے": "لڑکا", "لڑکوں": "لڑکا", "لڑکیاں": "لڑکی", "لڑکیوں": "لڑکی",
    "بچے": "بچہ", "بچوں": "بچہ", "بچیاں": "بچی", "عورتیں": "عورت",
    "عورتوں": "عورت", "مرد": "مرد", "مردوں": "مرد", "آدمی": "آدمی",
    "لوگوں": "لوگ", "بہنیں": "بہن", "بھائیوں": "بھائی", "ماؤں": "ماں",
    "باپوں": "باپ", "بیٹے": "بیٹا", "بیٹوں": "بیٹا", "بیٹیاں": "بیٹی",
    "کتابیں": "کتاب", "کتابوں": "کتاب", "سڑکیں": "سڑک", "سڑکوں": "سڑک",
    "شہروں": "شہر", "ملکوں": "ملک", "دنوں": "دن", "راتوں": "رات",
    "سالوں": "سال", "لاکھوں": "لاکھ", "ہزاروں": "ہزار", "کروڑوں": "کروڑ",
    # auxiliaries -> canonical form (gender/number neutral)
    "ہوئے": "ہوا", "ہوئی": "ہوا", "ہوئیں": "ہوا", "ہوتے": "ہوتا",
    "ہوتی": "ہوتا", "ہوتیں": "ہوتا", "گئے": "گیا", "گئی": "گیا",
    "گئیں": "گیا", "کیے": "کیا", "کیں": "کیا", "کرتے": "کرتا",
    "کرتی": "کرتا", "کرتیں": "کرتا", "رہے": "رہا", "رہی": "رہا",
    "رہیں": "رہا", "دیے": "دیا", "دی": "دیا", "لی": "لیا", "لیے": "لیا",
}

# Function words that must never be lemmatized or stemmed: mangling them
# destroys the grammatical skeleton the language model needs for fluency.
PROTECTED = {
    "کا", "کی", "کے", "کو", "نے", "سے", "میں", "پر", "تک", "پہ", "ہی",
    "بھی", "تو", "نہ", "نہیں", "اور", "یا", "کہ", "جو", "جب", "تب",
    "اگر", "مگر", "لیکن", "پھر", "ابھی", "اب", "یہ", "وہ", "اس", "ان",
    "میرا", "تمھارا", "ہمارا", "آپ", "ہم", "تم", "میں", "ہے", "ہیں",
    "تھا", "تھی", "تھے", "تھیں", "ہو", "ہوں", "گا", "گی", "گے",
    "ایک", "دو", "کچھ", "سب", "ہر", "کوئی", "کیا", "کیوں", "کیسے",
    "کہاں", "کون", "جس", "جن", "وہاں", "یہاں", "بعد", "قبل", "دوران",
    "ساتھ", "لیے", "بارے", "طرف", "خلاف", "مطابق", "ذریعے", "باعث",
    "<NUM>", "<s>", "</s>", "<unk>",
}

_PLURAL_RULES = [
    ("یوں", "ی"),   # لڑکیوں -> لڑکی
    ("ئیں", "ی"),   # ??ئیں  -> ??ی
    ("یاں", "ی"),   # لڑکیاں -> لڑکی
    ("وں", ""),     # لوگوں  -> لوگ
    ("ات", ""),     # اقدامات-> اقدام (Arabic sound plural)
    ("یں", ""),     # کتابیں -> کتاب
]


def lemmatize(token: str) -> str:
    """Map plural / gender-inflected forms onto a single dictionary form."""
    if token in PROTECTED or token.startswith("<"):
        return token
    if token in LEMMA_DICT:
        return LEMMA_DICT[token]
    for suffix, repl in _PLURAL_RULES:
        if token.endswith(suffix) and len(token) - len(suffix) + len(repl) >= 3:
            return token[: -len(suffix)] + repl
    return token


# --------------------------------------------------------------------------
# 9. CUSTOM URDU STEMMER  (suffix stripping to a root form)
# --------------------------------------------------------------------------
# Applied after lemmatization, longest suffix first, one pass, and only while
# the surviving root stays >= MIN_STEM_LEN characters.
MIN_STEM_LEN = 3
# NOTE: "ان" and "کر" are deliberately excluded. They are real Urdu suffixes but
# they over-trigger on proper nouns (پاکستان -> پاکست, لشکر -> لش), which is the
# classic over-stemming error. Excluding them trades a little recall for roots
# that stay recognisable to the language model.
SUFFIXES = sorted(
    [
        "یوں", "یاں", "ئیں", "اؤں", "وں", "یں", "یا", "یے", "ئی",
        "ات", "گی", "گا", "گے", "نا", "نی", "نے", "تا", "تی", "تے",
        "ہٹ", "پن", "مند", "دار", "ی", "ے", "ا", "ں",
    ],
    key=len,
    reverse=True,
)


def stem(token: str) -> str:
    """Strip one common inflectional suffix to approximate the root."""
    if token in PROTECTED or token.startswith("<") or len(token) <= MIN_STEM_LEN:
        return token
    for suffix in SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= MIN_STEM_LEN:
            return token[: -len(suffix)]
    return token


# --------------------------------------------------------------------------
# 10. FULL PIPELINE
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Stages 1-5: everything up to (but not including) segmentation."""
    text = normalize_unicode(text)
    text = remove_diacritics(text)
    text = remove_noise(text)
    text = remove_non_urdu(text)
    return normalize_whitespace(text)


def preprocess(text: str, apply_lemma: bool = True, apply_stem: bool = True) -> list[list[str]]:
    """
    Run the whole Part-1 pipeline.

    Returns a list of sentences, each a list of tokens - the exact shape the
    Part-2 language models consume.
    """
    sentences = []
    for sent in segment_sentences(clean_text(text)):
        tokens = tokenize(sent)
        if apply_lemma and apply_stem:
            # A token the lemmatizer already rewrote is a dictionary form; stemming
            # it again double-reduces (ہدایات -> ہدای -> ہدا). Stem only what the
            # lemmatizer left untouched.
            tokens = [lemma if (lemma := lemmatize(t)) != t else stem(t) for t in tokens]
        elif apply_lemma:
            tokens = [lemmatize(t) for t in tokens]
        elif apply_stem:
            tokens = [stem(t) for t in tokens]
        tokens = [t for t in tokens if t.strip()]
        if len(tokens) >= 2:
            sentences.append(tokens)
    return sentences


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------
def _demo() -> None:
    assert remove_diacritics("عِلمِ") == "علم"
    assert normalize_unicode("كتاب ي ه") == "کتاب ی ہ"
    assert "بی" not in remove_noise("نیوز بریکنگ Visit www.bbc.com 😊")
    assert remove_non_urdu("یہ رپورٹ breaking news کے بارے میں ہے").split() == \
        "یہ رپورٹ کے بارے میں ہے".split()
    assert normalize_whitespace("پاکستان   میں  بارش   ہوئی") == "پاکستان میں بارش ہوئی"
    assert segment_sentences("پاکستان میں بارش ہوئی۔ لوگ متاثر ہوئے۔") == \
        ["پاکستان میں بارش ہوئی۔", "لوگ متاثر ہوئے۔"]
    # a stray closing quote must not become its own sentence
    assert segment_sentences("وہ بولے ’بارش ہوئی۔‘ لوگ گئے۔") == \
        ["وہ بولے ’بارش ہوئی۔‘", "لوگ گئے۔"]
    assert tokenize("سنہ 2024 میں۔") == ["سنہ", "<NUM>", "میں", "۔"]
    assert tokenize("ٹی20") == ["ٹی", "<NUM>"]        # digits fused to letters
    assert tokenize("19و") == ["<NUM>", "و"]
    assert not any(ch.isdigit() for t in tokenize("کہ9 ٹی20 1990") for ch in t)
    assert tokenize("کیلیے") == ["کے", "لیے"]
    for infinitive in ("بولنے", "کرنے", "ہونے", "جانے", "دینے"):
        assert tokenize(infinitive) == [infinitive], infinitive   # never split on نے
    assert tokenize("پاکستانمیں") == ["پاکستان", "میں"]           # real fusion still splits
    assert lemmatize("لڑکیاں") == "لڑکی" and lemmatize("اساتذہ") == "استاد"
    assert lemmatize("میں") == "میں"                 # protected function word
    assert stem("کتابوں") == "کتاب" and stem("ہے") == "ہے"
    assert stem("پاکستان") == "پاکستان"          # proper noun must survive
    assert stem("لشکر") == "لشکر"
    sents = preprocess("پاکستان میں شدید بارش ہوئی۔ متاثر ہونے والے لوگ 500 تھے۔")
    assert len(sents) == 2 and "<NUM>" in sents[1]
    # lemmatizer output must not be stemmed a second time
    assert preprocess("ہدایات جاری", apply_stem=True)[0][0] == "ہدایت"
    print("preprocess.py: all self-checks passed")
    print("  sample ->", sents)


if __name__ == "__main__":
    _demo()
