"""English glosses for Urdu tokens shown in result tables (cleaned.txt forms).

A1-stemmed forms are glossed with the word they come from, e.g.
"روپ" <- روپے (rupees), "امر" <- امریکی (American; A1 split امری + کی).
Only words that were checked by hand are listed; others show no gloss.
"""

GLOSS = {
    # politics / government
    "حکومت": "government", "وزیر": "minister", "صدر": "president", "ٹرمپ": "Trump",
    "مود": "Modi (<-مودی)", "صحافت": "journalism", "صحافی": "journalist", "میڈ": "media (<-میڈیا)",
    "خان": "Khan", "وائٹ": "White (House)", "علیمہ": "Aleema", "عدالت": "court", "جج": "judge",
    "کورٹ": "court (Eng.)", "مقدم": "case (<-مقدمے)", "سماعت": "hearing", "درخواست": "petition",
    "اپیل": "appeal", "ریمانڈ": "remand", "انصاف": "justice", "وفاق": "federation",
    "مطالبہ": "demand", "پیٹرولیم": "petroleum", "ہارپ": "Harp (name)", "سیاست": "politics",
    "انتخاب": "election", "پارٹ": "party (<-پارٹی)", "نورین": "Noreen", "تحریک": "movement",
    # sport
    "کرکٹ": "cricket", "کھیل": "sport / game", "ٹیم": "team", "ٹیسٹ": "Test (match)",
    "میچ": "match", "رنز": "runs", "کھلاڑی": "player", "انگلینڈ": "England", "وکٹ": "wicket",
    "کپتان": "captain", "بولر": "bowler", "بابر": "Babar", "سکواڈ": "squad", "ہاک": "hockey (<-ہاکی)",
    "لارڈز": "Lord's", "کوچ": "coach", "بورڈ": "board", "لیگ": "league",
    # economy
    "تیل": "oil", "قیمت": "price", "پیٹرول": "petrol", "روپ": "rupees (<-روپے)", "لیٹر": "litre",
    "بیرل": "barrel", "ڈیزل": "diesel", "آبنائ": "strait (<-آبنائے)", "خام": "crude",
    "بندرگاہ": "port", "ڈالر": "dollar", "معیشت": "economy", "مہنگا": "expensive / inflation (<-مہنگائی)",
    "زر": "money (افراطِ زر = inflation)", "قلت": "shortage", "گیس": "gas", "کمپن": "company (<-کمپنی)",
    # international / conflict
    "ایران": "Iran", "امر": "American (<-امریکی)", "سعود": "Saudi (<-سعودی)", "عرب": "Arab(ia)",
    "میزائل": "missile", "معاہد": "agreement (<-معاہدے)", "روس": "Russia", "فوج": "army",
    "جہاز": "ship / aircraft", "خالد": "Khalid", "اسرائیل": "Israel", "جنگ": "war",
    "حمل": "attack (<-حملے) / pregnancy", "افواج": "armed forces", "کمانڈر": "commander",
    "جنرل": "general", "ڈرون": "drone", "یوکرین": "Ukraine", "غزہ": "Gaza", "تہران": "Tehran",
    "حوث": "Houthi (<-حوثی)", "ہرمز": "Hormuz", "امریکہ": "America", "پاکستان": "Pakistan",
    "انڈ": "India (<-انڈیا)", "انڈین": "Indian", "پاکستا": "Pakistani (<-پاکستانی)",
    "ایرا": "Iranian (<-ایرانی)", "چین": "China", "طالبان": "Taliban", "القاعدہ": "al-Qaeda",
    "لادن": "(bin) Laden", "شیخ": "Sheikh", "عسکر": "military (<-عسکری)",
    # health & society
    "بچہ": "child", "پولیس": "police", "خواتین": "women", "ملزم": "accused", "ریپ": "rape",
    "خاتون": "woman", "فلم": "film", "بچی": "girl", "ڈاکٹر": "doctor", "ڈرام": "drama (<-ڈرامہ)",
    "صحت": "health", "تعلیم": "education", "ہسپتال": "hospital", "علاج": "treatment",
    "سکول": "school", "استاد": "teacher", "طالبعلم": "student", "اداکار": "actor",
    "یاب": "-recovered (صحت یاب)", "مند": "-ful (صحت مند = healthy)", "ذہن": "mind",
    "آباد": "-abad / populated (<-آبادی)", "اسلام": "Islam(-abad)", "دارالحکومت": "capital city",
    "لاہور": "Lahore", "کابل": "Kabul", "شوہر": "husband", "بیو": "wife (<-بیوی)",
    "عورت": "woman", "مرد": "man", "بہن": "sister", "والدہ": "mother", "والد": "father",
    "دہل": "Delhi (<-دہلی)", "ماسکو": "Moscow", "پوتن": "Putin", "نیتن": "Netanyahu (<-نیتن یاہو)",
    "شریف": "Sharif", "شہباز": "Shehbaz", "عمران": "Imran", "ڈونلڈ": "Donald", "نریندر": "Narendra",
    "افغانستان": "Afghanistan", "حماس": "Hamas", "بھا": "brother (<-بھائی)", "سہیل": "Sohail (name)",
    "کے": "of (genitive)", "ولادیمیر": "Vladimir", "گاندھ": "Gandhi (<-گاندھی)",
    "پمز": "PIMS (hospital)", "وارڈ": "ward",
}


def gloss(tok: str) -> str:
    return GLOSS.get(tok, "")


def with_gloss(tok: str) -> str:
    g = GLOSS.get(tok)
    return f"{tok} ({g})" if g else tok
