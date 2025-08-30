

import re
from collections import Counter
from typing import Dict, Any


CLICK_AR = {
    "عاجل","خطير","صادم","لن تصدق","كارثة","فضيحة","حصري","مذهل","مفاجأة","بالفيديو","شاهد"
}
CLICK_EN = {
    "breaking","shocking","you won't believe","won’t believe","exclusive",
    "miracle","scandal","must see","viral"
}

HEDGE_AR  = {"يُقال","بحسب مصادر","مصادر مطلعة","تداول","زُعم","ربما","قد"}
HEDGE_EN  = {"reportedly","allegedly","sources say","it is said","rumor","maybe","might","could"}

STRONG_AR = {"مؤكد","بالتأكيد","حقيقة كاملة","%100","دون أدنى شك","لا جدال"}
STRONG_EN = {"definitely","undeniable","proven","100%","for sure","no doubt"}


REPUTABLE_DOMAINS = {
    "reuters.com","apnews.com","bbc.com","cnn.com","nytimes.com","aljazeera.com",
    "washingtonpost.com","theguardian.com","wsj.com","bloomberg.com","afp.com"
}

REL_TIME_AR = {"اليوم","أمس","الآن","توه","حالاً","قريباً"}
REL_TIME_EN = {"today","yesterday","now","soon","currently","just"}

AR_CHARS     = re.compile(r'[\u0600-\u06FF]')
LATIN        = re.compile(r'[A-Za-z]')
EMOJI        = re.compile(r'[\U0001F300-\U0001FAFF]')
URL_RE       = re.compile(r'https?://[^\s]+|www\.[^\s]+', re.IGNORECASE)
DOMAIN_RE    = re.compile(r'(?:https?://)?(?:www\.)?([^/\s]+)', re.IGNORECASE)
NUM_RE       = re.compile(r'(?<!\w)[+-]?\d+(?:[.,]\d+)?(?!\w)')
PCT_RE       = re.compile(r'(\d+(?:[.,]\d+)?)\s*%')
ELLI_RE      = re.compile(r'…|\.{3,}')
MULTI_PUNCT  = re.compile(r'!!+|؟؟+|\?\?+')
ABS_YEAR     = re.compile(r'\b(19|20)\d{2}\b')
REPEATED_CHR = re.compile(r'(.)\1\1+')


def _sent_split(t: str):

    return [s for s in re.split(r'(?<=[\.\!\؟\?])\s+', t or "") if s]

def _word_split(t: str):
    return [w for w in re.findall(r'\b\w+\b', (t or "").lower())]

def _contains_any(text: str, vocab) -> bool:
    if not text: return False
    return any(k in text for k in vocab)

def _is_arabic(t: str) -> bool:
    return bool(AR_CHARS.search(t or ""))

def _to_float(x) -> float:
    try: return float(str(x).replace(",", "."))
    except: return 0.0

def _is_absurd_number(s: str) -> bool:
    v = _to_float(s)

    return v >= 1e9


def extract_features(text: str, tok=None) -> Dict[str, Any]:

    t = (text or "").strip()
    words = _word_split(t)
    sents = _sent_split(t)

    n_tokens = len(words)
    n_sents  = len(sents)
    len_chars = len(t)

    avg_sent_len = (n_tokens / n_sents) if n_sents else float(n_tokens)
    avg_word_len = (sum(len(w) for w in words) / n_tokens) if n_tokens else 0.0
    type_token_ratio = (len(set(words)) / n_tokens) if n_tokens else 0.0


    exclamations   = t.count("!")
    question_marks = t.count("?") + t.count("؟")
    ellipses       = len(ELLI_RE.findall(t))
    multi_punct    = 1 if MULTI_PUNCT.search(t) else 0
    quote_marks    = t.count('"') + t.count("“") + t.count("”") + t.count("«") + t.count("»")
    emojis         = len(EMOJI.findall(t))
    all_caps_words = len(re.findall(r'\b[A-Z]{2,}\b', t))


    title_case_words = len(re.findall(r'\b[A-Z][a-z]+\b', t))
    headline_style_ratio = (min(title_case_words / max(1, n_tokens), 1.0) if not _is_arabic(t) else 0.0)


    repeated_words_ratio = 0.0
    if n_tokens:
        c = Counter(words)
        repeated_words_ratio = sum(1 for _, v in c.items() if v >= 3) / max(1, len(c))
    repeated_chars_flag = 1 if REPEATED_CHR.search(t) else 0

    nums = NUM_RE.findall(t)
    pct_vals = [_to_float(m) for m in PCT_RE.findall(t)]
    pct_over_100   = 1 if any(p > 100.0 for p in pct_vals) else 0
    num_diversity  = len(set(nums))
    num_density    = (len(nums) / max(1, n_tokens))
    absurd_big_number = 1 if any(_is_absurd_number(x) for x in nums) else 0

    num_count = len(nums)
    pct_count = len(pct_vals)


    urls = URL_RE.findall(t)
    domains_set = { m.group(1).lower() for u in urls if (m := DOMAIN_RE.search(u)) }
    urls_count    = len(urls)
    domains_count = len(domains_set)
    reputable_source_present = 1 if any(d in REPUTABLE_DOMAINS for d in domains_set) else 0
    if domains_set:
        unknowns = sum(1 for d in domains_set if d not in REPUTABLE_DOMAINS)
        unknown_domain_ratio = unknowns / len(domains_set)
    else:
        unknown_domain_ratio = 0.0

    hashtags_count = t.count("#")
    mentions_count = t.count("@")


    lang_ar = _is_arabic(t)
    relative_time_markers = 1 if _contains_any(t if lang_ar else t.lower(), REL_TIME_AR if lang_ar else REL_TIME_EN) else 0
    absolute_time_markers = 1 if ABS_YEAR.search(t) else 0
    hedging_markers       = 1 if (_contains_any(t, HEDGE_AR) or _contains_any(t.lower(), HEDGE_EN)) else 0
    strong_claim_markers  = 1 if (_contains_any(t, STRONG_AR) or _contains_any(t.lower(), STRONG_EN)) else 0


    clickbait_match = 1 if (_contains_any(t, CLICK_AR) or _contains_any(t.lower(), CLICK_EN)) else 0


    arabic_chars = len(AR_CHARS.findall(t))
    latin_chars  = len(LATIN.findall(t))
    code_switch_ratio = 0.0
    if arabic_chars and latin_chars:

        code_switch_ratio = min(latin_chars / max(1, arabic_chars + latin_chars), 1.0)

    if tok is not None and n_tokens:
        try:
            pieces = tok.tokenize(t)
            subword_oov_ratio = len(pieces) / max(1, n_tokens)
        except Exception:
            subword_oov_ratio = 0.0
    else:
        subword_oov_ratio = 0.0


    feats = {

        "len_chars": len_chars,
        "n_tokens": n_tokens,
        "n_sents": n_sents,
        "avg_sent_len": float(avg_sent_len),
        "avg_word_len": float(avg_word_len),
        "type_token_ratio": float(type_token_ratio),


        "exclamations": int(exclamations),
        "question_marks": int(question_marks),
        "multi_punct": int(multi_punct),
        "ellipses": int(ellipses),
        "quote_marks": int(quote_marks),
        "emojis": int(emojis),
        "all_caps_words": int(all_caps_words),
        "headline_style_ratio": float(headline_style_ratio),


        "repeated_words_ratio": float(repeated_words_ratio),
        "repeated_chars_flag": int(repeated_chars_flag),


        "num_count": int(num_count),
        "pct_count": int(pct_count),
        "pct_over_100": int(pct_over_100),
        "num_diversity": int(num_diversity),
        "num_density": float(num_density),
        "absurd_big_number": int(absurd_big_number),


        "urls_count": int(urls_count),
        "domains_count": int(domains_count),
        "reputable_source_present": int(reputable_source_present),
        "unknown_domain_ratio": float(unknown_domain_ratio),
        "hashtags_count": int(hashtags_count),
        "mentions_count": int(mentions_count),


        "relative_time_markers": int(relative_time_markers),
        "absolute_time_markers": int(absolute_time_markers),
        "hedging_markers": int(hedging_markers),
        "strong_claim_markers": int(strong_claim_markers),


        "clickbait_match": int(clickbait_match),


        "code_switch_ratio": float(code_switch_ratio),
        "subword_oov_ratio": float(subword_oov_ratio),
        "lang_is_ar": int(lang_ar),
    }

    return feats
