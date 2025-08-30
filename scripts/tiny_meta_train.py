# tiny_meta_train.py  — fast meta-trainer over BERT probs + handcrafted features
import os, json, math
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizerFast
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib



torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


MODEL_DIR = os.getenv("MODEL_DIR", "models/mbert-fake-news-bf16/best")
DATA_DIR  = os.getenv("DATA_DIR",  "data/processed")
OUT_DIR   = os.getenv("OUT_DIR",   "models/hybrid_meta")
os.makedirs(OUT_DIR, exist_ok=True)



BASE_MODEL_NAME = "distilbert-base-multilingual-cased"
TOKENIZER_FILE = os.path.join(MODEL_DIR, "tokenizer.json")
VOCAB_FILE     = os.path.join(MODEL_DIR, "vocab.txt")

def load_tokenizer():

    try:
        tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
        print("Loaded tokenizer:", type(tok).__name__, "from", MODEL_DIR)
        return tok
    except Exception as e:
        print("AutoTokenizer(use_fast=True) failed:", e)


    try:
        tok = BertTokenizerFast.from_pretrained(MODEL_DIR)
        print("Loaded BertTokenizerFast from", MODEL_DIR)
        return tok
    except Exception as e:
        print("BertTokenizerFast.from_pretrained failed:", e)


    try:
        tok = BertTokenizerFast(vocab_file=VOCAB_FILE, tokenizer_file=TOKENIZER_FILE)
        print("Loaded BertTokenizerFast via files")
        return tok
    except Exception as e:
        print("Direct files load failed:", e)


    tok = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    print("Loaded tokenizer from base model name:", BASE_MODEL_NAME)
    return tok

tok = load_tokenizer()




DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE_BERT   = int(os.getenv("BATCH_SIZE_BERT", "128"))
MAX_LENGTH        = int(os.getenv("MAX_LENGTH", "256"))
USE_BF16          = int(os.getenv("USE_BF16", "1"))
USE_FP16          = int(os.getenv("USE_FP16", "0"))


from features_hybrid import extract_features


FEATURE_ORDER = [

    "len_chars","n_tokens","n_sents","avg_sent_len","avg_word_len","type_token_ratio",

    "exclamations","question_marks","multi_punct","ellipses","quote_marks",
    "emojis","all_caps_words","headline_style_ratio",

    "repeated_words_ratio","repeated_chars_flag",

    "num_count","pct_count","pct_over_100","num_diversity","num_density","absurd_big_number",

    "urls_count","domains_count","reputable_source_present","unknown_domain_ratio",
    "hashtags_count","mentions_count",
    "relative_time_markers","absolute_time_markers","hedging_markers","strong_claim_markers",

    "clickbait_match",

    "code_switch_ratio","subword_oov_ratio","lang_is_ar",
]

def vec_from_features(d: dict):
    return np.array([float(d.get(k, 0.0)) for k in FEATURE_ORDER], dtype=np.float32)


def batch_iter(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs]

def compute_bert_probs(texts, tok, model, cache_path=None):

    if cache_path and os.path.isfile(cache_path):
        print(f"[cache] loading BERT probs from {cache_path}")
        return np.load(cache_path)

    print(f"[BERT] running inference on {len(texts):,} samples (bs={BATCH_SIZE_BERT}, max_len={MAX_LENGTH})")
    model.eval().to(DEVICE)


    original_dtype = None
    if torch.cuda.is_available():
        if USE_BF16 and torch.cuda.get_device_properties(0).major >= 8:  # Ampere+
            original_dtype = next(model.parameters()).dtype
            model.to(torch.bfloat16)
            print("[BERT] using bfloat16")
        elif USE_FP16:
            original_dtype = next(model.parameters()).dtype
            model.half()
            print("[BERT] using float16")

    probs_list = []
    with torch.inference_mode():
        for batch in batch_iter(texts, BATCH_SIZE_BERT):
            enc = tok(batch, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            logits = model(**enc).logits
            if logits.dtype != torch.float32:
                logits = logits.float()
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs_list.append(p)

    probs = np.concatenate(probs_list, axis=0)
    if cache_path:
        np.save(cache_path, probs)
        print(f"[cache] saved BERT probs → {cache_path}")

    if original_dtype is not None:
        model.to(original_dtype)

    return probs

def compute_features_matrix(texts, tok, cache_path=None, chunk=5000):

    if cache_path and os.path.isfile(cache_path):
        print(f"[cache] loading features matrix from {cache_path}")
        return np.load(cache_path)

    print(f"[FEATS] extracting features for {len(texts):,} samples (chunk={chunk})")
    parts = []
    for sub in batch_iter(texts, chunk):
        rows = []
        for t in sub:
            d = extract_features(t, tok)
            rows.append(vec_from_features(d))
        parts.append(np.stack(rows))
    X = np.concatenate(parts, axis=0)

    if cache_path:
        np.save(cache_path, X)
        print(f"[cache] saved features → {cache_path}")
    return X

def main():

    raw = load_dataset("csv", data_files={
        "train":      f"{DATA_DIR}/ml_train.csv",
        "validation": f"{DATA_DIR}/ml_val.csv",
        "test":       f"{DATA_DIR}/ml_test.csv",
    })

    for split in raw.keys():
        raw[split] = raw[split].map(lambda ex: {"label": int(ex["label"])})
    tr_texts = list(raw["train"]["text"]) + list(raw["validation"]["text"])
    tr_y = np.array(list(raw["train"]["label"]) + list(raw["validation"]["label"]), dtype=np.int64)
    te_texts = list(raw["test"]["text"])
    te_y = np.array(list(raw["test"]["label"]), dtype=np.int64)

    print(f"train+val: {len(tr_texts):,}  |  test: {len(te_texts):,}")


    tok = load_tokenizer()

    bert = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)


    p_tr = compute_bert_probs(tr_texts, tok, bert, cache_path=os.path.join(OUT_DIR, "p_tr.npy"))[:, None]
    p_te = compute_bert_probs(te_texts, tok, bert, cache_path=os.path.join(OUT_DIR, "p_te.npy"))[:, None]


    Xtr_f = compute_features_matrix(tr_texts, tok, cache_path=os.path.join(OUT_DIR, "Xtr_f.npy"))
    Xte_f = compute_features_matrix(te_texts, tok, cache_path=os.path.join(OUT_DIR, "Xte_f.npy"))


    Xtr = np.concatenate([Xtr_f, p_tr], axis=1)
    Xte = np.concatenate([Xte_f, p_te], axis=1)
    feature_order_plus = FEATURE_ORDER + ["bert_prob_real"]


    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1))
    ])

    print("[META] fitting LogisticRegression on features + BERT prob…")
    pipe.fit(Xtr, tr_y)
    preds = pipe.predict(Xte)

    print("\n=== META REPORT (test) ===")
    print(classification_report(te_y, preds, target_names=["FAKE","REAL"], digits=4))


    joblib.dump({"pipe": pipe, "feature_order": feature_order_plus},
                os.path.join(OUT_DIR, "meta.joblib"))
    with open(os.path.join(OUT_DIR, "feature_order.json"), "w", encoding="utf-8") as f:
        json.dump(feature_order_plus, f, ensure_ascii=False, indent=2)
    print("Saved →", OUT_DIR)


    try:
        clf = pipe.named_steps["clf"]
        coefs = clf.coef_[0]
        names = feature_order_plus
        top_pos = sorted(zip(names, coefs), key=lambda x: x[1], reverse=True)[:12]
        top_neg = sorted(zip(names, coefs), key=lambda x: x[1])[:12]
        print("\nTop features → REAL:")
        for n,w in top_pos: print(f"{n:30s} {w:+.3f}")
        print("\nTop features → FAKE:")
        for n,w in top_neg: print(f"{n:30s} {w:+.3f}")
    except Exception as e:
        print("Feature importance print skipped:", e)

if __name__ == "__main__":
    main()

