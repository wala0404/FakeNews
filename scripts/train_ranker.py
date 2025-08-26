"""
Train a simple ranker (Logistic Regression baseline; optional XGBoost) on MINDsub_train_100k.

Inputs
- data_dir: data/MINDsub_train_100k containing behaviors.tsv and news.tsv

Features per candidate impression
- retrieval/cosine similarity (session vector vs candidate) — approximated by dot with normalized vectors
- recency_days (days since published), fallback to 7 when missing
- source domain (parsed from url)
- category (from news.tsv)
- language (constant 'en' for MIND)
- popularity (click count aggregated from training behaviors)

Output
- models/ranker/ranker.pkl (sklearn LogisticRegression + DictVectorizer) or ranker.xgb

Notes
- Do NOT use MINDlarge_test here.
- Keep memory in check via --limit_sessions or --max_samples.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

# Ensure repo root on sys.path for `app.*` imports when running as a script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.ml.embedder import get_embedder  # type: ignore

try:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.pipeline import Pipeline
    import joblib
except Exception as e:  # pragma: no cover
    print("[train_ranker] scikit-learn is required. Please pip install scikit-learn joblib.")
    raise

try:
    import xgboost as xgb  # type: ignore  # optional
except Exception:
    xgb = None  # type: ignore


def parse_news(news_path: str) -> Dict[str, Dict[str, str]]:
    """Parse MIND news.tsv → {news_id: {title, category, url, ...}}"""
    news = {}
    with open(news_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            # MIND news.tsv columns (typical):
            # news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities
            if not row:
                continue
            nid = row[0]
            rec = {
                "category": row[1] if len(row) > 1 else "",
                "subcategory": row[2] if len(row) > 2 else "",
                "title": row[3] if len(row) > 3 else "",
                "abstract": row[4] if len(row) > 4 else "",
                "url": row[5] if len(row) > 5 else "",
            }
            news[nid] = rec
    return news


def parse_behaviors(behaviors_path: str, limit_sessions: int | None = None):
    """Yield (impr_time, history_ids, impressions) per session.

    impressions: list of tuples (news_id, label_int)
    """
    count = 0
    with open(behaviors_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            # columns: ImpressionID, UserID, Time, History, Impressions
            if not row or len(row) < 5:
                continue
            timestamp = row[2]
            # Parse time if possible
            try:
                impr_time = dt.datetime.strptime(timestamp, "%m/%d/%Y %I:%M:%S %p")
            except Exception:
                # If format varies, fallback to naive now
                impr_time = dt.datetime.utcnow()
            history = row[3].split() if row[3] else []
            impressions_raw = row[4].split()
            impressions = []
            for imp in impressions_raw:
                # format: Nxxxxx-1 or Nxxxxx-0
                if "-" in imp:
                    nid, lbl = imp.split("-")
                    try:
                        label = int(lbl)
                    except Exception:
                        label = 0
                else:
                    nid, label = imp, 0
                impressions.append((nid, label))
            yield impr_time, history, impressions
            count += 1
            if limit_sessions and count >= limit_sessions:
                break


def domain_from_url(url: str) -> str:
    try:
        from urllib.parse import urlparse

        netloc = urlparse(url).netloc
        return netloc.lower() or "unknown"
    except Exception:
        return "unknown"


def build_embeddings(embedder, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, embedder.dim), dtype=np.float32)
    vecs = embedder.encode(texts)
    # normalize
    vecs = vecs.astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return (vecs / norms).astype(np.float32)


def mean_pool(vectors: List[np.ndarray]) -> np.ndarray:
    if not vectors:
        return np.zeros((0,), dtype=np.float32)
    arr = np.stack(vectors, axis=0)
    v = arr.mean(axis=0).astype(np.float32)
    n = np.linalg.norm(v) + 1e-12
    return (v / n).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join("data", "MINDsub_train_100k"))
    ap.add_argument("--out_dir", default=os.path.join("models", "ranker"))
    ap.add_argument("--model", choices=["logreg", "xgb"], default="logreg")
    ap.add_argument("--limit_sessions", type=int, default=None, help="Limit number of sessions for a quick train")
    ap.add_argument("--max_samples", type=int, default=None, help="Cap total candidate samples")
    ap.add_argument("--skip_samples", type=int, default=0, help="Skip the first N candidate samples (for chunked training)")
    ap.add_argument("--resume_from", type=str, default=None, help="Path to existing model (xgb) to continue training")
    ap.add_argument("--num_boost_round", type=int, default=200, help="Additional boosting rounds when resuming XGBoost")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    news_path = os.path.join(args.data_dir, "news.tsv")
    behaviors_path = os.path.join(args.data_dir, "behaviors.tsv")
    if not (os.path.exists(news_path) and os.path.exists(behaviors_path)):
        raise FileNotFoundError(f"Expected news.tsv and behaviors.tsv in {args.data_dir}")

    print(f"[train] Loading news from {news_path}")
    news = parse_news(news_path)

    print(f"[train] Computing popularity from {behaviors_path}")
    popularity = Counter()
    # First pass to compute popularity per news_id
    for _, _, impressions in parse_behaviors(behaviors_path, limit_sessions=None):
        for nid, lbl in impressions:
            if lbl == 1:
                popularity[nid] += 1

    print("[train] Initializing embedder")
    embedder = get_embedder()
    cache_vec: Dict[str, np.ndarray] = {}

    def get_news_vec(nid: str) -> np.ndarray:
        if nid in cache_vec:
            return cache_vec[nid]
        rec = news.get(nid, {})
        text = rec.get("title") or rec.get("abstract") or nid
        vec = build_embeddings(embedder, [text])
        v = vec[0] if vec.shape[0] else np.zeros((embedder.dim,), dtype=np.float32)
        cache_vec[nid] = v
        return v

    print(f"[train] Building training samples from {behaviors_path}")
    X_dicts: List[dict] = []
    y: List[int] = []

    total = 0
    skipped = 0
    for impr_time, history, impressions in parse_behaviors(behaviors_path, limit_sessions=args.limit_sessions):
        # session vector from history
        hist_vecs = [get_news_vec(h) for h in history if h]
        session_vec = mean_pool(hist_vecs) if hist_vecs else np.zeros((embedder.dim,), dtype=np.float32)
        # candidate features
        for nid, label in impressions:
            nrec = news.get(nid, {})
            c_vec = get_news_vec(nid)
            cosine = float(np.dot(session_vec, c_vec))
            url = nrec.get("url", "")
            source = domain_from_url(url)
            category = nrec.get("category", "") or "unknown"
            lang = "en"
            # MIND doesn't have published_at; fallback constant recency
            recency_days = 7.0
            pop = float(popularity.get(nid, 0))
            feats = {
                "cosine": cosine,
                "recency_days": recency_days,
                "popularity": pop,
                "source": source,
                "category": category,
                "lang": lang,
            }
            # Skip if requested (for chunked training)
            if skipped < args.skip_samples:
                skipped += 1
            else:
                X_dicts.append(feats)
                y.append(int(label))
                total += 1
            if args.max_samples and total >= args.max_samples:
                break
        if args.max_samples and total >= args.max_samples:
            break

    built = len(y)
    print(f"[train] Built {built} samples (skipped={skipped})")

    # Vectorize features
    # Handle resume for XGBoost: reuse vectorizer
    vec = None
    if args.model == "xgb" and args.resume_from:
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f"--resume_from not found: {args.resume_from}")
        prev = joblib.load(args.resume_from)
        vec = prev.get("vectorizer")
        if vec is None:
            raise RuntimeError("Resume model missing vectorizer")
        X = vec.transform(X_dicts)
    else:
        vec = DictVectorizer(sparse=True)
        X = vec.fit_transform(X_dicts)

    model_obj = None
    model_path = None
    if args.model == "logreg":
        print("[train] Training LogisticRegression")
        # Balanced to be robust to class imbalance
        clf = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced", random_state=args.seed)
        clf.fit(X, np.array(y))
        model_obj = {"model_type": "logreg", "model": clf, "vectorizer": vec}
        model_path = os.path.join(args.out_dir, "ranker.pkl")
        joblib.dump(model_obj, model_path)
    else:  # xgb
        if xgb is None:
            raise RuntimeError("xgboost not installed. pip install xgboost or use --model logreg")
        dtrain = xgb.DMatrix(X, label=np.array(y))
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": args.seed,
        }
        if args.resume_from:
            print("[train] Resuming XGBoost training")
            prev = joblib.load(args.resume_from)
            bst_prev = prev.get("model")
            if bst_prev is None:
                raise RuntimeError("Resume model missing booster")
            bst = xgb.train(params, dtrain, num_boost_round=args.num_boost_round, xgb_model=bst_prev)
        else:
            print("[train] Training XGBoost")
            bst = xgb.train(params, dtrain, num_boost_round=args.num_boost_round)
        model_obj = {"model_type": "xgb", "model": bst, "vectorizer": vec}
        model_path = os.path.join(args.out_dir, "ranker.xgb")
        joblib.dump(model_obj, model_path)

    print(f"[train] Saved model to {model_path}")


if __name__ == "__main__":
    main()
