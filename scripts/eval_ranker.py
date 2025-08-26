"""
Evaluate a trained ranker on MINDlarge_dev (validation).

Inputs
- data_dir: data/MINDlarge_dev (behaviors.tsv, news.tsv)
- model_dir: models/ranker/ (ranker.pkl or ranker.xgb)

Metrics
- AUC, LogLoss (over all impressions)
- Precision@K (default 10) aggregated over impressions
- NDCG@10

Notes
- Do NOT use MINDlarge_test.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.ml.embedder import get_embedder  # type: ignore

from sklearn.metrics import roc_auc_score, log_loss
import joblib

try:
    import xgboost as xgb  # type: ignore  # optional
except Exception:
    xgb = None  # type: ignore


def parse_news(news_path: str) -> Dict[str, Dict[str, str]]:
    news = {}
    with open(news_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
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


def parse_behaviors(behaviors_path: str):
    with open(behaviors_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or len(row) < 5:
                continue
            history = row[3].split() if row[3] else []
            impressions_raw = row[4].split()
            impressions = []
            for imp in impressions_raw:
                if "-" in imp:
                    nid, lbl = imp.split("-")
                    label = int(lbl) if lbl.isdigit() else 0
                else:
                    nid, label = imp, 0
                impressions.append((nid, label))
            yield history, impressions


def domain_from_url(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return (urlparse(url).netloc or "unknown").lower()
    except Exception:
        return "unknown"


def build_embeddings(embedder, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, embedder.dim), dtype=np.float32)
    vecs = embedder.encode(texts)
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


def precision_at_k(labels: List[int], scores: List[float], k: int) -> float:
    idx = np.argsort(scores)[::-1][:k]
    return float(np.sum(np.array(labels)[idx] == 1)) / max(1, k)


def ndcg_at_k(labels: List[int], scores: List[float], k: int) -> float:
    order = np.argsort(scores)[::-1]
    labels = np.array(labels)
    gains = (2 ** labels[order] - 1)[:k]
    discounts = 1.0 / np.log2(np.arange(2, 2 + len(gains)))
    dcg = float(np.sum(gains * discounts))
    # Ideal DCG
    ideal_order = np.argsort(labels)[::-1]
    ideal_gains = (2 ** labels[ideal_order] - 1)[:k]
    ideal_dcg = float(np.sum(ideal_gains * discounts[: len(ideal_gains)]))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join("data", "MINDlarge_dev"))
    ap.add_argument("--model_dir", default=os.path.join("models", "ranker"))
    ap.add_argument("--prefer", choices=["logreg", "xgb"], required=True, help="Which model to load: logreg or xgb")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    news_path = os.path.join(args.data_dir, "news.tsv")
    behaviors_path = os.path.join(args.data_dir, "behaviors.tsv")
    model_pkl = os.path.join(args.model_dir, "ranker.pkl")
    model_xgb = os.path.join(args.model_dir, "ranker.xgb")

    if not (os.path.exists(news_path) and os.path.exists(behaviors_path)):
        raise FileNotFoundError(f"Expected news.tsv and behaviors.tsv in {args.data_dir}")
    # Require the chosen model to exist
    if args.prefer == "logreg" and not os.path.exists(model_pkl):
        raise FileNotFoundError(f"LogReg model not found: {model_pkl}")
    if args.prefer == "xgb" and not os.path.exists(model_xgb):
        raise FileNotFoundError(f"XGBoost model not found: {model_xgb}")

    news = parse_news(news_path)
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

    # Load model
    model_obj = None
    model_path = model_pkl if args.prefer == "logreg" else model_xgb
    model_obj = joblib.load(model_path)
    model_type = model_obj.get("model_type")
    vectorizer = model_obj.get("vectorizer")
    model = model_obj.get("model")

    y_true_all: List[int] = []
    y_prob_all: List[float] = []

    p_at_k_list: List[float] = []
    ndcg_list: List[float] = []

    for history, impressions in parse_behaviors(behaviors_path):
        hist_vecs = [get_news_vec(h) for h in history if h]
        session_vec = mean_pool(hist_vecs) if hist_vecs else np.zeros((embedder.dim,), dtype=np.float32)

        X_dicts = []
        labels = []
        for nid, label in impressions:
            nrec = news.get(nid, {})
            c_vec = get_news_vec(nid)
            cosine = float(np.dot(session_vec, c_vec))
            source = domain_from_url(nrec.get("url", ""))
            category = nrec.get("category", "") or "unknown"
            feats = {
                "cosine": cosine,
                "recency_days": 7.0,
                "popularity": 0.0,  # unknown for dev; set 0
                "source": source,
                "category": category,
                "lang": "en",
            }
            X_dicts.append(feats)
            labels.append(int(label))

        if not X_dicts:
            continue
        X = vectorizer.transform(X_dicts)

        if model_type == "xgb":
            if xgb is None:
                raise RuntimeError("xgboost not installed but model is xgb")
            dmat = xgb.DMatrix(X)
            y_prob = model.predict(dmat)
        else:
            y_prob = model.predict_proba(X)[:, 1]

        y_true_all.extend(labels)
        y_prob_all.extend(y_prob.tolist())

        p_at_k = precision_at_k(labels, y_prob.tolist(), k=args.k)
        ndcg = ndcg_at_k(labels, y_prob.tolist(), k=10)
        p_at_k_list.append(p_at_k)
        ndcg_list.append(ndcg)

    # Global metrics
    auc = roc_auc_score(y_true_all, y_prob_all) if len(set(y_true_all)) > 1 else float("nan")
    ll = log_loss(y_true_all, y_prob_all, labels=[0, 1])
    print(f"AUC: {auc:.4f}")
    print(f"LogLoss: {ll:.5f}")
    print(f"Precision@{args.k}: {np.mean(p_at_k_list):.4f}")
    print(f"NDCG@10: {np.mean(ndcg_list):.4f}")


if __name__ == "__main__":
    main()
