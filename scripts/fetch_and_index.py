import argparse
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

from app.ml.embedder import get_embedder


def load_index(out_dir: str, metric: str = "ip"):
    os.makedirs(out_dir, exist_ok=True)
    vec_path = os.path.join(out_dir, "item_vectors.npy")
    ids_path = os.path.join(out_dir, "article_ids.npy")
    idx_path = os.path.join(out_dir, "faiss.index")
    meta_path = os.path.join(out_dir, "meta.json")

    X = np.zeros((0, 384), dtype="float32")
    ids: List[str] = []
    meta: Dict[str, dict] = {}
    index = None

    if os.path.exists(vec_path) and os.path.exists(ids_path) and os.path.exists(idx_path):
        X = np.load(vec_path).astype("float32")
        ids = np.load(ids_path, allow_pickle=True).tolist()
        if faiss is None:
            raise RuntimeError("faiss is required to load the index")
        index = faiss.read_index(idx_path)
    else:
        # Create empty index
        if faiss is None:
            raise RuntimeError("faiss is required to create the index")
        d = X.shape[1] if X.size > 0 else 384
        index = faiss.IndexFlatIP(d) if metric == "ip" else faiss.IndexFlatL2(d)

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    return X, ids, index, meta


def save_index(out_dir: str, X: np.ndarray, ids: List[str], index, meta: Dict[str, dict]):
    np.save(os.path.join(out_dir, "item_vectors.npy"), X.astype("float32"))
    np.save(os.path.join(out_dir, "article_ids.npy"), np.array(ids, dtype=object))
    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def fetch_news_from_api(api_key: str, page_size: int = 50) -> List[dict]:
    # Placeholder: implement NewsAPI or any provider
    # If no key, return empty list to keep worker running
    if not api_key:
        return []
    try:
        import importlib
        requests = importlib.import_module("requests")
        url = "https://newsapi.org/v2/top-headlines?language=en&pageSize=%d" % page_size
        headers = {"X-Api-Key": api_key}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        out = []
        for a in data.get("articles", []):
            out.append(
                {
                    "id": a.get("url"),
                    "title": a.get("title"),
                    "content": a.get("description") or "",
                    "lang": "en",
                    "source": (a.get("source") or {}).get("name") or "unknown",
                    "published_at": a.get("publishedAt"),
                    "url": a.get("url"),
                }
            )
        return out
    except Exception:
        return []


def dedup_items(items: List[dict], existing_ids: set) -> List[dict]:
    out = []
    for it in items:
        aid = it.get("id") or it.get("url")
        if not aid:
            # synthesize an id from title+published_at
            aid = f"{it.get('title','')}-{it.get('published_at','')}-{it.get('source','')}"
        if aid in existing_ids:
            continue
        it["id"] = aid
        out.append(it)
    return out


def evict_old(meta: Dict[str, dict], ids: List[str], ttl_days: int) -> Tuple[Dict[str, dict], List[str], set]:
    if ttl_days <= 0:
        return meta, ids, set()
    cutoff = datetime.utcnow() - timedelta(days=ttl_days)
    keep_ids: List[str] = []
    removed: set = set()
    for aid in ids:
        m = meta.get(aid) or {}
        ts = m.get("published_at")
        ok = True
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
                ok = dt >= cutoff
            except Exception:
                ok = True
        if ok:
            keep_ids.append(aid)
        else:
            removed.add(aid)
    # Prune meta
    for aid in list(meta.keys()):
        if aid in removed:
            meta.pop(aid, None)
    return meta, keep_ids, removed


def run_once(out_dir: str, ttl_days: int, api_key: str, batch_size: int = 64, metric: str = "ip", json_file: str = ""):
    X, ids, index, meta = load_index(out_dir, metric=metric)
    id_set = set(ids)

    # Fetch
    fresh: List[dict] = []
    if json_file and os.path.exists(json_file):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "articles" in data:
                data = data["articles"]
            if isinstance(data, list):
                for a in data:
                    src = a.get("source")
                    if isinstance(src, dict):
                        src = src.get("name")
                    fresh.append(
                        {
                            "id": a.get("id") or a.get("url"),
                            "title": a.get("title"),
                            "content": a.get("content") or a.get("description") or "",
                            "lang": a.get("lang") or a.get("language") or "en",
                            "source": src or "unknown",
                            "published_at": a.get("published_at") or a.get("publishedAt"),
                            "url": a.get("url"),
                        }
                    )
        except Exception:
            fresh = []
    else:
        fresh = fetch_news_from_api(api_key, page_size=50)
    if os.environ.get("WORKER_DEBUG"):
        print(f"fresh_pre_dedup={len(fresh)}")
    fresh = dedup_items(fresh, id_set)
    if os.environ.get("WORKER_DEBUG"):
        print(f"fresh_post_dedup={len(fresh)} existing={len(id_set)}")
    if not fresh:
        return False, 0, 0

    # Embed
    titles = [it.get("title") or "" for it in fresh]
    contents = [it.get("content") or "" for it in fresh]
    texts = [f"{t}. {c}".strip() for t, c in zip(titles, contents)]
    embedder = get_embedder(prefer_e5=True)
    V = embedder.encode(texts)
    if metric == "ip":
        try:
            faiss.normalize_L2(V)
        except Exception:
            # already normalized for E5
            pass

    # Append to arrays and index
    start_len = len(ids)
    ids.extend([it["id"] for it in fresh])
    meta.update({it["id"]: {k: it.get(k) for k in ["title", "lang", "source", "published_at", "url"]} for it in fresh})

    if X.size == 0:
        X = V.astype("float32")
        # re-create index with proper dim
        d = X.shape[1]
        index = faiss.IndexFlatIP(d) if metric == "ip" else faiss.IndexFlatL2(d)
        index.add(X)
    else:
        X = np.vstack([X, V]).astype("float32")
        index.add(V.astype("float32"))

    # Evict old
    old_ids = list(ids)
    meta, ids, removed = evict_old(meta, ids, ttl_days)
    if removed:
        # Rebuild arrays and index to ensure vector/id alignment
        id_to_row = {aid: i for i, aid in enumerate(old_ids)}
        rows = [id_to_row[aid] for aid in ids if aid in id_to_row]
        X_keep = X[rows].astype("float32")
        index = faiss.IndexFlatIP(X_keep.shape[1]) if metric == "ip" else faiss.IndexFlatL2(X_keep.shape[1])
        index.add(X_keep)
        X = X_keep

    save_index(out_dir, X, ids, index, meta)
    added = len(ids) - start_len
    return True, added, len(removed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.environ.get("INDEX_DIR", "./models/index"))
    ap.add_argument("--ttl_days", type=int, default=int(os.environ.get("CATALOG_TTL_DAYS", 30)))
    ap.add_argument("--interval", type=int, default=int(os.environ.get("FETCH_INTERVAL_SECS", 180)))
    ap.add_argument("--metric", default="ip", choices=["ip", "l2"])
    ap.add_argument("--one_shot", action="store_true", help="Run once and exit")
    ap.add_argument("--news_api_key", default=os.environ.get("NEWS_API_KEY", ""))
    ap.add_argument("--json_file", default="", help="Optional JSON file with articles for ingestion")
    args = ap.parse_args()

    while True:
        updated, added, removed = run_once(
            args.out_dir, args.ttl_days, args.news_api_key, metric=args.metric, json_file=args.json_file
        )
        ts = datetime.utcnow().isoformat() + "Z"
        print(f"[{ts}] updated={updated} added={added} removed={removed}")
        if args.one_shot:
            break
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    main()
