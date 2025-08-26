import argparse
import json
import os
from typing import List

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    raise SystemExit("faiss is required. Install faiss-cpu.")


def read_ids(path: str) -> List[str]:
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    raise ValueError("Unsupported ids file; expected .txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Directory with news_embs.npy and news_ids.txt")
    ap.add_argument("--out_dir", required=True, help="Output directory for FAISS index and arrays")
    ap.add_argument("--metric", default="ip", choices=["ip", "l2"], help="Similarity metric (default ip)")
    ap.add_argument("--meta", help="Optional JSON metadata file with article_id → meta")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    embs_path = os.path.join(args.data_dir, "news_embs.npy")
    ids_path = os.path.join(args.data_dir, "news_ids.txt")
    if not (os.path.exists(embs_path) and os.path.exists(ids_path)):
        raise SystemExit(f"Missing inputs in {args.data_dir}: news_embs.npy, news_ids.txt")

    X = np.load(embs_path).astype("float32")
    ids = read_ids(ids_path)
    if len(ids) != X.shape[0]:
        raise SystemExit(f"Rows mismatch: {len(ids)} ids vs {X.shape[0]} vectors")

    # Normalize for IP (cosine)
    if args.metric == "ip":
        faiss.normalize_L2(X)

    d = X.shape[1]
    index = faiss.IndexFlatIP(d) if args.metric == "ip" else faiss.IndexFlatL2(d)
    index.add(X)

    # Save artifacts
    np.save(os.path.join(args.out_dir, "item_vectors.npy"), X)
    np.save(os.path.join(args.out_dir, "article_ids.npy"), np.array(ids, dtype=object))
    faiss.write_index(index, os.path.join(args.out_dir, "faiss.index"))

    # Minimal meta stub; real worker will populate
    meta_path = os.path.join(args.out_dir, "meta.json")
    if not os.path.exists(meta_path):
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({}, f)

    # If external meta is provided, overwrite meta.json with it
    if getattr(args, "meta", None) and os.path.exists(args.meta):
        try:
            with open(args.meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: failed to apply --meta file {args.meta}: {e}")

    print("Index built:")
    print("  vectors:", X.shape)
    print("  ids:", len(ids))
    print("  out_dir:", args.out_dir)


if __name__ == "__main__":
    main()
