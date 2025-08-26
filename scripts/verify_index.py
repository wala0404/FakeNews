import os
import sys
import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    raise SystemExit("faiss is required. Install faiss-cpu.")


def main(index_dir: str):
    vec_path = os.path.join(index_dir, "item_vectors.npy")
    ids_path = os.path.join(index_dir, "article_ids.npy")
    idx_path = os.path.join(index_dir, "faiss.index")

    if not (os.path.exists(vec_path) and os.path.exists(ids_path) and os.path.exists(idx_path)):
        raise SystemExit(f"Missing files in {index_dir}")

    X = np.load(vec_path).astype("float32")
    ids = np.load(ids_path, allow_pickle=True)
    index = faiss.read_index(idx_path)

    # Quick probe: self-similarities of a few items should be top-1
    n = min(5, X.shape[0])
    import random
    rows = random.sample(range(X.shape[0]), n)
    q = X[rows]
    D, I = index.search(q, 1)
    ok = 0
    for r, i in zip(rows, I.flatten().tolist()):
        if r == i:
            ok += 1
    print(f"Probe recall@1 on {n} samples: {ok}/{n}")
    assert ok >= max(1, n - 1), "Low self-recall; check normalization/metric"
    print("Index verified OK.")

    # Random query test
    q = np.random.randn(1, X.shape[1]).astype("float32")
    try:
        faiss.normalize_L2(q)
    except Exception:
        pass
    D, I = index.search(q, 10)
    print("Random query top-10:", [ids[i] for i in I[0] if i >= 0])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/verify_index.py <index_dir>")
        sys.exit(1)
    main(sys.argv[1])
