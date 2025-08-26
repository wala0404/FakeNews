from typing import List, Optional

import numpy as np

from app.state import get_article_vector


def session_vector_from_items(last_item_ids: List[str], weights: Optional[List[float]] = None) -> Optional[np.ndarray]:
    """
    Build a session vector as a weighted mean of recent item vectors.
    """
    if not last_item_ids:
        return None

    vecs = []
    ws = []
    for i, aid in enumerate(last_item_ids):
        v = get_article_vector(aid)
        if v is None:
            continue
        vecs.append(v)
        if weights and i < len(weights):
            ws.append(float(weights[i]))
        else:
            ws.append(1.0)

    if not vecs:
        return None
    V = np.vstack(vecs).astype("float32")
    w = np.asarray(ws, dtype="float32").reshape(-1, 1)
    q = (V * w).sum(axis=0) / (w.sum() + 1e-8)
    # Normalize to unit length (cosine via IP)
    norm = np.linalg.norm(q) + 1e-12
    return (q / norm).astype("float32")


def normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype="float32")
    n = np.linalg.norm(v) + 1e-12
    return (v / n).astype("float32")
