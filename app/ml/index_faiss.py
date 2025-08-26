from typing import Tuple

import numpy as np

from app.state import get_state


def search(query_vec: np.ndarray, topk: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform FAISS search with a single query vector.
    Returns (ids[str], scores[float]) arrays of length <= topk.
    """
    st = get_state()
    if st.index is None:
        raise RuntimeError("FAISS index not loaded")

    # Ensure correct shape and dtype
    q = np.asarray(query_vec, dtype="float32").reshape(1, -1)

    # FAISS expects float32, assume vectors already L2-normalized in index
    D, I = st.index.search(q, topk)
    I = I[0]
    D = D[0]
    # Map indices to article IDs
    ids = np.array([str(st.article_ids[i]) for i in I if i >= 0])
    scores = np.array([float(d) for i, d in zip(I, D) if i >= 0])
    return ids, scores
