import os
import json
from functools import lru_cache
from typing import Dict, Optional, List, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    faiss = None  # Allow import even if faiss not installed yet


class _State:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.index = None
        self.item_vectors: Optional[np.ndarray] = None
        self.article_ids: Optional[np.ndarray] = None
        self.id_to_idx: Dict[str, int] = {}
        self.meta: Dict[str, dict] = {}

    def load(self):
        if not self.index_dir:
            raise RuntimeError("INDEX_DIR env var not set")

        # Load vectors and ids
        vec_path = os.path.join(self.index_dir, "item_vectors.npy")
        ids_path = os.path.join(self.index_dir, "article_ids.npy")
        index_path = os.path.join(self.index_dir, "faiss.index")
        meta_path = os.path.join(self.index_dir, "meta.json")

        if not (os.path.exists(vec_path) and os.path.exists(ids_path) and os.path.exists(index_path)):
            raise FileNotFoundError(
                f"Missing index files in {self.index_dir}. Expected item_vectors.npy, article_ids.npy, faiss.index"
            )

        self.item_vectors = np.load(vec_path).astype("float32")
        self.article_ids = np.load(ids_path, allow_pickle=True)
        if self.article_ids.dtype != object:
            # Ensure string dtype
            self.article_ids = self.article_ids.astype(str)
        self.id_to_idx = {str(aid): idx for idx, aid in enumerate(self.article_ids.tolist())}

        if faiss is None:
            raise RuntimeError("faiss is not installed. Please install faiss-cpu.")

        self.index = faiss.read_index(index_path)

        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.meta = json.load(f)
            except Exception:
                self.meta = {}

    def get_meta(self, article_id: str) -> Optional[dict]:
        return self.meta.get(article_id)

    def reload(self):
        """Reload index and metadata without restarting the app."""
        self.load()


@lru_cache(maxsize=1)
def get_state() -> _State:
    index_dir = os.environ.get("INDEX_DIR", "")
    st = _State(index_dir=index_dir)
    st.load()
    return st


def get_article_vector(article_id: str) -> Optional[np.ndarray]:
    st = get_state()
    idx = st.id_to_idx.get(article_id)
    if idx is None or st.item_vectors is None:
        return None
    return st.item_vectors[idx]


def get_all_ids() -> List[str]:
    st = get_state()
    return [str(x) for x in (st.article_ids or [])]
