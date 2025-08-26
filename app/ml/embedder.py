"""
Embedding backends for the transient catalog worker.

Primary: intfloat/multilingual-e5-small (384 dims) if sentence-transformers is available.
Fallback: deterministic hashing-based embedding to 384 dims to keep pipeline runnable offline.
"""
from __future__ import annotations

from typing import Iterable, List, Optional
import numpy as np


class BaseEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        raise NotImplementedError


class HashingEmbedder(BaseEmbedder):
    """Very simple deterministic hashing-based embedding.

    Not semantically meaningful, but stable and fast for local/dev runs.
    """

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        dim = self.dim
        out = []
        for t in texts:
            v = np.zeros(dim, dtype="float32")
            # Use a basic character hashing scheme
            for ch in (t or ""):
                h = (ord(ch) * 1315423911) & 0xFFFFFFFF
                idx = h % dim
                v[idx] += 1.0
            # L2 norm
            n = np.linalg.norm(v) + 1e-12
            out.append((v / n).astype("float32"))
        return np.vstack(out) if out else np.zeros((0, dim), dtype="float32")


class E5Embedder(BaseEmbedder):
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", device: Optional[str] = None):
        super().__init__(dim=384)
        import importlib
        st_mod = importlib.import_module("sentence_transformers")
        SentenceTransformer = getattr(st_mod, "SentenceTransformer")
        self.model = SentenceTransformer(model_name, device=device)
        # infer dim from model
        try:
            test = self.model.encode(["passage: test"], normalize_embeddings=True)
            self.dim = int(test.shape[1])
        except Exception:
            self.dim = 384

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        prefixed = [f"passage: {t}" for t in texts]
        X = self.model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)
        return X.astype("float32")


def get_embedder(prefer_e5: bool = True) -> BaseEmbedder:
    if prefer_e5:
        try:
            return E5Embedder()
        except Exception:
            pass
    return HashingEmbedder(dim=384)
