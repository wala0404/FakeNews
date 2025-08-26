"""
Runtime ranker loader and scorer.

If no ranker is available, functions fall back to identity ordering using retrieval scores.
Models are stored in models/ranker/ as ranker.pkl (sklearn) or ranker.xgb.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

try:
    import xgboost as xgb  # type: ignore  # optional
except Exception:
    xgb = None  # type: ignore


_CACHE: Dict[str, Any] = {"loaded": False, "model": None, "vectorizer": None, "model_type": None}


def _model_paths(base_dir: str) -> Tuple[str, str]:
    return os.path.join(base_dir, "ranker.pkl"), os.path.join(base_dir, "ranker.xgb")


def load_ranker(model_dir: str = os.path.join("models", "ranker")) -> bool:
    """Load a trained ranker if present. Returns True if loaded, False otherwise."""
    if _CACHE.get("loaded"):
        return _CACHE.get("model") is not None
    if joblib is None:
        _CACHE.update({"loaded": True, "model": None})
        return False

    pkl, xg = _model_paths(model_dir)
    path = pkl if os.path.exists(pkl) else (xg if os.path.exists(xg) else None)
    if not path:
        _CACHE.update({"loaded": True, "model": None})
        return False

    try:
        obj = joblib.load(path)
        _CACHE.update(
            {
                "loaded": True,
                "model": obj.get("model"),
                "vectorizer": obj.get("vectorizer"),
                "model_type": obj.get("model_type"),
            }
        )
        return True
    except Exception:
        _CACHE.update({"loaded": True, "model": None})
        return False


def _ensure_loaded():
    if not _CACHE.get("loaded"):
        load_ranker()


def rank(candidates: Sequence[Tuple[str, float]], feature_dicts: Sequence[dict]) -> List[Tuple[str, float]]:
    """Rank candidates using the trained model.

    Inputs
    - candidates: list of (article_id, retrieval_score)
    - feature_dicts: list of feature dicts aligned with candidates

    Output
    - list of (article_id, final_score) sorted desc
    """
    _ensure_loaded()
    model = _CACHE.get("model")
    vec = _CACHE.get("vectorizer")
    model_type = _CACHE.get("model_type")

    if model is None or vec is None:
        # Fallback: return by retrieval score
        return sorted(list(candidates), key=lambda x: x[1], reverse=True)

    try:
        X = vec.transform(feature_dicts)
        if model_type == "xgb":
            if xgb is None:
                return sorted(list(candidates), key=lambda x: x[1], reverse=True)
            dmat = xgb.DMatrix(X)
            probs = model.predict(dmat)
        else:
            probs = model.predict_proba(X)[:, 1]
        out = [(aid, float(s)) for (aid, _), s in zip(candidates, probs)]
        out.sort(key=lambda x: x[1], reverse=True)
        return out
    except Exception:
        # Safety fallback
        return sorted(list(candidates), key=lambda x: x[1], reverse=True)
