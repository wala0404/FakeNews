from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import numpy as np

from app.ml.index_faiss import search
from app.ml.session_repr import session_vector_from_items
from app.ml.filters import apply_filters
from app.state import get_state
from app.ml.ranker import rank as rank_candidates, load_ranker


class RecommendRequest(BaseModel):
    session_id: Optional[str] = None
    lang: Optional[str] = None
    k: int = 20
    last_item_ids: Optional[List[str]] = None
    recency_days: Optional[int] = 30
    max_per_source: Optional[int] = 3


router = APIRouter()


@router.post("/recommend")
def recommend(req: RecommendRequest):
    st = get_state()
    # Try to load ranker once (no-op if unavailable)
    try:
        load_ranker()
    except Exception:
        pass

    # Try to build session vector from last_item_ids
    qvec = None
    if req.last_item_ids:
        qvec = session_vector_from_items(req.last_item_ids)

    # Cold-start: use a simple centroid of recent items if no history
    if qvec is None:
        if st.item_vectors is None or st.item_vectors.shape[0] == 0:
            return []
        # Use mean vector over last N items (e.g., 1024) as a fallback query
        tail = st.item_vectors[-min(1024, st.item_vectors.shape[0]) :]
        centroid = tail.mean(axis=0).astype("float32")
        n = np.linalg.norm(centroid) + 1e-12
        qvec = (centroid / n).astype("float32")

    ids, scores = search(qvec, topk=max(500, req.k))

    # Apply filters
    pairs = apply_filters(
        ids,
        scores,
        lang=req.lang,
    recency_days=req.recency_days,
    max_per_source=req.max_per_source,
        exclude_seen=set(req.last_item_ids or []),
        k=req.k,
    )

    # Optional Phase 4 re-ranking
    # Build simple feature dicts aligned with pairs
    meta = st.meta or {}
    feature_dicts = []
    cand_list = []
    for aid, sc in pairs:
        m = meta.get(aid) or {}
        # recency_days from published_at when available; else 7.0
        recency = 7.0
        if m.get("published_at"):
            try:
                from datetime import datetime, timezone

                ts = datetime.fromisoformat(m["published_at"].replace("Z", "+00:00"))
                recency = max(0.0, (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0)
            except Exception:
                pass
        src = m.get("source")
        if isinstance(src, dict):
            src = src.get("name") or "unknown"
        if not isinstance(src, str):
            src = str(src or "unknown")
        category = m.get("category") or "unknown"
        lang = m.get("lang") or (req.lang or "unknown")
        feats = {
            "cosine": float(sc),
            "recency_days": float(recency),
            "popularity": float(m.get("click_count", 0) or 0.0),
            "source": src,
            "category": category,
            "lang": lang,
        }
        feature_dicts.append(feats)
        cand_list.append((aid, float(sc)))

    try:
        reranked = rank_candidates(cand_list, feature_dicts)
    except Exception:
        reranked = cand_list

    # Enrich with minimal meta when available
    items = []
    for aid, sc in reranked:
        m = meta.get(aid) or {}
        items.append(
            {
                "article_id": aid,
                "score": sc,
                "title": m.get("title") or aid,
                "content": m.get("content", ""),
                "url": m.get("url"),
                "published_at": m.get("published_at")
            }
        )

    return items


@router.get("/recommend")
def recommend_get():
    """Cold-start GET for frontend compatibility: returns array items."""
    st = get_state()
    if st.item_vectors is None or st.item_vectors.shape[0] == 0:
        return []
    tail = st.item_vectors[-min(1024, st.item_vectors.shape[0]) :]
    centroid = tail.mean(axis=0).astype("float32")
    n = np.linalg.norm(centroid) + 1e-12
    qvec = (centroid / n).astype("float32")

    ids, scores = search(qvec, topk=200)
    pairs = apply_filters(ids, scores, recency_days=30, max_per_source=3, k=20)

    items = []
    meta = st.meta or {}
    for aid, sc in pairs:
        m = meta.get(aid) or {}
        items.append({
            "article_id": aid,
            "score": sc,
            "title": m.get("title") or aid,
            "content": m.get("content", ""),
            "url": m.get("url"),
            "published_at": m.get("published_at")
        })
    return items

