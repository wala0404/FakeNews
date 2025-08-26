from typing import Iterable, List, Optional, Tuple
from datetime import datetime, timedelta

from app.state import get_state


def apply_filters(
    ids: Iterable[str],
    scores: Iterable[float],
    lang: Optional[str] = None,
    recency_days: Optional[int] = None,
    max_per_source: Optional[int] = None,
    exclude_seen: Optional[Iterable[str]] = None,
    k: int = 20,
) -> List[Tuple[str, float]]:
    st = get_state()
    seen = set(exclude_seen or [])
    per_source = {}

    cutoff = None
    if recency_days is not None and recency_days > 0:
        cutoff = datetime.utcnow() - timedelta(days=recency_days)

    results: List[Tuple[str, float]] = []
    for aid, sc in zip(ids, scores):
        if aid in seen:
            continue

        meta = st.get_meta(aid) or {}

        if lang and meta.get("lang") and meta.get("lang") != lang:
            continue

        if cutoff is not None and meta.get("published_at"):
            try:
                # Expect ISO8601
                ts = datetime.fromisoformat(meta["published_at"].replace("Z", "+00:00")).replace(tzinfo=None)
                if ts < cutoff:
                    continue
            except Exception:
                pass

        if max_per_source is not None and max_per_source > 0:
            src = meta.get("source") or "unknown"
            # Ensure source is a hashable string for per-source bucketing
            if isinstance(src, dict):
                src = src.get("name") or "unknown"
            if not isinstance(src, str):
                src = str(src)
            cnt = per_source.get(src, 0)
            if cnt >= max_per_source:
                continue
            # Increment count using a clearer pattern
            per_source[src] = per_source.get(src, 0) + 1

        # TODO: Popularity adjustment placeholder
        # In the future, we can modify the score `sc` using popularity signals
        # such as click counts stored in meta, e.g.:
        # pop = float(meta.get("click_count", 0))
        # sc = sc + alpha * log(1 + pop)

        results.append((aid, float(sc)))
        if len(results) >= k:
            break

    return results
