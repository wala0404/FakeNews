from fastapi import APIRouter, HTTPException
from app.schemas import NewsClassifyRequest, NewsClassifyResponse
from app.ml.infer import classify_news as infer_classify
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/classify", response_model=NewsClassifyResponse)
async def classify_news(request: NewsClassifyRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty text provided")
        label, score = infer_classify(request.text)
        return NewsClassifyResponse(label=label, score=score)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Classification error")