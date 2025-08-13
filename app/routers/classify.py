from fastapi import APIRouter, HTTPException
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/classify")
async def classify_news(text: str):
    try:
        # Replace this with your actual classification logic
        # This is just a mock implementation
        if not text.strip():
            raise HTTPException(status_code=400, detail="Empty text provided")

        # Mock response - replace with your ML model
        return {
            "label": "REAL" if len(text) > 50 else "FAKE",
            "score": 0.95 if len(text) > 50 else 0.35
        }

    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Classification error")