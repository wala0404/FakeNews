from fastapi import APIRouter

router = APIRouter()

@router.post("/classify")
async def classify_news(text: dict):
    # Dummy classifier: always returns 'real' with score 0.99
    return {"label": "real", "score": 0.99}
