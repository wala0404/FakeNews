from fastapi import APIRouter
from app.schemas import NewsRecommendResponse, NewsArticle

router = APIRouter()

@router.get("/recommend", response_model=NewsRecommendResponse)
def recommend():
    # Dummy data for now
    articles = [
        NewsArticle(title="AI Revolutionizes News", content="AI is changing the news industry...", url="#"),
        NewsArticle(title="Climate Change Update", content="Latest on climate change...", url="#"),
    ]
    return NewsRecommendResponse(articles=articles)

